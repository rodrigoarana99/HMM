"""
Módulo de Señales - Integración de HMM y Mean Reversion (VERSIÓN CORREGIDA)
Combina ideas de los 3 papers para generar señales de trading
"""

import numpy as np
from scipy import stats
from typing import Tuple, Dict, Optional
import pandas as pd
import warnings


class HMMRegimeDetector:
    """
    Hidden Markov Model para detectar régimen de mercado (Paper 3)
    Usa absolute returns y VIX term structure
    
    MEJORAS:
    - Validación robusta de datos
    - Mejor regularización numérica
    - Manejo de casos extremos
    """
    
    def __init__(self, n_states: int = 2, window_days: int = 252,
                 min_regularization: float = 1e-4):
        """
        Args:
            n_states: Número de estados (2 = bull/bear)
            window_days: Ventana de datos para entrenamiento (252 = 1 año trading, antes 504)
            min_regularization: Regularización mínima para covarianzas
        """
        self.n_states = n_states
        self.window_days = window_days
        self.min_regularization = min_regularization
        
        # Parámetros del modelo (se estiman con Baum-Welch)
        self.means = None
        self.covs = None
        self.transition_matrix = None
        self.initial_probs = None
        
    def fit(self, returns: np.ndarray, vix_term_structure: np.ndarray) -> None:
        """
        Entrena el HMM usando EM algorithm (Baum-Welch)
        
        Args:
            returns: Array de returns diarios
            vix_term_structure: VIX F2 - VIX F1 (contango/backwardation)
        """
        # VALIDACIÓN ROBUSTA DE DATOS
        returns = np.asarray(returns, dtype=np.float64)
        vix_term_structure = np.asarray(vix_term_structure, dtype=np.float64)
        
        # Limpiar NaN e Inf
        returns_clean = self._clean_data(returns, "returns")
        vix_clean = self._clean_data(vix_term_structure, "vix_term_structure")
        
        # Preparar observaciones
        abs_returns = np.abs(returns_clean) * 100 * np.sqrt(252)
        
        # Limitar valores extremos (outliers)
        abs_returns = np.clip(abs_returns, 0, 200)  # Max 200% volatilidad anualizada
        vix_clean = np.clip(vix_clean, -10, 10)  # Limitar term structure
        
        observations = np.column_stack([abs_returns, vix_clean])
        
        # Usar ventana deslizante
        if len(observations) > self.window_days:
            observations = observations[-self.window_days:]
        
        # Verificación final
        if len(observations) < 20:
            raise ValueError(f"Muy pocas observaciones: {len(observations)} < 20")
        
        if np.any(~np.isfinite(observations)):
            raise ValueError("Observaciones contienen valores no finitos después de limpieza")
        
        # Inicialización con K-means
        from sklearn.cluster import KMeans
        try:
            kmeans = KMeans(n_clusters=self.n_states, random_state=42, n_init=10)
            initial_states = kmeans.fit_predict(observations)
        except Exception as e:
            warnings.warn(f"K-means falló: {e}. Usando inicialización aleatoria.")
            initial_states = np.random.randint(0, self.n_states, len(observations))
        
        # Estimar parámetros iniciales con regularización fuerte
        self.means = []
        self.covs = []
        
        for i in range(self.n_states):
            state_obs = observations[initial_states == i]
            
            if len(state_obs) < 2:
                # Si hay muy pocas observaciones, usar medias globales
                state_obs = observations[np.random.choice(len(observations), 
                                                         min(10, len(observations)), 
                                                         replace=False)]
            
            mean = np.mean(state_obs, axis=0)
            cov = np.cov(state_obs.T)
            
            # Regularización adaptativa
            regularization = max(self.min_regularization, 
                               np.mean(np.diagonal(cov)) * 0.01)
            cov_reg = cov + np.eye(2) * regularization
            
            # Asegurar que la matriz es definida positiva
            cov_reg = self._ensure_positive_definite(cov_reg)
            
            self.means.append(mean)
            self.covs.append(cov_reg)
        
        # Matriz de transición inicial (uniforme)
        self.transition_matrix = np.ones((self.n_states, self.n_states)) / self.n_states
        self.initial_probs = np.ones(self.n_states) / self.n_states
        
        # Ejecutar EM algorithm
        try:
            # CORREGIDO: Reducir iteraciones de 50 → 10 para velocidad
            self._baum_welch(observations, max_iter=10)
        except Exception as e:
            warnings.warn(f"Baum-Welch tuvo problemas: {e}. Usando parámetros iniciales.")
    
    def _clean_data(self, data: np.ndarray, name: str) -> np.ndarray:
        """Limpia datos removiendo/interpolando NaN e Inf"""
        data = data.copy()
        
        # Detectar problemas
        nan_mask = np.isnan(data)
        inf_mask = np.isinf(data)
        
        if np.any(nan_mask) or np.any(inf_mask):
            n_bad = np.sum(nan_mask | inf_mask)
            warnings.warn(f"{name}: {n_bad} valores NaN/Inf detectados. Interpolando...")
            
            # Reemplazar Inf con NaN para interpolación consistente
            data[inf_mask] = np.nan
            
            # Interpolación lineal
            if np.all(nan_mask):
                # Todos son NaN, usar zeros
                data = np.zeros_like(data)
            else:
                # Interpolación
                valid_indices = np.where(~nan_mask)[0]
                if len(valid_indices) > 0:
                    data = np.interp(np.arange(len(data)), 
                                   valid_indices, 
                                   data[valid_indices])
        
        return data
    
    def _ensure_positive_definite(self, cov: np.ndarray) -> np.ndarray:
        """Asegura que la matriz de covarianza sea definida positiva"""
        # Calcular eigenvalores
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Si hay eigenvalues negativos o muy pequeños, corregir
        min_eigenvalue = self.min_regularization
        eigenvalues = np.maximum(eigenvalues, min_eigenvalue)
        
        # Reconstruir matriz
        cov_fixed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        
        # Asegurar simetría
        cov_fixed = (cov_fixed + cov_fixed.T) / 2
        
        return cov_fixed
    
    def _baum_welch(self, observations: np.ndarray, max_iter: int = 50) -> None:
        """
        Implementación mejorada de Baum-Welch (EM algorithm)
        """
        n_obs = len(observations)
        prev_likelihood = -np.inf
        
        for iteration in range(max_iter):
            try:
                # E-step: Forward-backward algorithm
                alpha, scale = self._forward_scaled(observations)
                beta = self._backward_scaled(observations, scale)
                
                # Calcular gamma (probabilidad de estar en estado i en tiempo t)
                gamma = alpha * beta
                gamma_sum = gamma.sum(axis=1, keepdims=True)
                gamma_sum = np.where(gamma_sum > 0, gamma_sum, 1.0)  # Evitar división por 0
                gamma = gamma / gamma_sum
                
                # Calcular log-likelihood para convergencia
                log_likelihood = np.sum(np.log(scale + 1e-100))
                
                # Verificar convergencia
                if abs(log_likelihood - prev_likelihood) < 1e-4:
                    break
                prev_likelihood = log_likelihood
                
                # Calcular xi (probabilidad de transición i->j en tiempo t)
                xi = np.zeros((n_obs - 1, self.n_states, self.n_states))
                for t in range(n_obs - 1):
                    for i in range(self.n_states):
                        for j in range(self.n_states):
                            emission = self._emission_prob_safe(observations[t + 1], j)
                            xi[t, i, j] = (alpha[t, i] * 
                                          self.transition_matrix[i, j] * 
                                          emission * 
                                          beta[t + 1, j])
                    
                    xi_sum = xi[t].sum()
                    if xi_sum > 0:
                        xi[t] = xi[t] / xi_sum
                
                # M-step: Actualizar parámetros
                # Actualizar probabilidades iniciales
                self.initial_probs = gamma[0]
                self.initial_probs = self.initial_probs / self.initial_probs.sum()
                
                # Actualizar matriz de transición
                for i in range(self.n_states):
                    denominator = gamma[:-1, i].sum()
                    if denominator > 0:
                        for j in range(self.n_states):
                            self.transition_matrix[i, j] = xi[:, i, j].sum() / denominator
                    else:
                        self.transition_matrix[i, :] = 1.0 / self.n_states
                
                # Normalizar filas
                row_sums = self.transition_matrix.sum(axis=1, keepdims=True)
                self.transition_matrix = self.transition_matrix / row_sums
                
                # Actualizar medias y covarianzas
                for i in range(self.n_states):
                    weight = gamma[:, i]
                    weight_sum = weight.sum()
                    
                    if weight_sum > 1e-10:
                        self.means[i] = np.average(observations, axis=0, weights=weight)
                        diff = observations - self.means[i]
                        self.covs[i] = np.dot(weight * diff.T, diff) / weight_sum
                        
                        # Regularización adaptativa
                        reg = max(self.min_regularization,
                                np.mean(np.diagonal(self.covs[i])) * 0.01)
                        self.covs[i] += np.eye(2) * reg
                        
                        # Asegurar definida positiva
                        self.covs[i] = self._ensure_positive_definite(self.covs[i])
                
            except Exception as e:
                warnings.warn(f"Iteración {iteration} falló: {e}. Terminando entrenamiento.")
                break
    
    def _forward_scaled(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward algorithm con scaling para estabilidad numérica"""
        n_obs = len(observations)
        alpha = np.zeros((n_obs, self.n_states))
        scale = np.zeros(n_obs)
        
        # Inicialización
        for i in range(self.n_states):
            alpha[0, i] = self.initial_probs[i] * self._emission_prob_safe(observations[0], i)
        
        scale[0] = alpha[0].sum()
        if scale[0] > 0:
            alpha[0] = alpha[0] / scale[0]
        else:
            alpha[0] = 1.0 / self.n_states
            scale[0] = 1.0
        
        # Recursión
        for t in range(1, n_obs):
            for j in range(self.n_states):
                alpha[t, j] = (alpha[t-1] @ self.transition_matrix[:, j]) * \
                              self._emission_prob_safe(observations[t], j)
            
            scale[t] = alpha[t].sum()
            if scale[t] > 0:
                alpha[t] = alpha[t] / scale[t]
            else:
                alpha[t] = 1.0 / self.n_states
                scale[t] = 1.0
        
        return alpha, scale
    
    def _backward_scaled(self, observations: np.ndarray, scale: np.ndarray) -> np.ndarray:
        """Backward algorithm con scaling"""
        n_obs = len(observations)
        beta = np.zeros((n_obs, self.n_states))
        
        # Inicialización
        beta[-1] = 1.0 / scale[-1] if scale[-1] > 0 else 1.0
        
        # Recursión
        for t in range(n_obs - 2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = sum(self.transition_matrix[i, j] * 
                               self._emission_prob_safe(observations[t + 1], j) * 
                               beta[t + 1, j] 
                               for j in range(self.n_states))
            
            if scale[t] > 0:
                beta[t] = beta[t] / scale[t]
        
        return beta
    
    def _emission_prob_safe(self, observation: np.ndarray, state: int) -> float:
        """
        Probabilidad de emisión (Gaussian) con manejo robusto de errores
        """
        try:
            # Verificar inputs
            if not np.all(np.isfinite(observation)):
                return 1e-100
            
            if not np.all(np.isfinite(self.means[state])):
                return 1e-100
            
            # Calcular PDF
            prob = stats.multivariate_normal.pdf(
                observation, 
                mean=self.means[state], 
                cov=self.covs[state],
                allow_singular=True  # Permitir matrices casi singulares
            )
            
            # Manejar valores muy pequeños/grandes
            if not np.isfinite(prob):
                return 1e-100
            
            return max(prob, 1e-100)  # Evitar exactamente 0
            
        except Exception as e:
            warnings.warn(f"Error en emission_prob: {e}")
            return 1e-100
    
    def _emission_prob(self, observation: np.ndarray, state: int) -> float:
        """Alias para compatibilidad"""
        return self._emission_prob_safe(observation, state)
    
    def get_risky_state(self) -> int:
        """
        Identifica cuál estado es el riesgoso según Paper 3:
        - Mayor volatilidad (mayor mean de absolute return)
        - VIX en backwardation (menor VIX term structure)
        """
        if self.means is None or len(self.means) < 2:
            return 0
        
        risky_state = 0
        
        # El estado con mayor absolute return y menor VIX term structure
        for i in range(self.n_states):
            if (self.means[i][0] > self.means[risky_state][0] and 
                self.means[i][1] < self.means[risky_state][1]):
                risky_state = i
        
        return risky_state
    
    def predict_current_state(self, returns: np.ndarray, 
                            vix_term_structure: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Predice el estado actual y probabilidad de estar en estado riesgoso
        
        Returns:
            state_probs: Probabilidades de cada estado
            risky_prob: Probabilidad de estar en estado riesgoso
        """
        # Limpiar datos
        returns_clean = self._clean_data(returns, "returns")
        vix_clean = self._clean_data(vix_term_structure, "vix")
        
        # Preparar observaciones
        abs_returns = np.abs(returns_clean) * 100 * np.sqrt(252)
        abs_returns = np.clip(abs_returns, 0, 200)
        vix_clean = np.clip(vix_clean, -10, 10)
        
        observations = np.column_stack([abs_returns, vix_clean])
        
        if len(observations) > self.window_days:
            observations = observations[-self.window_days:]
        
        try:
            alpha, scale = self._forward_scaled(observations)
            state_probs = alpha[-1]
            state_probs = state_probs / state_probs.sum()  # Normalizar
        except Exception as e:
            warnings.warn(f"Error en predict: {e}. Usando probabilidades uniformes.")
            state_probs = np.ones(self.n_states) / self.n_states
        
        risky_state = self.get_risky_state()
        risky_prob = state_probs[risky_state]
        
        return state_probs, risky_prob


class MeanReversionSignal:
    """
    Señales de Mean Reversion basadas en Paper 2
    Modela: dSt = a(St - Ct)dt + σStdWt
    donde Ct es el nivel "justo" de la industria/mercado
    """
    
    def __init__(self, reversion_speed: float = 5.0, lookback_days: int = 20):
        """
        Args:
            reversion_speed: Parámetro 'a' de mean reversion (mayor = más rápido)
            lookback_days: Días para calcular nivel medio (REDUCIDO: 60 → 20)
        """
        self.reversion_speed = reversion_speed
        self.lookback_days = lookback_days

    def calculate_fair_value(self, prices: np.ndarray,
                            market_prices: Optional[np.ndarray] = None) -> float:
        """
        Calcula el nivel "justo" Ct al que el precio debería revertir

        CORREGIDO: Usar SMA de 20 días con ajuste por tendencia
        """
        if market_prices is not None:
            return market_prices[-1]
        else:
            # Usar media móvil de 20 días (más responsive)
            sma = np.mean(prices[-self.lookback_days:])
            return sma
    
    def calculate_signal(self, current_price: float, fair_value: float,
                        volatility: float) -> Dict[str, float]:
        """
        Calcula señal de mean reversion

        Del Paper 2: Comprar cuando precio < fair value (reversion al alza esperada)
        La fuerza depende de la desviación y la volatilidad

        Returns:
            Dict con 'signal' (-1 a 1), 'deviation', 'z_score'
        """
        deviation = (current_price - fair_value) / fair_value
        z_score = deviation / (volatility / np.sqrt(252)) if volatility > 0 else 0

        # CORREGIDO: Usar tanh con factor 0.5 para NO saturar
        # Señal positiva cuando precio < fair value (comprar barato)
        # Señal negativa cuando precio > fair value (vender caro)
        # Factor bajo (0.5) permite señales graduales en vez de todo-o-nada
        signal = -np.tanh(z_score * 0.5)  # Factor muy reducido para evitar saturación

        return {
            'signal': signal,
            'deviation': deviation,
            'z_score': z_score,
            'fair_value': fair_value,
            'reversion_strength': abs(signal)
        }


class MomentumSignal:
    """
    Señales de Momentum (Paper 1 y Paper 2)
    El momentum aparece a largo plazo cuando todo el mercado se mueve
    """
    
    def __init__(self, short_window: int = 20, long_window: int = 60):
        self.short_window = short_window
        self.long_window = long_window
    
    def calculate_signal(self, prices: np.ndarray) -> Dict[str, float]:
        """
        Calcula señal de momentum usando moving averages

        Returns:
            Dict con 'signal', 'short_ma', 'long_ma'
        """
        short_ma = np.mean(prices[-self.short_window:])
        long_ma = np.mean(prices[-self.long_window:])

        # Señal basada en cruce de medias
        signal = (short_ma - long_ma) / long_ma if long_ma > 0 else 0
        # CORREGIDO: Factor reducido de 10 → 3 para saturar MUCHO menos
        signal = np.tanh(signal * 3)  # Normalizar con factor bajo para gradualidad

        return {
            'signal': signal,
            'short_ma': short_ma,
            'long_ma': long_ma,
            'strength': abs(signal)
        }


class IntegratedSignalGenerator:
    """
    Combina todas las señales según el régimen de mercado
    """
    
    def __init__(self):
        self.hmm = HMMRegimeDetector(n_states=2)
        self.mean_reversion = MeanReversionSignal(reversion_speed=5.0)
        self.momentum = MomentumSignal()
        
    def generate_signals(self, 
                        prices: np.ndarray,
                        returns: np.ndarray,
                        vix_term_structure: np.ndarray,
                        market_prices: Optional[np.ndarray] = None) -> Dict:
        """
        Genera todas las señales integradas
        
        Lógica de Paper 3: En bear markets (high risk), proteger
        Lógica de Paper 2: En bull markets, usar mean reversion para timing
        """
        try:
            # 1. Detectar régimen con HMM
            self.hmm.fit(returns, vix_term_structure)
            state_probs, risky_prob = self.hmm.predict_current_state(returns, vix_term_structure)
            
            # 2. Calcular señales individuales
            current_price = prices[-1]
            volatility = np.std(returns[-60:]) if len(returns) >= 60 else np.std(returns)
            
            fair_value = self.mean_reversion.calculate_fair_value(prices, market_prices)
            mr_signal = self.mean_reversion.calculate_signal(current_price, fair_value, volatility)
            mom_signal = self.momentum.calculate_signal(prices)
            
            # 3. Combinar señales según régimen
            if risky_prob > 0.90:
                # Bear market: PROTEGER (Paper 3)
                regime = 'bear'
                final_signal = -1.0
                confidence = risky_prob
            elif risky_prob > 0.65:
                # Transición: CAUTELOSO - dar más peso a mean reversion
                regime = 'transition'
                final_signal = mr_signal['signal'] * 0.7 + mom_signal['signal'] * 0.3
                confidence = 0.5
            else:
                # Bull market: USAR MEAN REVERSION y MOMENTUM (Paper 2)
                # CORREGIDO: 70% mean reversion + 30% momentum
                # Mean reversion es más confiable para timing, momentum confirma tendencia
                regime = 'bull'
                final_signal = mr_signal['signal'] * 0.7 + mom_signal['signal'] * 0.3
                confidence = 1 - risky_prob
            
            return {
                'regime': regime,
                'risky_probability': risky_prob,
                'state_probabilities': state_probs,
                'final_signal': final_signal,
                'confidence': confidence,
                'mean_reversion': mr_signal,
                'momentum': mom_signal,
                'fair_value': fair_value,
                'current_price': current_price
            }
            
        except Exception as e:
            warnings.warn(f"Error generando señales: {e}")
            # Retornar señales neutras en caso de error
            return {
                'regime': 'unknown',
                'risky_probability': 0.5,
                'state_probabilities': np.array([0.5, 0.5]),
                'final_signal': 0.0,
                'confidence': 0.0,
                'mean_reversion': {'signal': 0.0, 'deviation': 0.0, 'z_score': 0.0, 
                                  'fair_value': prices[-1], 'reversion_strength': 0.0},
                'momentum': {'signal': 0.0, 'short_ma': prices[-1], 'long_ma': prices[-1], 
                           'strength': 0.0},
                'fair_value': prices[-1],
                'current_price': prices[-1]
            }


def calculate_vix_term_structure(vix_f1: np.ndarray, vix_f2: np.ndarray) -> np.ndarray:
    """
    Calcula VIX term structure (F2 - F1)
    Positivo = contango (normal)
    Negativo = backwardation (miedo)
    """
    return vix_f2 - vix_f1