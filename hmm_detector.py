"""
Módulo HMM Simplificado - Solo Detección de Régimen
Basado en Paper 3: Donninger (2017) - "Trading Bull and Bear Markets with HMM"

SIMPLIFICADO:
- Solo HMM para detección de régimen
- Eliminadas señales de mean reversion y momentum
- Código optimizado y limpio
"""

import numpy as np
from scipy import stats
from typing import Tuple
import warnings


class HMMRegimeDetector:
    """
    Hidden Markov Model para detectar régimen de mercado
    Usa absolute returns y VIX term structure

    Estados:
    - Estado 0: Bajo riesgo (Bull market)
    - Estado 1: Alto riesgo (Bear market)
    """

    def __init__(self, n_states: int = 2, window_days: int = 252,
                 min_regularization: float = 1e-4):
        """
        Args:
            n_states: Número de estados (2 = bull/bear)
            window_days: Ventana de datos para entrenamiento (252 = 1 año)
            min_regularization: Regularización mínima para covarianzas
        """
        self.n_states = n_states
        self.window_days = window_days
        self.min_regularization = min_regularization

        # Parámetros del modelo (estimados con Baum-Welch)
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
        # Validar y limpiar datos
        returns = np.asarray(returns, dtype=np.float64)
        vix_term_structure = np.asarray(vix_term_structure, dtype=np.float64)

        returns_clean = self._clean_data(returns, "returns")
        vix_clean = self._clean_data(vix_term_structure, "vix_term_structure")

        # Preparar observaciones
        abs_returns = np.abs(returns_clean) * 100 * np.sqrt(252)  # Volatilidad anualizada
        abs_returns = np.clip(abs_returns, 0, 200)  # Max 200% vol
        vix_clean = np.clip(vix_clean, -10, 10)  # Limitar term structure

        observations = np.column_stack([abs_returns, vix_clean])

        # Usar ventana deslizante
        if len(observations) > self.window_days:
            observations = observations[-self.window_days:]

        # Verificación
        if len(observations) < 20:
            raise ValueError(f"Muy pocas observaciones: {len(observations)} < 20")

        if np.any(~np.isfinite(observations)):
            raise ValueError("Observaciones contienen valores no finitos")

        # Inicialización con K-means
        self._initialize_parameters(observations)

        # Ejecutar EM algorithm
        try:
            self._baum_welch(observations, max_iter=10)
        except Exception as e:
            warnings.warn(f"Baum-Welch tuvo problemas: {e}. Usando parámetros iniciales.")

    def _clean_data(self, data: np.ndarray, name: str) -> np.ndarray:
        """Limpia datos removiendo/interpolando NaN e Inf"""
        data = data.copy()

        nan_mask = np.isnan(data)
        inf_mask = np.isinf(data)

        if np.any(nan_mask) or np.any(inf_mask):
            n_bad = np.sum(nan_mask | inf_mask)
            warnings.warn(f"{name}: {n_bad} valores NaN/Inf. Interpolando...")

            data[inf_mask] = np.nan

            if np.all(nan_mask):
                data = np.zeros_like(data)
            else:
                valid_indices = np.where(~nan_mask)[0]
                if len(valid_indices) > 0:
                    data = np.interp(np.arange(len(data)),
                                   valid_indices,
                                   data[valid_indices])

        return data

    def _initialize_parameters(self, observations: np.ndarray):
        """Inicializa parámetros del HMM usando K-means"""
        from sklearn.cluster import KMeans

        try:
            kmeans = KMeans(n_clusters=self.n_states, random_state=42, n_init=10)
            initial_states = kmeans.fit_predict(observations)
        except Exception as e:
            warnings.warn(f"K-means falló: {e}. Usando inicialización aleatoria.")
            initial_states = np.random.randint(0, self.n_states, len(observations))

        self.means = []
        self.covs = []

        for i in range(self.n_states):
            state_obs = observations[initial_states == i]

            if len(state_obs) < 2:
                state_obs = observations[np.random.choice(len(observations),
                                                         min(10, len(observations)),
                                                         replace=False)]

            mean = np.mean(state_obs, axis=0)
            cov = np.cov(state_obs.T)

            # Regularización
            regularization = max(self.min_regularization,
                               np.mean(np.diagonal(cov)) * 0.01)
            cov_reg = cov + np.eye(2) * regularization
            cov_reg = self._ensure_positive_definite(cov_reg)

            self.means.append(mean)
            self.covs.append(cov_reg)

        # Matriz de transición inicial
        self.transition_matrix = np.ones((self.n_states, self.n_states)) / self.n_states
        self.initial_probs = np.ones(self.n_states) / self.n_states

    def _ensure_positive_definite(self, cov: np.ndarray) -> np.ndarray:
        """Asegura que la matriz de covarianza sea definida positiva"""
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        eigenvalues = np.maximum(eigenvalues, self.min_regularization)
        cov_fixed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        cov_fixed = (cov_fixed + cov_fixed.T) / 2  # Asegurar simetría
        return cov_fixed

    def _baum_welch(self, observations: np.ndarray, max_iter: int = 10) -> None:
        """Baum-Welch (EM algorithm) para entrenar HMM"""
        n_obs = len(observations)
        prev_likelihood = -np.inf

        for iteration in range(max_iter):
            try:
                # E-step: Forward-backward
                alpha, scale = self._forward_scaled(observations)
                beta = self._backward_scaled(observations, scale)

                # Gamma (probabilidad de estar en estado i en tiempo t)
                gamma = alpha * beta
                gamma_sum = gamma.sum(axis=1, keepdims=True)
                gamma_sum = np.where(gamma_sum > 0, gamma_sum, 1.0)
                gamma = gamma / gamma_sum

                # Log-likelihood para convergencia
                log_likelihood = np.sum(np.log(scale + 1e-100))

                if abs(log_likelihood - prev_likelihood) < 1e-4:
                    break
                prev_likelihood = log_likelihood

                # Xi (probabilidad de transición i->j)
                xi = np.zeros((n_obs - 1, self.n_states, self.n_states))
                for t in range(n_obs - 1):
                    for i in range(self.n_states):
                        for j in range(self.n_states):
                            emission = self._emission_prob(observations[t + 1], j)
                            xi[t, i, j] = (alpha[t, i] *
                                          self.transition_matrix[i, j] *
                                          emission *
                                          beta[t + 1, j])

                    xi_sum = xi[t].sum()
                    if xi_sum > 0:
                        xi[t] = xi[t] / xi_sum

                # M-step: Actualizar parámetros
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

                        # Regularización
                        reg = max(self.min_regularization,
                                np.mean(np.diagonal(self.covs[i])) * 0.01)
                        self.covs[i] += np.eye(2) * reg
                        self.covs[i] = self._ensure_positive_definite(self.covs[i])

            except Exception as e:
                warnings.warn(f"Iteración {iteration} falló: {e}. Terminando entrenamiento.")
                break

    def _forward_scaled(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward algorithm con scaling"""
        n_obs = len(observations)
        alpha = np.zeros((n_obs, self.n_states))
        scale = np.zeros(n_obs)

        # Inicialización
        for i in range(self.n_states):
            alpha[0, i] = self.initial_probs[i] * self._emission_prob(observations[0], i)

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
                              self._emission_prob(observations[t], j)

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

        beta[-1] = 1.0 / scale[-1] if scale[-1] > 0 else 1.0

        for t in range(n_obs - 2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = sum(self.transition_matrix[i, j] *
                               self._emission_prob(observations[t + 1], j) *
                               beta[t + 1, j]
                               for j in range(self.n_states))

            if scale[t] > 0:
                beta[t] = beta[t] / scale[t]

        return beta

    def _emission_prob(self, observation: np.ndarray, state: int) -> float:
        """Probabilidad de emisión (Gaussian PDF)"""
        try:
            if not np.all(np.isfinite(observation)):
                return 1e-100

            if not np.all(np.isfinite(self.means[state])):
                return 1e-100

            prob = stats.multivariate_normal.pdf(
                observation,
                mean=self.means[state],
                cov=self.covs[state],
                allow_singular=True
            )

            if not np.isfinite(prob):
                return 1e-100

            return max(prob, 1e-100)

        except Exception:
            return 1e-100

    def get_risky_state(self) -> int:
        """
        Identifica cuál estado es el riesgoso:
        - Mayor volatilidad (mayor absolute return)
        - VIX en backwardation (menor VIX term structure)
        """
        if self.means is None or len(self.means) < 2:
            return 0

        risky_state = 0

        for i in range(self.n_states):
            if (self.means[i][0] > self.means[risky_state][0] and
                self.means[i][1] < self.means[risky_state][1]):
                risky_state = i

        return risky_state

    def predict_current_state(self, returns: np.ndarray,
                            vix_term_structure: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Predice el estado actual y probabilidad de riesgo

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
            state_probs = state_probs / state_probs.sum()
        except Exception as e:
            warnings.warn(f"Error en predict: {e}. Usando probabilidades uniformes.")
            state_probs = np.ones(self.n_states) / self.n_states

        risky_state = self.get_risky_state()
        risky_prob = state_probs[risky_state]

        return state_probs, risky_prob


def calculate_vix_term_structure(vix_f1: np.ndarray, vix_f2: np.ndarray) -> np.ndarray:
    """
    Calcula VIX term structure (F2 - F1)
    Positivo = contango (normal, mercado tranquilo)
    Negativo = backwardation (miedo, mercado riesgoso)
    """
    return vix_f2 - vix_f1
