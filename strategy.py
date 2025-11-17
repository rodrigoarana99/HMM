#cspell:disable
"""
Módulo de Estrategia - Reglas de Trading
Implementa las estrategias de los papers adaptadas a nuestro contexto
"""

import numpy as np
from typing import Dict, Optional, Tuple
from enum import Enum


class Position(Enum):
    """Estados de posición"""
    LONG = 1
    NEUTRAL = 0
    SHORT = -1


class TradingStrategy:
    """
    Estrategia de trading que integra:
    - HMM para regime detection (Paper 3)
    - Mean reversion para timing (Paper 2)
    - Protection en bear markets (Paper 3)
    """
    
    def __init__(self, 
                 risk_threshold: float = 0.90,
                 position_size: float = 1.0,
                 use_leverage: bool = False,
                 max_leverage: float = 2.0):
        """
        Args:
            risk_threshold: Umbral de probabilidad para considerar bear market
            position_size: Tamaño base de posición (1.0 = 100%)
            use_leverage: Si usar leverage en bull markets fuertes
            max_leverage: Leverage máximo permitido
        """
        self.risk_threshold = risk_threshold
        self.position_size = position_size
        self.use_leverage = use_leverage
        self.max_leverage = max_leverage
        
        # Estado actual
        self.current_position = Position.NEUTRAL
        self.current_size = 0.0
        self.entry_price = None
        
    def calculate_position(self, signals: Dict) -> Dict:
        """
        ESTRATEGIA SIMPLE DEL PAPER 3 (HMM):
        - risk_prob < 0.95: 100% LONG en el activo
        - risk_prob >= 0.95: 0% (CASH o TREASURIES)

        NO usa señales de mean reversion ni momentum.
        Es una estrategia binaria ON/OFF basada SOLO en regime detection.
        """
        risky_prob = signals['risky_probability']
        regime = signals['regime']

        # Inicializar decisión
        target_position = Position.NEUTRAL
        target_size = 0.0
        action = 'HOLD'
        reasoning = []

        # REGLA SIMPLE: Si risk > 0.95, salir. Si risk < 0.95, estar 100% long
        if risky_prob >= self.risk_threshold:
            # BEAR MARKET: Salir completamente (Paper 3)
            target_position = Position.NEUTRAL
            target_size = 0.0

            if self.current_position != Position.NEUTRAL:
                action = 'CLOSE'
                reasoning.append(f"BEAR: Closing position (risk={risky_prob:.2f} >= {self.risk_threshold})")
            else:
                action = 'HOLD'
                reasoning.append(f"BEAR: Stay out (risk={risky_prob:.2f})")

        else:
            # BULL/NORMAL MARKET: Estar 100% long (Paper 3)
            target_position = Position.LONG
            target_size = self.position_size  # 100%

            if self.current_position != Position.LONG:
                action = 'BUY'
                reasoning.append(f"BULL: Enter 100% (risk={risky_prob:.2f} < {self.risk_threshold})")
            else:
                action = 'HOLD'
                reasoning.append(f"BULL: Hold 100% (risk={risky_prob:.2f})")

        return {
            'target_position': target_position,
            'target_size': target_size,
            'action': action,
            'reasoning': reasoning,
            'regime': regime,
            'risk_probability': risky_prob,
            'signal_strength': 1.0 if risky_prob >= self.risk_threshold else 0.0
        }
    
    def _determine_action(self, target_position: Position, 
                         target_size: float) -> str:
        """Determina la acción a tomar"""
        if target_position != self.current_position:
            if target_position == Position.LONG:
                return 'BUY'
            elif target_position == Position.NEUTRAL:
                return 'CLOSE'
            else:
                return 'SHORT'  # Si implementamos shorts
        elif abs(target_size - self.current_size) > 0.1:
            return 'ADJUST'
        else:
            return 'HOLD'
    
    def execute_trade(self, decision: Dict, current_price: float) -> Dict:
        """
        Ejecuta el trade y actualiza estado interno
        
        Returns:
            Dict con detalles de la ejecución
        """
        action = decision['action']
        target_size = decision['target_size']
        
        trade_details = {
            'action': action,
            'executed': False,
            'size': 0.0,
            'price': current_price,
            'old_position': self.current_position,
            'new_position': decision['target_position']
        }
        
        if action in ['BUY', 'CLOSE', 'ADJUST']:
            # Calcular cambio en posición
            size_delta = target_size - self.current_size
            
            trade_details['executed'] = True
            trade_details['size'] = abs(size_delta)
            trade_details['direction'] = 'BUY' if size_delta > 0 else 'SELL'
            
            # Actualizar estado
            self.current_position = decision['target_position']
            self.current_size = target_size
            
            if self.current_position == Position.LONG and self.entry_price is None:
                self.entry_price = current_price
            elif self.current_position == Position.NEUTRAL:
                self.entry_price = None
        
        return trade_details


class MultiAssetStrategy:
    """
    Estrategia para múltiples activos
    Puede rotar entre: Stock, Treasuries, Cash según régimen
    """
    
    def __init__(self, 
                 enable_treasuries: bool = True,
                 enable_leverage_etfs: bool = False):
        """
        Args:
            enable_treasuries: Si rotar a treasuries en bear markets
            enable_leverage_etfs: Si usar ETFs apalancados en bull markets
        """
        self.enable_treasuries = enable_treasuries
        self.enable_leverage_etfs = enable_leverage_etfs
        
        self.base_strategy = TradingStrategy()
        self.current_asset = 'stock'  # 'stock', 'treasury', 'cash'
        
    def calculate_allocation(self, signals: Dict) -> Dict:
        """
        Calcula allocation entre diferentes activos
        
        Strategy del Paper 3:
        - Bull (risk < 0.7): Stock (o leveraged stock)
        - Transition (0.7 < risk < 0.9): Stock sin leverage
        - Bear (risk > 0.9): Treasuries o Cash
        """
        risky_prob = signals['risky_probability']
        final_signal = signals['final_signal']
        
        allocation = {
            'stock': 0.0,
            'stock_leveraged': 0.0,
            'treasury': 0.0,
            'cash': 0.0
        }
        
        reasoning = []
        
        # Bear market: Protection
        if risky_prob > 0.9:
            if self.enable_treasuries:
                allocation['treasury'] = 1.0
                reasoning.append("Bear market: Rotate to treasuries")
                self.current_asset = 'treasury'
            else:
                allocation['cash'] = 1.0
                reasoning.append("Bear market: Move to cash")
                self.current_asset = 'cash'
        
        # Strong bull: Leverage (if enabled)
        elif risky_prob < 0.7 and final_signal > 0.5 and self.enable_leverage_etfs:
            allocation['stock'] = 0.5
            allocation['stock_leveraged'] = 0.5
            reasoning.append("Strong bull: 50% base + 50% leveraged")
            self.current_asset = 'stock'
        
        # Normal bull: Base stock
        elif risky_prob < 0.7:
            allocation['stock'] = 1.0
            reasoning.append("Bull market: Full stock allocation")
            self.current_asset = 'stock'
        
        # Transition: Reduce risk
        else:
            allocation['stock'] = 0.7
            if self.enable_treasuries:
                allocation['treasury'] = 0.3
                reasoning.append("Transition: 70% stock / 30% treasury")
            else:
                allocation['cash'] = 0.3
                reasoning.append("Transition: 70% stock / 30% cash")
            self.current_asset = 'stock'
        
        return {
            'allocation': allocation,
            'primary_asset': self.current_asset,
            'reasoning': reasoning,
            'regime': signals['regime']
        }


class RiskManager:
    """
    Gestión de riesgo y stops
    """
    
    def __init__(self, 
                 max_drawdown: float = 0.2,
                 stop_loss: float = 0.10,
                 trailing_stop: float = 0.05):
        """
        Args:
            max_drawdown: Máximo drawdown permitido antes de reducir posición
            stop_loss: Stop loss desde entry
            trailing_stop: Trailing stop desde peak
        """
        self.max_drawdown = max_drawdown
        self.stop_loss = stop_loss
        self.trailing_stop = trailing_stop
        
        self.peak_value = 0.0
        self.entry_value = 0.0
        
    def check_stops(self, current_value: float, entry_price: float, 
                   current_price: float) -> Tuple[bool, str]:
        """
        Verifica si se debe cerrar posición por stops
        
        Returns:
            (should_close, reason)
        """
        # Actualizar peak
        if current_value > self.peak_value:
            self.peak_value = current_value
        
        # Stop loss desde entrada
        if entry_price is not None:
            loss = (current_price - entry_price) / entry_price
            if loss < -self.stop_loss:
                return True, f"Stop loss hit: {loss:.2%}"
        
        # Trailing stop desde peak
        if self.peak_value > 0:
            drawdown = (self.peak_value - current_value) / self.peak_value
            if drawdown > self.trailing_stop:
                return True, f"Trailing stop hit: {drawdown:.2%}"
        
        # Max drawdown
        if self.entry_value > 0:
            drawdown = (self.entry_value - current_value) / self.entry_value
            if drawdown > self.max_drawdown:
                return True, f"Max drawdown hit: {drawdown:.2%}"
        
        return False, ""
    
    def update_values(self, current_value: float, is_new_position: bool = False):
        """Actualiza valores para tracking"""
        if is_new_position:
            self.entry_value = current_value
            self.peak_value = current_value
        else:
            if current_value > self.peak_value:
                self.peak_value = current_value