#cspell:disable
"""
PAPER 2: Modeling Momentum and Reversals
Estrategia de Mean Reversion (Stein & Pozharny, 2018)

Lógica:
- Stocks revierten a un "fair value" (nivel de industria)
- "Buy losers" cuando precio < fair value
- "Sell winners" cuando precio > fair value
- Trading activo basado en desviación del fair value
"""

import numpy as np
from typing import Dict
from enum import Enum


class Position(Enum):
    """Posiciones posibles"""
    LONG = "long"
    NEUTRAL = "neutral"


class Paper2Strategy:
    """
    Estrategia Paper 2: Mean Reversion Trading
    
    Basada en el paper de Stein & Pozharny (2018).
    Explota reversiones de corto plazo comprando "losers".
    """
    
    def __init__(self, 
                 entry_threshold: float = -0.3,
                 exit_threshold: float = 0.3,
                 position_size: float = 1.0,
                 use_momentum_filter: bool = True):
        """
        Args:
            entry_threshold: Z-score negativo para entrar (precio barato)
            exit_threshold: Z-score positivo para salir (precio caro)
            position_size: Tamaño base de posición
            use_momentum_filter: Si usar momentum como filtro adicional
        """
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.position_size = position_size
        self.use_momentum_filter = use_momentum_filter
        
        # Estado actual
        self.current_position = Position.NEUTRAL
        self.current_size = 0.0
        self.entry_price = None
        
    def calculate_position(self, signals: Dict) -> Dict:
        """
        Trading basado en mean reversion
        
        Del Paper 2:
        - Precio < Fair Value → BUY (reversion al alza esperada)
        - Precio > Fair Value → SELL (reversion a la baja esperada)
        
        Returns:
            Dict con: target_position, target_size, action, reasoning
        """
        mr_signal = signals['mean_reversion']
        mom_signal = signals['momentum']
        
        z_score = mr_signal['z_score']
        deviation = mr_signal['deviation']
        signal = mr_signal['signal']
        
        reasoning = []
        action = 'HOLD'
        target_position = self.current_position
        target_size = self.current_size
        
        # Información de mercado
        current_price = signals['current_price']
        fair_value = signals['fair_value']
        
        reasoning.append(f"Price=${current_price:.2f}, Fair=${fair_value:.2f}")
        reasoning.append(f"Deviation={deviation:.2%}, Z-score={z_score:.2f}")
        
        # SEÑAL DE COMPRA: Precio muy por debajo del fair value
        if z_score < self.entry_threshold:
            reasoning.append(f"Stock is CHEAP (z={z_score:.2f})")
            
            # Filtro de momentum (opcional)
            momentum_ok = True
            if self.use_momentum_filter:
                momentum_ok = mom_signal['signal'] > -0.5
                if not momentum_ok:
                    reasoning.append("BUT momentum is too negative - skip")
            
            if momentum_ok:
                target_position = Position.LONG
                
                # Tamaño variable según intensidad de la señal
                signal_strength = min(abs(signal), 1.0)
                target_size = self.position_size * (0.5 + 0.5 * signal_strength)
                
                if self.current_position != Position.LONG:
                    action = 'BUY'
                    reasoning.append(f"ENTER long {target_size:.1%} - mean reversion")
                elif abs(target_size - self.current_size) > 0.1:
                    action = 'ADJUST'
                    reasoning.append(f"INCREASE position to {target_size:.1%}")
                else:
                    action = 'HOLD'
        
        # SEÑAL DE VENTA: Precio muy por encima del fair value  
        elif z_score > self.exit_threshold:
            reasoning.append(f"Stock is EXPENSIVE (z={z_score:.2f})")
            
            if self.current_position == Position.LONG:
                target_position = Position.NEUTRAL
                target_size = 0.0
                action = 'CLOSE'
                reasoning.append("EXIT position - overvalued")
            else:
                action = 'HOLD'
                reasoning.append("Stay out - overvalued")
        
        # ZONA NEUTRAL: Cerca del fair value
        else:
            reasoning.append(f"Stock near FAIR VALUE (z={z_score:.2f})")
            
            # Mantener posición existente o posición pequeña
            if self.current_position == Position.LONG:
                # Reducir gradualmente si nos acercamos a overvalued
                if z_score > 0.1:
                    target_size = self.position_size * 0.5
                    if abs(target_size - self.current_size) > 0.1:
                        action = 'ADJUST'
                        reasoning.append("REDUCE position - approaching fair value")
                else:
                    action = 'HOLD'
                    reasoning.append("HOLD position - near fair value")
            else:
                # Considerar entrada pequeña si está ligeramente barato
                if z_score < -0.1:
                    target_position = Position.LONG
                    target_size = self.position_size * 0.5
                    action = 'BUY'
                    reasoning.append("SMALL position - slightly cheap")
                else:
                    action = 'HOLD'
                    reasoning.append("Stay out - neutral zone")
        
        return {
            'target_position': target_position,
            'target_size': target_size,
            'action': action,
            'reasoning': reasoning,
            'z_score': z_score,
            'deviation': deviation,
            'signal_strength': abs(signal)
        }
    
    def execute_decision(self, decision: Dict, current_price: float):
        """Actualiza estado interno"""
        self.current_position = decision['target_position']
        self.current_size = decision['target_size']
        
        if self.current_position == Position.LONG and self.entry_price is None:
            self.entry_price = current_price
        elif self.current_position == Position.NEUTRAL:
            self.entry_price = None