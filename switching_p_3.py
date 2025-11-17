#cspell:disable
"""
PAPER 3: Trading Bull and Bear Markets with HMM
Estrategia de Regime Switching (Donninger, 2017)

Lógica:
- Bear market (risk > 0.9): Switch a Treasuries (TLT)
- Transition (0.7 < risk < 0.9): Stocks sin leverage
- Bull market (risk < 0.7): Stocks con leverage opcional
"""

import numpy as np
from typing import Dict
from enum import Enum


class Asset(Enum):
    """Activos disponibles"""
    STOCK = "stock"
    TREASURY = "treasury"
    LEVERAGED_STOCK = "leveraged_stock"


class Paper3Strategy:
    """
    Estrategia Paper 3: Regime Switching
    
    Basada en el paper de Donninger (2017) sobre trading con HMM.
    Switch entre equities y treasuries según régimen de mercado.
    """
    
    def __init__(self, 
                 bear_threshold: float = 0.90,
                 bull_threshold: float = 0.70,
                 use_leverage: bool = False,
                 leverage_threshold: float = 0.50):
        """
        Args:
            bear_threshold: Probabilidad de riesgo para considerar bear market
            bull_threshold: Probabilidad de riesgo bajo la cual es bull market
            use_leverage: Si usar leverage (2x) en bull markets fuertes
            leverage_threshold: Umbral de riesgo para activar leverage
        """
        self.bear_threshold = bear_threshold
        self.bull_threshold = bull_threshold
        self.use_leverage = use_leverage
        self.leverage_threshold = leverage_threshold
        
        # Estado actual
        self.current_asset = Asset.STOCK
        
    def calculate_position(self, signals: Dict) -> Dict:
        """
        Decide qué activo mantener según el régimen de mercado
        
        Returns:
            Dict con: target_asset, target_size, action, reasoning
        """
        risky_prob = signals['risky_probability']
        regime = signals['regime']
        
        reasoning = []
        action = 'HOLD'
        target_asset = self.current_asset
        target_size = 1.0  # Siempre 100% en el activo elegido
        
        # BEAR MARKET: Proteger con treasuries
        if risky_prob > self.bear_threshold:
            reasoning.append(f"Bear market detected (risk={risky_prob:.2f})")
            target_asset = Asset.TREASURY
            
            if self.current_asset != Asset.TREASURY:
                action = 'SWITCH_TO_TREASURY'
                reasoning.append("Switching to treasuries for protection")
            else:
                action = 'HOLD'
                reasoning.append("Stay in treasuries")
        
        # BULL MARKET FUERTE: Leverage opcional
        elif risky_prob < self.leverage_threshold and self.use_leverage:
            reasoning.append(f"Strong bull market (risk={risky_prob:.2f})")
            target_asset = Asset.LEVERAGED_STOCK
            
            if self.current_asset != Asset.LEVERAGED_STOCK:
                action = 'SWITCH_TO_LEVERAGED'
                reasoning.append("Switching to leveraged stocks")
            else:
                action = 'HOLD'
        
        # BULL MARKET NORMAL
        elif risky_prob < self.bull_threshold:
            reasoning.append(f"Bull market (risk={risky_prob:.2f})")
            target_asset = Asset.STOCK
            
            if self.current_asset != Asset.STOCK:
                action = 'SWITCH_TO_STOCK'
                reasoning.append("Switching to stocks")
            else:
                action = 'HOLD'
        
        # TRANSITION: Mantener stocks pero sin leverage
        else:
            reasoning.append(f"Transition regime (risk={risky_prob:.2f})")
            target_asset = Asset.STOCK
            
            if self.current_asset == Asset.LEVERAGED_STOCK:
                action = 'REDUCE_LEVERAGE'
                reasoning.append("Reducing leverage during transition")
            elif self.current_asset == Asset.TREASURY:
                action = 'SWITCH_TO_STOCK'
                reasoning.append("Cautiously moving back to stocks")
            else:
                action = 'HOLD'
        
        return {
            'target_asset': target_asset,
            'target_size': target_size,
            'action': action,
            'reasoning': reasoning,
            'regime': regime,
            'risk_probability': risky_prob
        }
    
    def execute_decision(self, decision: Dict, current_price: float):
        """Actualiza estado interno después de ejecutar decisión"""
        self.current_asset = decision['target_asset']