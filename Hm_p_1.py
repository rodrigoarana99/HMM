#cspell:disable
"""
PAPER 1: Determining Stock Trend Using Hidden Markov Model
Estrategia HMM Básica (Ahuja & Eksombatchai, 2012)

Lógica:
- 3 estados: Bull, Sideways, Bear
- Comprar cuando μ - σ > 0 (estado es "good")
- Cerrar cuando μ - σ < 0 (estado es "bad")
- No permite shorts, solo long o cash
"""

import numpy as np
from typing import Dict
from enum import Enum


class MarketState(Enum):
    """Estados del mercado según HMM"""
    BULL = "bull"
    SIDEWAYS = "sideways"  
    BEAR = "bear"


class Paper1Strategy:
    """
    Estrategia Paper 1: HMM Básico de 3 Estados
    
    Basada en el paper de Ahuja & Eksombatchai (2012).
    Usa HMM de 3 estados para determinar tendencia del mercado.
    """
    
    def __init__(self, 
                 confidence_threshold: float = 0.80,
                 position_size: float = 1.0):
        """
        Args:
            confidence_threshold: Probabilidad mínima para confiar en el estado
            position_size: Tamaño de posición (1.0 = 100%)
        """
        self.confidence_threshold = confidence_threshold
        self.position_size = position_size
        
        # Estado actual
        self.is_invested = False
        self.current_state = None
        
    def calculate_position(self, signals: Dict) -> Dict:
        """
        Decide si estar long o en cash según el estado HMM
        
        Del Paper 1:
        - Si μ - σ > 0 → Estado "good" → Comprar
        - Si μ - σ < 0 → Estado "bad" → Cash
        
        Returns:
            Dict con: target_position, target_size, action, reasoning
        """
        regime = signals['regime']
        state_probs = signals.get('state_probabilities', None)
        risky_prob = signals['risky_probability']
        
        # Determinar confianza en el estado actual
        max_prob = 1.0 - risky_prob if regime == 'bull' else risky_prob
        
        reasoning = []
        action = 'HOLD'
        target_size = 0.0
        is_good_state = False
        
        # LÓGICA DEL PAPER 1: Estado es "good" si es bull con alta confianza
        if regime == 'bull' and max_prob > self.confidence_threshold:
            is_good_state = True
            reasoning.append(f"Bull state detected with {max_prob:.2%} confidence")
            target_size = self.position_size
            
            if not self.is_invested:
                action = 'BUY'
                reasoning.append(f"Enter long position {target_size:.1%}")
            else:
                action = 'HOLD'
                reasoning.append("Maintain long position")
        
        # Estado "bad" - salir a cash
        elif regime == 'bear' and risky_prob > self.confidence_threshold:
            reasoning.append(f"Bear state detected with {risky_prob:.2%} confidence")
            target_size = 0.0
            
            if self.is_invested:
                action = 'CLOSE'
                reasoning.append("Exit to cash - bear market")
            else:
                action = 'HOLD'
                reasoning.append("Stay in cash")
        
        # Sideways o baja confianza - mantener posición actual
        else:
            reasoning.append(f"Uncertain state: {regime} (confidence={max_prob:.2%})")
            
            if self.is_invested:
                target_size = self.position_size
                action = 'HOLD'
                reasoning.append("Maintain current position - uncertain")
            else:
                target_size = 0.0
                action = 'HOLD'
                reasoning.append("Stay in cash - uncertain")
        
        return {
            'is_good_state': is_good_state,
            'target_size': target_size,
            'action': action,
            'reasoning': reasoning,
            'regime': regime,
            'confidence': max_prob
        }
    
    def execute_decision(self, decision: Dict, current_price: float):
        """Actualiza estado interno"""
        self.is_invested = (decision['target_size'] > 0)
        self.current_state = decision['regime']