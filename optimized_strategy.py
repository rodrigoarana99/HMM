"""
ESTRATEGIA OPTIMIZADA: Regime Switching con HMM
Basada en Paper 3 de Donninger (2017) - "Trading Bull and Bear Markets with HMM"

MEJORAS:
- Umbrales optimizados para mayor actividad
- Detección de régimen más sensible
- Simplificado y enfocado en rentabilidad
"""

import numpy as np
from typing import Dict
from enum import Enum


class Regime(Enum):
    """Régimen de mercado detectado"""
    BULL = "bull"           # Risk < 0.50 → Stocks 100%
    TRANSITION = "transition"  # 0.50 < Risk < 0.75 → Stocks 70% / Cash 30%
    BEAR = "bear"           # Risk > 0.75 → Cash/Treasuries


class OptimizedHMMStrategy:
    """
    Estrategia de Regime Switching Optimizada

    Lógica mejorada:
    - BEAR (risk > 0.75): 100% Cash (protección)
    - TRANSITION (0.50-0.75): 70% Stock, 30% Cash (cautela)
    - BULL (risk < 0.50): 100% Stock (crecimiento)

    Ventajas:
    - Menos whipsaws (cambios frecuentes)
    - Protección en bear markets
    - Participa en bull markets
    - Trading moderado (~8-12 trades/año)
    """

    def __init__(self,
                 bear_threshold: float = 0.75,      # Antes: 0.90 (muy conservador)
                 bull_threshold: float = 0.50,      # Antes: 0.70
                 transition_stock_pct: float = 0.70,  # Exposición en transición
                 min_regime_days: int = 5):          # Días mínimos antes de cambiar
        """
        Args:
            bear_threshold: Probabilidad de riesgo para bear market
            bull_threshold: Probabilidad bajo la cual es bull market
            transition_stock_pct: % en stocks durante transición
            min_regime_days: Días mínimos en régimen para evitar whipsaws
        """
        self.bear_threshold = bear_threshold
        self.bull_threshold = bull_threshold
        self.transition_stock_pct = transition_stock_pct
        self.min_regime_days = min_regime_days

        # Estado interno
        self.current_regime = Regime.BULL
        self.current_stock_allocation = 1.0
        self.days_in_regime = 0
        self.last_regime_change = None

    def get_regime(self, risk_prob: float) -> Regime:
        """Determina el régimen según probabilidad de riesgo"""
        if risk_prob > self.bear_threshold:
            return Regime.BEAR
        elif risk_prob < self.bull_threshold:
            return Regime.BULL
        else:
            return Regime.TRANSITION

    def calculate_position(self, signals: Dict) -> Dict:
        """
        Calcula posición objetivo basada en régimen HMM

        Returns:
            Dict con: regime, stock_allocation, action, reasoning
        """
        risk_prob = signals['risky_probability']
        detected_regime = self.get_regime(risk_prob)

        reasoning = []
        action = 'HOLD'
        target_allocation = self.current_stock_allocation

        # Info del régimen
        reasoning.append(f"Risk probability: {risk_prob:.2%}")
        reasoning.append(f"Detected regime: {detected_regime.value}")

        # Incrementar días en régimen actual
        if detected_regime == self.current_regime:
            self.days_in_regime += 1

        # Decidir si cambiar de régimen (evitar whipsaws)
        should_change = False
        if detected_regime != self.current_regime:
            # Cambio inmediato si es a BEAR (protección prioritaria)
            if detected_regime == Regime.BEAR:
                should_change = True
                reasoning.append("⚠️ Bear market detected - immediate protection")
            # Para otros cambios, esperar días mínimos
            elif self.days_in_regime >= self.min_regime_days:
                should_change = True
                reasoning.append(f"Regime change confirmed after {self.days_in_regime} days")
            else:
                reasoning.append(f"Waiting for regime confirmation ({self.days_in_regime}/{self.min_regime_days} days)")

        # EJECUTAR CAMBIO DE RÉGIMEN
        if should_change:
            old_regime = self.current_regime
            self.current_regime = detected_regime
            self.days_in_regime = 0

            # Determinar nueva allocación
            if detected_regime == Regime.BEAR:
                target_allocation = 0.0  # 100% Cash
                action = 'SELL_TO_CASH'
                reasoning.append(f"🔴 BEAR MARKET: Exit to cash (was {old_regime.value})")

            elif detected_regime == Regime.TRANSITION:
                target_allocation = self.transition_stock_pct  # 70% Stocks

                if old_regime == Regime.BEAR:
                    action = 'PARTIAL_BUY'
                    reasoning.append(f"🟡 TRANSITION: Cautiously entering {target_allocation:.0%} stocks")
                else:  # From BULL
                    action = 'PARTIAL_SELL'
                    reasoning.append(f"🟡 TRANSITION: Reducing to {target_allocation:.0%} stocks")

            elif detected_regime == Regime.BULL:
                target_allocation = 1.0  # 100% Stocks
                action = 'BUY_FULL'
                reasoning.append(f"🟢 BULL MARKET: Full allocation to stocks (was {old_regime.value})")

        else:
            # Mantener posición actual
            target_allocation = self.current_stock_allocation
            action = 'HOLD'
            reasoning.append(f"Holding {self.current_regime.value} position ({target_allocation:.0%} stocks)")

        return {
            'regime': self.current_regime,
            'target_stock_allocation': target_allocation,
            'action': action,
            'reasoning': reasoning,
            'risk_probability': risk_prob,
            'days_in_regime': self.days_in_regime
        }

    def execute_decision(self, decision: Dict, current_date):
        """Actualiza estado interno después de ejecutar decisión"""
        self.current_stock_allocation = decision['target_stock_allocation']

        if decision['action'] != 'HOLD':
            self.last_regime_change = current_date

    def get_summary(self) -> Dict:
        """Retorna resumen del estado actual"""
        return {
            'current_regime': self.current_regime.value,
            'stock_allocation': self.current_stock_allocation,
            'days_in_regime': self.days_in_regime,
            'last_change': self.last_regime_change
        }


# Alias para compatibilidad
RegimeSwitchingStrategy = OptimizedHMMStrategy
