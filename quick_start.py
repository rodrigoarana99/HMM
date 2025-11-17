"""
Quick Start Example - Uso básico del sistema
"""

import numpy as np
from signals import IntegratedSignalGenerator, calculate_vix_term_structure
from strategy import TradingStrategy

def quick_example():
    """
    Ejemplo rápido de cómo usar el sistema para generar señales
    """
    print("="*70)
    print("QUICK START - Generación de Señales")
    print("="*70)
    
    # 1. Generar datos de ejemplo
    print("\n1. Generando datos de ejemplo...")
    np.random.seed(42)
    
    n_days = 600  # 2+ años para HMM
    
    # Simular precios (con tendencia alcista)
    returns = np.random.normal(0.0005, 0.01, n_days)
    returns[200:250] = np.random.normal(-0.002, 0.025, 50)  # Crash
    
    prices = 100 * np.cumprod(1 + returns)
    
    # VIX futures (contango normal, backwardation en crash)
    vix_f1 = 15 + np.random.normal(0, 2, n_days)
    vix_f1[200:250] += 15  # Spike en crash
    
    vix_f2 = vix_f1 + 1.0  # Contango de 1 punto
    vix_f2[200:250] = vix_f1[200:250] - 1.5  # Backwardation en crash
    
    vix_term_structure = calculate_vix_term_structure(vix_f1, vix_f2)
    
    print(f"   Datos: {n_days} días")
    print(f"   Precio inicial: ${prices[0]:.2f}")
    print(f"   Precio final: ${prices[-1]:.2f}")
    print(f"   Return total: {(prices[-1]/prices[0] - 1)*100:.1f}%")
    
    # 2. Inicializar generador de señales
    print("\n2. Inicializando sistema...")
    signal_gen = IntegratedSignalGenerator()
    strategy = TradingStrategy(risk_threshold=0.95)
    
    # 3. Generar señales para el último día
    print("\n3. Generando señales...")
    signals = signal_gen.generate_signals(
        prices=prices,
        returns=returns,
        vix_term_structure=vix_term_structure,
        market_prices=None
    )
    
    # 4. Mostrar señales
    print("\n" + "="*70)
    print("SEÑALES GENERADAS")
    print("="*70)
    
    print(f"\nRégimen de Mercado: {signals['regime'].upper()}")
    print(f"Probabilidad de Riesgo: {signals['risky_probability']:.2%}")
    print(f"\nPrecio Actual: ${signals['current_price']:.2f}")
    print(f"Fair Value (Mean Reversion): ${signals['fair_value']:.2f}")
    print(f"Desviación: {signals['mean_reversion']['deviation']:.2%}")
    
    print(f"\nSeñales:")
    print(f"  Mean Reversion: {signals['mean_reversion']['signal']:+.3f}")
    print(f"  Momentum: {signals['momentum']['signal']:+.3f}")
    print(f"  Señal Final: {signals['final_signal']:+.3f}")
    print(f"  Confianza: {signals['confidence']:.2%}")
    
    # 5. Decisión de trading
    print("\n" + "="*70)
    print("DECISIÓN DE TRADING")
    print("="*70)
    
    decision = strategy.calculate_position(signals)
    
    print(f"\nAcción Recomendada: {decision['action']}")
    print(f"Posición Objetivo: {decision['target_size']:.1%}")
    print(f"Fuerza de Señal: {decision['signal_strength']:.2f}")
    
    print(f"\nRazonamiento:")
    for reason in decision['reasoning']:
        print(f"  • {reason}")
    
    # 6. Probar con diferentes escenarios
    print("\n" + "="*70)
    print("ESCENARIOS ALTERNATIVOS")
    print("="*70)
    
    # Escenario 1: Durante el crash (día 225)
    print("\nESCENARIO 1: Durante el Crash (día 225)")
    print("-" * 70)
    
    crash_signals = signal_gen.generate_signals(
        prices=prices[:226],
        returns=returns[:226],
        vix_term_structure=vix_term_structure[:226],
        market_prices=None
    )
    
    print(f"Régimen: {crash_signals['regime'].upper()}")
    print(f"Risk Probability: {crash_signals['risky_probability']:.2%}")
    print(f"Decisión: {strategy.calculate_position(crash_signals)['action']}")
    
    # Escenario 2: Después del crash (día 300)
    print("\nESCENARIO 2: Post-Crash Recovery (día 300)")
    print("-" * 70)
    
    recovery_signals = signal_gen.generate_signals(
        prices=prices[:301],
        returns=returns[:301],
        vix_term_structure=vix_term_structure[:301],
        market_prices=None
    )
    
    print(f"Régimen: {recovery_signals['regime'].upper()}")
    print(f"Risk Probability: {recovery_signals['risky_probability']:.2%}")
    print(f"Mean Reversion Signal: {recovery_signals['mean_reversion']['signal']:+.3f}")
    print(f"Decisión: {strategy.calculate_position(recovery_signals)['action']}")
    
    print("\n" + "="*70)
    print("Quick Start Completado!")
    print("="*70)
    print("\nPara backtest completo, ejecuta: python main.py")


if __name__ == '__main__':
    quick_example()