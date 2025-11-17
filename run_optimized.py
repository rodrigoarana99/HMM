#!/usr/bin/env python3
"""
Script Principal - Estrategia HMM Optimizada
Versión simplificada y optimizada basada en Paper 3 (Donninger, 2017)

Uso:
    python run_optimized.py [SYMBOL] [START_DATE] [END_DATE]

Ejemplo:
    python run_optimized.py SPY 2020-01-01 2024-01-01
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import warnings

# Módulos del proyecto
from hmm_detector import HMMRegimeDetector, calculate_vix_term_structure
from optimized_strategy import OptimizedHMMStrategy
from backtester import Backtest
from evaluation import PerformanceMetrics

# Data source
try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance es requerido. Instalar con: pip install yfinance")
    sys.exit(1)


def download_data(symbol: str, start_date: str, end_date: str) -> dict:
    """
    Descarga datos usando Yahoo Finance

    Args:
        symbol: Ticker (ej: SPY, QQQ)
        start_date: Fecha inicio (YYYY-MM-DD)
        end_date: Fecha fin (YYYY-MM-DD)

    Returns:
        Dict con: dates, prices, returns, vix_f1, vix_f2
    """

    print(f"\n{'='*70}")
    print(f"DESCARGANDO DATOS")
    print(f"{'='*70}")
    print(f"Símbolo: {symbol}")
    print(f"Período: {start_date} → {end_date}\n")

    # Descargar activo principal
    print(f"Descargando {symbol}...")
    stock = yf.download(symbol, start=start_date, end=end_date, progress=False)

    if stock.empty:
        raise ValueError(f"No se pudo descargar datos para {symbol}")

    # Descargar VIX (para detectar régimen)
    print("Descargando VIX...")
    vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)

    # Descargar VIX3M (VXV) - usado para term structure
    print("Descargando VIX3M...")
    vix3m = yf.download('^VIX3M', start=start_date, end=end_date, progress=False)

    # Alinear fechas
    common_dates = stock.index.intersection(vix.index).intersection(vix3m.index)

    if len(common_dates) < 100:
        raise ValueError(f"Muy pocas fechas comunes: {len(common_dates)}")

    # Preparar datos
    stock_aligned = stock.loc[common_dates]
    vix_aligned = vix.loc[common_dates]
    vix3m_aligned = vix3m.loc[common_dates]

    prices = stock_aligned['Close'].values
    vix_f1 = vix_aligned['Close'].values  # VIX 1 mes
    vix_f2 = vix3m_aligned['Close'].values  # VIX 3 meses

    # Calcular returns
    returns = np.diff(prices) / prices[:-1]
    returns = np.concatenate([[0], returns])  # Agregar 0 al inicio

    print(f"\n✓ Datos descargados exitosamente")
    print(f"  Períodos: {len(common_dates)}")
    print(f"  Primera fecha: {common_dates[0].date()}")
    print(f"  Última fecha: {common_dates[-1].date()}")
    print(f"  Precio inicial: ${prices[0]:.2f}")
    print(f"  Precio final: ${prices[-1]:.2f}")

    return {
        'dates': common_dates,
        'prices': prices,
        'returns': returns,
        'vix_f1': vix_f1,
        'vix_f2': vix_f2,
        'symbol': symbol
    }


def run_backtest(data: dict, initial_capital: float = 1_000_000,
                bear_threshold: float = 0.75,
                bull_threshold: float = 0.50) -> tuple:
    """
    Ejecuta backtest con la estrategia optimizada

    Args:
        data: Datos de mercado
        initial_capital: Capital inicial
        bear_threshold: Umbral para bear market
        bull_threshold: Umbral para bull market

    Returns:
        (portfolio_df, trades_df, metrics)
    """
    print(f"\n{'='*70}")
    print(f"EJECUTANDO BACKTEST")
    print(f"{'='*70}")
    print(f"Estrategia: HMM Regime Switching Optimizada")
    print(f"Capital inicial: ${initial_capital:,.0f}")
    print(f"Bear threshold: {bear_threshold:.2f}")
    print(f"Bull threshold: {bull_threshold:.2f}\n")

    # Inicializar componentes
    hmm = HMMRegimeDetector(n_states=2, window_days=252)
    strategy = OptimizedHMMStrategy(
        bear_threshold=bear_threshold,
        bull_threshold=bull_threshold,
        transition_stock_pct=0.70,
        min_regime_days=5
    )
    backtest = Backtest(initial_capital=initial_capital)

    # Calcular VIX term structure
    vix_ts = calculate_vix_term_structure(data['vix_f1'], data['vix_f2'])

    # Período mínimo de historia
    min_history = min(252, len(data['dates']) // 4)

    # Loop principal
    for i in range(min_history, len(data['dates'])):
        current_date = data['dates'][i]

        # Entrenar HMM con datos históricos
        hist_returns = data['returns'][:i+1]
        hist_vix_ts = vix_ts[:i+1]

        try:
            hmm.fit(hist_returns, hist_vix_ts)
            state_probs, risky_prob = hmm.predict_current_state(hist_returns, hist_vix_ts)
        except Exception as e:
            warnings.warn(f"Error en HMM en {current_date.date()}: {e}")
            risky_prob = 0.5
            state_probs = np.array([0.5, 0.5])

        # Generar señales
        signals = {
            'risky_probability': risky_prob,
            'state_probabilities': state_probs,
            'regime': 'bull' if risky_prob < bull_threshold else
                     ('bear' if risky_prob > bear_threshold else 'transition')
        }

        # Decisión de la estrategia
        decision = strategy.calculate_position(signals)

        # Precio actual
        current_price = data['prices'][i]
        current_prices = {data['symbol']: current_price}

        # Ejecutar trade si es necesario
        action = decision['action']

        if action != 'HOLD':
            # Actualizar valor del portfolio
            portfolio_value = backtest.update_portfolio_value(
                current_date, current_prices, decision['regime'].value
            )

            # Calcular target value
            target_allocation = decision['target_stock_allocation']
            target_value = portfolio_value * target_allocation

            # Ejecutar trade
            backtest.execute_trade(
                timestamp=current_date,
                action=action,
                asset=data['symbol'],
                price=current_price,
                target_size=target_value,
                reasoning=decision['reasoning'],
                regime=decision['regime'].value
            )

            # Actualizar estrategia
            strategy.execute_decision(decision, current_date)

        # Actualizar portfolio value
        backtest.update_portfolio_value(
            current_date, current_prices, decision['regime'].value
        )

        # Progress cada 3 meses
        if i % 63 == 0:
            summary = backtest.get_portfolio_summary()
            regime = decision['regime'].value
            print(f"{current_date.date()}: ${summary['total_value']:,.0f} | "
                  f"Return={summary['total_return']:+.2%} | "
                  f"Regime={regime} | Risk={risky_prob:.2%}")

    # Resultados finales
    portfolio_df = backtest.get_portfolio_history_df()
    trades_df = backtest.get_trades_df()
    summary = backtest.get_portfolio_summary()

    print(f"\n{'='*70}")
    print(f"RESULTADOS FINALES")
    print(f"{'='*70}")
    print(f"Valor final:       ${summary['total_value']:,.2f}")
    print(f"Return total:      {summary['total_return']:+.2%}")
    print(f"Número de trades:  {summary['num_trades']}")
    print(f"Comisiones:        ${summary['total_commission']:,.2f}")
    print(f"Slippage:          ${summary['total_slippage']:,.2f}")

    # Calcular métricas
    benchmark_returns = np.diff(data['prices'][min_history:]) / data['prices'][min_history:-1]

    metrics = PerformanceMetrics.calculate_all_metrics(
        portfolio_df,
        benchmark_returns=benchmark_returns,
        risk_free_rate=0.02
    )

    print(f"\nMÉTRICAS DE PERFORMANCE:")
    print(f"  Return anualizado:  {metrics['annualized_return']:+.2%}")
    print(f"  Volatilidad:        {metrics['annualized_volatility']:.2%}")
    print(f"  Sharpe Ratio:       {metrics['sharpe_ratio']:.3f}")
    print(f"  Calmar Ratio:       {metrics['calmar_ratio']:.3f}")
    print(f"  Max Drawdown:       {metrics['max_drawdown']:.2%}")
    print(f"  Win Rate:           {metrics.get('win_rate', 0):.1%}")

    # Guardar resultados
    portfolio_df.to_csv('optimized_portfolio.csv', index=False)
    trades_df.to_csv('optimized_trades.csv', index=False)

    print(f"\n✓ Resultados guardados:")
    print(f"  - optimized_portfolio.csv")
    print(f"  - optimized_trades.csv")

    return portfolio_df, trades_df, metrics


def main():
    """Función principal"""
    # Parsear argumentos
    if len(sys.argv) >= 4:
        symbol = sys.argv[1]
        start_date = sys.argv[2]
        end_date = sys.argv[3]
    else:
        # Valores por defecto
        symbol = 'SPY'
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')

    print("\n" + "="*70)
    print("ESTRATEGIA HMM OPTIMIZADA - REGIME SWITCHING")
    print("="*70)

    try:
        # Descargar datos
        data = download_data(symbol, start_date, end_date)

        # Ejecutar backtest
        portfolio_df, trades_df, metrics = run_backtest(data)

        print("\n" + "="*70)
        print("✓ COMPLETADO EXITOSAMENTE")
        print("="*70 + "\n")

        return 0

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
