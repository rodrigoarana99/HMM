#cspell:disable
"""
Ejecutar Paper 2 (Mean Reversion) con Parámetros Optimizados

Este script ejecuta la estrategia Paper 2 usando los parámetros
encontrados por el optimizador.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import sys

# Importar estrategia y módulos
from momemtum_p_2 import Paper2Strategy
from signals import IntegratedSignalGenerator, calculate_vix_term_structure
from backtester import Backtest
from evaluation import PerformanceMetrics, print_metrics_report


def run_paper2_backtest(
        dates: List[datetime],
        prices: Dict[str, np.ndarray],
        returns: np.ndarray,
        vix_f1: np.ndarray,
        vix_f2: np.ndarray,
        market_prices=None,
        min_history: int = 252,
        strategy_params: Dict = None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Ejecuta backtest de Paper 2 con parámetros personalizados

    Args:
        strategy_params: Dict con parámetros de la estrategia
                        Si es None, usa parámetros por defecto optimizados

    Returns:
        (portfolio_df, trades_df, summary)
    """
    # Parámetros por defecto (ACTUALIZAR DESPUÉS DE OPTIMIZACIÓN)
    if strategy_params is None:
        strategy_params = {
            'entry_threshold': -1.0,      # PLACEHOLDER - actualizar con resultado de optimización
            'exit_threshold': 0.5,         # PLACEHOLDER - actualizar con resultado de optimización
            'position_size': 1.0,          # PLACEHOLDER - actualizar con resultado de optimización
            'use_momentum_filter': True    # PLACEHOLDER - actualizar con resultado de optimización
        }

    print("\n" + "="*80)
    print("BACKTEST: PAPER 2 - MEAN REVERSION STRATEGY")
    print("="*80)
    print(f"Período: {dates[min_history]} a {dates[-1]}")
    print(f"\nParámetros de la estrategia:")
    for param, value in strategy_params.items():
        print(f"  {param:25s}: {value}")
    print()

    # Crear estrategia y backtester
    strategy = Paper2Strategy(**strategy_params)
    signal_generator = IntegratedSignalGenerator()
    backtest = Backtest(initial_capital=1_000_000)

    # Preparar datos
    vix_term_struct = calculate_vix_term_structure(vix_f1, vix_f2)
    main_asset = list(prices.keys())[0]
    main_prices = prices[main_asset]

    # Loop de backtest
    for i in range(min_history, len(dates)):
        current_date = dates[i]

        # Generar señales
        hist_returns = returns[:i+1]
        hist_vix_ts = vix_term_struct[:i+1]
        hist_prices = main_prices[:i+1]
        hist_market = market_prices[:i+1] if market_prices is not None else None

        signals = signal_generator.generate_signals(
            prices=hist_prices,
            returns=hist_returns,
            vix_term_structure=hist_vix_ts,
            market_prices=hist_market
        )

        # Decisión de la estrategia
        decision = strategy.calculate_position(signals)

        # Precios actuales
        current_prices = {asset: price_array[i] for asset, price_array in prices.items()}

        # Ejecutar trades
        action = decision.get('action', 'HOLD')
        if action in ['BUY', 'SELL', 'CLOSE', 'ADJUST']:
            target_size = decision.get('target_size', 0.0)
            portfolio_value = backtest.update_portfolio_value(
                current_date, current_prices, signals['regime']
            )
            target_value = portfolio_value * target_size

            backtest.execute_trade(
                timestamp=current_date,
                action=action,
                asset=main_asset,
                price=current_prices[main_asset],
                target_size=target_value,
                reasoning=decision.get('reasoning', []),
                regime=signals['regime']
            )

        # Actualizar portfolio
        backtest.update_portfolio_value(
            current_date, current_prices, signals['regime']
        )

        # Progress cada año
        if i % 252 == 0:
            summary = backtest.get_portfolio_summary()
            print(f"{current_date.date()}: ${summary['total_value']:,.2f} | "
                  f"Return={summary['total_return']:.2%} | "
                  f"Trades={summary['num_trades']}")

    # Resultados
    portfolio_df = backtest.get_portfolio_history_df()
    trades_df = backtest.get_trades_df()
    summary = backtest.get_portfolio_summary()

    # Calcular métricas
    benchmark_returns = np.diff(main_prices[min_history:]) / main_prices[min_history:-1]
    metrics = PerformanceMetrics.calculate_all_metrics(
        portfolio_df,
        benchmark_returns=benchmark_returns,
        risk_free_rate=0.02
    )

    # Imprimir resultados
    print("\n" + "="*80)
    print("RESULTADOS FINALES")
    print("="*80)
    print_metrics_report(metrics, summary)

    # Comparación con Buy & Hold
    spy_final = main_prices[-1]
    spy_initial = main_prices[min_history]
    bh_return = (spy_final - spy_initial) / spy_initial

    print("\n" + "="*80)
    print("COMPARACIÓN CON BUY & HOLD")
    print("="*80)
    print(f"Strategy Return:          {summary['total_return']:.2%}")
    print(f"Buy & Hold Return:        {bh_return:.2%}")
    print(f"Excess Return:            {(summary['total_return'] - bh_return):.2%}")
    print()
    print(f"Strategy Sharpe:          {metrics['sharpe_ratio']:.3f}")
    print(f"Buy & Hold Sharpe:        {((bh_return - 0.02) / (np.std(benchmark_returns) * np.sqrt(252))):.3f}")
    print()

    # Guardar resultados
    portfolio_df.to_csv('Paper2_Optimized_portfolio.csv', index=False)
    trades_df.to_csv('Paper2_Optimized_trades.csv', index=False)

    print("Archivos guardados:")
    print("  - Paper2_Optimized_portfolio.csv")
    print("  - Paper2_Optimized_trades.csv")
    print()

    return portfolio_df, trades_df, summary


def main():
    """Función principal"""
    from main import ThetaDataConnector

    # Parámetros de datos
    symbol = 'SPY'
    end_date = datetime(2024, 12, 31)
    start_date = end_date - timedelta(days=4*365)

    # Obtener datos
    theta = ThetaDataConnector()
    if not theta.connect():
        print("No se pudo conectar a Thetadata. Abortando.")
        return

    try:
        # Descargar SPY
        print(f"\nDescargando {symbol}...")
        stock_data = theta.get_market_data_eod(symbol, start_date, end_date)

        if stock_data.empty:
            print(f"No se pudieron obtener datos para {symbol}. Abortando.")
            return

        stock_data['returns'] = stock_data['close'].pct_change()
        stock_data = stock_data.dropna(subset=['returns'])

        # Descargar VIX/VXV
        vix_futures = theta.get_vix_futures(start_date, end_date)
        vix_f1_data = vix_futures['F1']
        vix_f2_data = vix_futures['F2']

        if vix_f1_data.empty or vix_f2_data.empty:
            print("No se pudieron obtener datos de VIX/VXV. Abortando.")
            return

        # Alinear datos
        print("\nAlineando datos...")
        vix_f1_data = vix_f1_data[['date', 'close']].rename(columns={'close': 'vix_f1'})
        vix_f2_data = vix_f2_data[['date', 'close']].rename(columns={'close': 'vix_f2'})

        merged_data = stock_data.merge(vix_f1_data, on='date', how='left')
        merged_data = merged_data.merge(vix_f2_data, on='date', how='left')

        merged_data['vix_f1'] = merged_data['vix_f1'].fillna(method='ffill', limit=5)
        merged_data['vix_f2'] = merged_data['vix_f2'].fillna(method='ffill', limit=5)
        merged_data = merged_data.dropna(subset=['vix_f1', 'vix_f2', 'close', 'returns'])

        print(f"✓ Datos alineados: {len(merged_data)} días\n")

        # Preparar datos
        dates = merged_data['date'].tolist()
        prices = {symbol: merged_data['close'].values}
        returns = merged_data['returns'].values
        vix_f1 = merged_data['vix_f1'].values
        vix_f2 = merged_data['vix_f2'].values

        # EJECUTAR BACKTEST
        # TODO: Actualizar estos parámetros con los resultados de la optimización
        optimized_params = {
            'entry_threshold': -1.0,       # ACTUALIZAR
            'exit_threshold': 0.5,         # ACTUALIZAR
            'position_size': 1.0,          # ACTUALIZAR
            'use_momentum_filter': True    # ACTUALIZAR
        }

        portfolio_df, trades_df, summary = run_paper2_backtest(
            dates=dates,
            prices=prices,
            returns=returns,
            vix_f1=vix_f1,
            vix_f2=vix_f2,
            market_prices=None,
            min_history=252,
            strategy_params=optimized_params
        )

        print("\n" + "="*80)
        print("BACKTEST COMPLETADO")
        print("="*80)

    finally:
        theta.disconnect()


if __name__ == '__main__':
    main()
