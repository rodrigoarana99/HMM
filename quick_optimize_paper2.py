#cspell:disable
"""
Optimizador RÁPIDO y SIMPLE para Paper 2
- Sin walk-forward (más rápido)
- Parámetros MUY agresivos para asegurar trades
- Para encontrar parámetros que funcionen primero
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List
import itertools
from tqdm import tqdm
import warnings

from momemtum_p_2 import Paper2Strategy
from signals import IntegratedSignalGenerator, calculate_vix_term_structure
from backtester import Backtest
from evaluation import PerformanceMetrics


def run_single_test(strategy_params: Dict,
                   dates: List[datetime],
                   prices: Dict[str, np.ndarray],
                   returns: np.ndarray,
                   vix_f1: np.ndarray,
                   vix_f2: np.ndarray,
                   min_history: int = 252) -> Dict:
    """Ejecuta un backtest simple"""

    strategy = Paper2Strategy(**strategy_params)
    signal_generator = IntegratedSignalGenerator()
    backtest = Backtest(initial_capital=1_000_000)

    vix_term_struct = calculate_vix_term_structure(vix_f1, vix_f2)
    main_asset = list(prices.keys())[0]
    main_prices = prices[main_asset]

    # Loop de backtest
    for i in range(min_history, len(dates)):
        current_date = dates[i]

        signals = signal_generator.generate_signals(
            prices=main_prices[:i+1],
            returns=returns[:i+1],
            vix_term_structure=vix_term_struct[:i+1],
            market_prices=None
        )

        decision = strategy.calculate_position(signals)
        current_prices = {main_asset: main_prices[i]}

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

        backtest.update_portfolio_value(current_date, current_prices, signals['regime'])

    # Calcular métricas
    portfolio_df = backtest.get_portfolio_history_df()
    summary = backtest.get_portfolio_summary()

    if len(portfolio_df) == 0:
        return {'valid': False, 'num_trades': 0}

    benchmark_returns = np.diff(main_prices[min_history:]) / main_prices[min_history:-1]

    try:
        metrics = PerformanceMetrics.calculate_all_metrics(
            portfolio_df, benchmark_returns=benchmark_returns, risk_free_rate=0.02
        )

        return {
            'total_return': metrics['total_return'],
            'sharpe_ratio': metrics['sharpe_ratio'],
            'max_drawdown': metrics['max_drawdown'],
            'num_trades': summary['num_trades'],
            'valid': True
        }
    except:
        return {
            'total_return': summary.get('total_return', 0),
            'sharpe_ratio': 0.0,
            'num_trades': summary.get('num_trades', 0),
            'valid': True
        }


def main():
    from main import ThetaDataConnector

    symbol = 'SPY'
    end_date = datetime(2024, 12, 31)
    start_date = end_date - timedelta(days=4*365)

    print("="*80)
    print("OPTIMIZACIÓN RÁPIDA - PAPER 2")
    print("="*80)

    # Obtener datos
    theta = ThetaDataConnector()
    if not theta.connect():
        return

    try:
        stock_data = theta.get_market_data_eod(symbol, start_date, end_date)
        if stock_data.empty:
            return

        stock_data['returns'] = stock_data['close'].pct_change()
        stock_data = stock_data.dropna(subset=['returns'])

        vix_futures = theta.get_vix_futures(start_date, end_date)
        vix_f1_data = vix_futures['F1']
        vix_f2_data = vix_futures['F2']

        if vix_f1_data.empty or vix_f2_data.empty:
            return

        # Alinear datos
        vix_f1_data = vix_f1_data[['date', 'close']].rename(columns={'close': 'vix_f1'})
        vix_f2_data = vix_f2_data[['date', 'close']].rename(columns={'close': 'vix_f2'})

        merged_data = stock_data.merge(vix_f1_data, on='date', how='left')
        merged_data = merged_data.merge(vix_f2_data, on='date', how='left')
        merged_data['vix_f1'] = merged_data['vix_f1'].fillna(method='ffill', limit=5)
        merged_data['vix_f2'] = merged_data['vix_f2'].fillna(method='ffill', limit=5)
        merged_data = merged_data.dropna(subset=['vix_f1', 'vix_f2', 'close', 'returns'])

        print(f"Datos: {len(merged_data)} días\n")

        dates = merged_data['date'].tolist()
        prices = {symbol: merged_data['close'].values}
        returns = merged_data['returns'].values
        vix_f1 = merged_data['vix_f1'].values
        vix_f2 = merged_data['vix_f2'].values

        # Grid AGRESIVO
        param_grid = {
            'entry_threshold': [-5.0, -3.0, -2.0, -1.5, -1.0, -0.5, -0.2, -0.1, 0.0],
            'exit_threshold': [0.0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
            'position_size': [0.5, 1.0],
            'use_momentum_filter': [False]
        }

        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        all_combinations = list(itertools.product(*param_values))

        print(f"Probando {len(all_combinations)} combinaciones...")
        print()

        results = []
        for combo in tqdm(all_combinations):
            params = dict(zip(param_names, combo))

            metrics = run_single_test(
                params, dates, prices, returns, vix_f1, vix_f2, min_history=252
            )

            result = {**params, **metrics}
            results.append(result)

        # Analizar resultados
        results_df = pd.DataFrame(results)
        results_df.to_csv('quick_optimization_results.csv', index=False)

        print("\n" + "="*80)
        print("RESULTADOS")
        print("="*80)

        # Configuraciones con trades
        with_trades = results_df[results_df['num_trades'] > 0]

        print(f"\nConfiguraciones totales:     {len(results_df)}")
        print(f"Con al menos 1 trade:        {len(with_trades)}")
        print(f"Sin trades:                  {len(results_df) - len(with_trades)}")

        if len(with_trades) > 0:
            print(f"\nPromedio de trades:          {with_trades['num_trades'].mean():.1f}")
            print(f"Máximo de trades:            {with_trades['num_trades'].max():.0f}")

            # Top 10 por número de trades
            print("\nTOP 10 POR NÚMERO DE TRADES:")
            print("-" * 80)
            top_trades = with_trades.nlargest(10, 'num_trades')
            print(top_trades[['entry_threshold', 'exit_threshold', 'position_size',
                             'num_trades', 'total_return', 'sharpe_ratio']].to_string(index=False))

            # Top 10 por Sharpe (con mínimo 10 trades)
            enough_trades = with_trades[with_trades['num_trades'] >= 10]
            if len(enough_trades) > 0:
                print("\nTOP 10 POR SHARPE RATIO (con >= 10 trades):")
                print("-" * 80)
                top_sharpe = enough_trades.nlargest(10, 'sharpe_ratio')
                print(top_sharpe[['entry_threshold', 'exit_threshold', 'position_size',
                                 'num_trades', 'total_return', 'sharpe_ratio']].to_string(index=False))

                # Mejor configuración
                best = top_sharpe.iloc[0]
                print("\n" + "="*80)
                print("MEJOR CONFIGURACIÓN (Sharpe con >= 10 trades):")
                print("="*80)
                print(f"entry_threshold:      {best['entry_threshold']}")
                print(f"exit_threshold:       {best['exit_threshold']}")
                print(f"position_size:        {best['position_size']}")
                print(f"use_momentum_filter:  {best['use_momentum_filter']}")
                print()
                print(f"Trades:               {best['num_trades']:.0f}")
                print(f"Total Return:         {best['total_return']:.2%}")
                print(f"Sharpe Ratio:         {best['sharpe_ratio']:.3f}")
                print(f"Max Drawdown:         {best['max_drawdown']:.2%}")
        else:
            print("\n❌ NINGUNA configuración generó trades!")
            print("\nVerificando z_scores en datos...")

            # Debug: Ver rango de z_scores
            signal_gen = IntegratedSignalGenerator()
            vix_ts = calculate_vix_term_structure(vix_f1, vix_f2)

            z_scores = []
            for i in range(252, len(dates), 50):  # Samplear cada 50 días
                signals = signal_gen.generate_signals(
                    prices=prices[symbol][:i+1],
                    returns=returns[:i+1],
                    vix_term_structure=vix_ts[:i+1],
                    market_prices=None
                )
                z_scores.append(signals['mean_reversion']['z_score'])

            z_scores = np.array(z_scores)
            print(f"\nZ-scores observados:")
            print(f"  Mínimo:  {z_scores.min():.2f}")
            print(f"  Máximo:  {z_scores.max():.2f}")
            print(f"  Media:   {z_scores.mean():.2f}")
            print(f"  p5:      {np.percentile(z_scores, 5):.2f}")
            print(f"  p95:     {np.percentile(z_scores, 95):.2f}")

        print("\n✓ Resultados guardados en: quick_optimization_results.csv")

    finally:
        theta.disconnect()


if __name__ == '__main__':
    main()
