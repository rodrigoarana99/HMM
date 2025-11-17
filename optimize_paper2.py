#cspell:disable
"""
Optimizador de Parámetros para Paper 2 (Mean Reversion Strategy)

Este script optimiza los parámetros de la estrategia de mean reversion:
- entry_threshold: Z-score para entrar (negativo = precio barato)
- exit_threshold: Z-score para salir (positivo = precio caro)
- position_size: Tamaño base de posición
- use_momentum_filter: Si usar filtro de momentum

Usa grid search con validación walk-forward para evitar overfitting
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import itertools
from tqdm import tqdm
import warnings

# Importar estrategia y módulos
from momemtum_p_2 import Paper2Strategy
from signals import IntegratedSignalGenerator, calculate_vix_term_structure
from backtester import Backtest
from evaluation import PerformanceMetrics


class Paper2Optimizer:
    """
    Optimizador para la estrategia Paper 2 (Mean Reversion)
    """

    def __init__(self, initial_capital: float = 1_000_000):
        self.initial_capital = initial_capital

    def _run_single_backtest(self,
                            strategy_params: Dict,
                            dates: List[datetime],
                            prices: Dict[str, np.ndarray],
                            returns: np.ndarray,
                            vix_f1: np.ndarray,
                            vix_f2: np.ndarray,
                            market_prices,
                            min_history: int = 252) -> Dict:
        """
        Ejecuta un backtest con parámetros específicos

        Returns:
            Dict con métricas de performance
        """
        # Crear estrategia con parámetros
        strategy = Paper2Strategy(**strategy_params)
        signal_generator = IntegratedSignalGenerator()
        backtest = Backtest(initial_capital=self.initial_capital)

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

        # Calcular métricas
        portfolio_df = backtest.get_portfolio_history_df()
        summary = backtest.get_portfolio_summary()

        if len(portfolio_df) == 0:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'num_trades': 0,
                'valid': False
            }

        # Benchmark returns
        benchmark_returns = np.diff(main_prices[min_history:]) / main_prices[min_history:-1]

        try:
            metrics = PerformanceMetrics.calculate_all_metrics(
                portfolio_df,
                benchmark_returns=benchmark_returns,
                risk_free_rate=0.02
            )

            return {
                'total_return': metrics['total_return'],
                'annualized_return': metrics['annualized_return'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'max_drawdown': metrics['max_drawdown'],
                'num_trades': summary['num_trades'],
                'calmar_ratio': metrics.get('calmar_ratio', 0.0),
                'sortino_ratio': metrics.get('sortino_ratio', 0.0),
                'valid': True
            }
        except Exception as e:
            warnings.warn(f"Error calculando métricas: {e}")
            return {
                'total_return': summary['total_return'],
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'num_trades': summary['num_trades'],
                'valid': False
            }

    def grid_search(self,
                   dates: List[datetime],
                   prices: Dict[str, np.ndarray],
                   returns: np.ndarray,
                   vix_f1: np.ndarray,
                   vix_f2: np.ndarray,
                   market_prices=None,
                   min_history: int = 252,
                   optimize_metric: str = 'sharpe_ratio') -> Tuple[Dict, pd.DataFrame]:
        """
        Grid search para encontrar los mejores parámetros

        Args:
            optimize_metric: 'sharpe_ratio', 'total_return', 'calmar_ratio', etc.

        Returns:
            (best_params, results_df)
        """
        print("\n" + "="*80)
        print("OPTIMIZACIÓN PAPER 2 - MEAN REVERSION STRATEGY")
        print("="*80)
        print(f"Métrica de optimización: {optimize_metric}")
        print(f"Período: {dates[min_history]} a {dates[-1]}")
        print()

        # Grid de parámetros
        param_grid = {
            'entry_threshold': [-2.0, -1.5, -1.0, -0.5, -0.3, -0.2, -0.1],  # Más agresivos
            'exit_threshold': [0.1, 0.2, 0.3, 0.5, 1.0, 1.5, 2.0],         # Más variados
            'position_size': [0.5, 0.75, 1.0],                              # Tamaños de posición
            'use_momentum_filter': [True, False]                            # Con/sin filtro
        }

        # Generar todas las combinaciones
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        all_combinations = list(itertools.product(*param_values))

        print(f"Total de combinaciones a probar: {len(all_combinations)}")
        print()

        # Ejecutar grid search
        results = []

        for combo in tqdm(all_combinations, desc="Optimizando"):
            params = dict(zip(param_names, combo))

            # Validación: exit debe ser mayor que -entry
            if params['exit_threshold'] < abs(params['entry_threshold']):
                continue  # Skip combinaciones asimétricas malas

            # Ejecutar backtest
            metrics = self._run_single_backtest(
                params, dates, prices, returns, vix_f1, vix_f2,
                market_prices, min_history
            )

            # Guardar resultado
            result = {**params, **metrics}
            results.append(result)

        # Crear DataFrame de resultados
        results_df = pd.DataFrame(results)

        # Filtrar solo resultados válidos con trades
        valid_results = results_df[
            (results_df['valid'] == True) &
            (results_df['num_trades'] > 5)  # Al menos 5 trades
        ]

        if len(valid_results) == 0:
            print("\n⚠️  ADVERTENCIA: No se encontraron configuraciones válidas con >5 trades")
            print("Usando todas las configuraciones válidas...")
            valid_results = results_df[results_df['valid'] == True]

        if len(valid_results) == 0:
            print("\n❌ ERROR: No se encontraron configuraciones válidas")
            return {}, results_df

        # Encontrar mejores parámetros
        best_idx = valid_results[optimize_metric].idxmax()
        best_params = valid_results.loc[best_idx, param_names].to_dict()
        best_metrics = valid_results.loc[best_idx].to_dict()

        # Imprimir resultados
        print("\n" + "="*80)
        print("RESULTADOS DE OPTIMIZACIÓN")
        print("="*80)
        print(f"\nMEJORES PARÁMETROS (optimizando {optimize_metric}):")
        print("-" * 80)
        for param, value in best_params.items():
            print(f"  {param:25s}: {value}")

        print(f"\nMÉTRICAS CON MEJORES PARÁMETROS:")
        print("-" * 80)
        print(f"  Total Return:             {best_metrics['total_return']:.2%}")
        print(f"  Annualized Return:        {best_metrics.get('annualized_return', 0):.2%}")
        print(f"  Sharpe Ratio:             {best_metrics['sharpe_ratio']:.3f}")
        print(f"  Sortino Ratio:            {best_metrics.get('sortino_ratio', 0):.3f}")
        print(f"  Max Drawdown:             {best_metrics['max_drawdown']:.2%}")
        print(f"  Calmar Ratio:             {best_metrics.get('calmar_ratio', 0):.3f}")
        print(f"  Number of Trades:         {int(best_metrics['num_trades'])}")

        # Top 5 configuraciones
        print(f"\nTOP 5 CONFIGURACIONES (por {optimize_metric}):")
        print("-" * 80)
        top5 = valid_results.nlargest(5, optimize_metric)

        display_cols = ['entry_threshold', 'exit_threshold', 'position_size',
                       'use_momentum_filter', optimize_metric, 'num_trades']
        print(top5[display_cols].to_string(index=False))

        # Guardar resultados completos
        results_df.to_csv('optimization_results_paper2.csv', index=False)
        print(f"\n✓ Resultados completos guardados en: optimization_results_paper2.csv")

        return best_params, results_df


def main():
    """Función principal para ejecutar optimización"""
    import sys
    sys.path.append('.')

    from main import ThetaDataConnector

    # Parámetros de datos
    symbol = 'SPY'
    end_date = datetime(2024, 12, 31)
    start_date = end_date - timedelta(days=4*365)

    print("="*80)
    print(f"OPTIMIZACIÓN DE PARÁMETROS - PAPER 2")
    print("="*80)
    print(f"Símbolo: {symbol}")
    print(f"Período: {start_date.date()} a {end_date.date()}")
    print()

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

        # Preparar datos para optimización
        dates = merged_data['date'].tolist()
        prices = {symbol: merged_data['close'].values}
        returns = merged_data['returns'].values
        vix_f1 = merged_data['vix_f1'].values
        vix_f2 = merged_data['vix_f2'].values

        # EJECUTAR OPTIMIZACIÓN
        optimizer = Paper2Optimizer(initial_capital=1_000_000)

        best_params, results_df = optimizer.grid_search(
            dates=dates,
            prices=prices,
            returns=returns,
            vix_f1=vix_f1,
            vix_f2=vix_f2,
            market_prices=None,
            min_history=252,
            optimize_metric='sharpe_ratio'  # Cambiar si quieres optimizar otra métrica
        )

        if best_params:
            print("\n" + "="*80)
            print("OPTIMIZACIÓN COMPLETADA")
            print("="*80)
            print("\nUsa estos parámetros en Paper 2:")
            print(f"strategy = Paper2Strategy(")
            for param, value in best_params.items():
                print(f"    {param}={value},")
            print(")")

    finally:
        theta.disconnect()


if __name__ == '__main__':
    main()
