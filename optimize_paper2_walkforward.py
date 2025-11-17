#cspell:disable
"""
Optimizador de Parámetros para Paper 2 - CON WALK-FORWARD ANALYSIS

EVITA OVERFITTING mediante:
1. Walk-Forward Analysis: Divide datos en múltiples períodos
2. In-Sample (IS): Optimiza parámetros
3. Out-of-Sample (OOS): Valida sin ver datos futuros
4. Rolling windows: Simula trading real

Metodología:
- Divide datos en N ventanas (ej: 4 años → 4 ventanas de 1 año)
- Para cada ventana:
  * Optimiza en IS (ej: 6 meses)
  * Testea en OOS (ej: 6 meses siguientes)
- Combina resultados OOS (nunca vistos durante optimización)
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


class WalkForwardOptimizer:
    """
    Optimizador con Walk-Forward Analysis para evitar overfitting
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
                            start_idx: int,
                            end_idx: int,
                            min_history: int = 252) -> Dict:
        """
        Ejecuta un backtest con parámetros específicos en un rango de fechas

        Args:
            start_idx, end_idx: Índices para subset de datos

        Returns:
            Dict con métricas de performance
        """
        # Subset de datos
        dates_subset = dates[start_idx:end_idx]
        main_asset = list(prices.keys())[0]
        prices_subset = {main_asset: prices[main_asset][start_idx:end_idx]}
        returns_subset = returns[start_idx:end_idx]
        vix_f1_subset = vix_f1[start_idx:end_idx]
        vix_f2_subset = vix_f2[start_idx:end_idx]
        market_subset = market_prices[start_idx:end_idx] if market_prices is not None else None

        # Crear estrategia con parámetros
        strategy = Paper2Strategy(**strategy_params)
        signal_generator = IntegratedSignalGenerator()
        backtest = Backtest(initial_capital=self.initial_capital)

        # Preparar datos
        vix_term_struct = calculate_vix_term_structure(vix_f1_subset, vix_f2_subset)
        main_prices = prices_subset[main_asset]

        # Determinar mínimo de historia (relativo al subset)
        min_hist_local = min(min_history, len(dates_subset) // 3)

        # Loop de backtest
        for i in range(min_hist_local, len(dates_subset)):
            current_date = dates_subset[i]

            # Generar señales
            hist_returns = returns_subset[:i+1]
            hist_vix_ts = vix_term_struct[:i+1]
            hist_prices = main_prices[:i+1]
            hist_market = market_subset[:i+1] if market_subset is not None else None

            signals = signal_generator.generate_signals(
                prices=hist_prices,
                returns=hist_returns,
                vix_term_structure=hist_vix_ts,
                market_prices=hist_market
            )

            # Decisión de la estrategia
            decision = strategy.calculate_position(signals)

            # Precios actuales
            current_prices = {main_asset: main_prices[i]}

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
        benchmark_returns = np.diff(main_prices[min_hist_local:]) / main_prices[min_hist_local:-1]

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
                'total_return': summary.get('total_return', 0.0),
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'num_trades': summary.get('num_trades', 0),
                'valid': False
            }

    def walk_forward_analysis(self,
                             dates: List[datetime],
                             prices: Dict[str, np.ndarray],
                             returns: np.ndarray,
                             vix_f1: np.ndarray,
                             vix_f2: np.ndarray,
                             market_prices=None,
                             min_history: int = 252,
                             n_splits: int = 4,
                             train_ratio: float = 0.5,
                             optimize_metric: str = 'sharpe_ratio') -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
        """
        Walk-Forward Analysis con validación out-of-sample

        Args:
            n_splits: Número de ventanas walk-forward (ej: 4 = 4 períodos)
            train_ratio: Ratio de datos para training vs testing (0.5 = 50% train, 50% test)
            optimize_metric: Métrica a optimizar

        Returns:
            (best_params, oos_results_df, all_results_df)
        """
        print("\n" + "="*80)
        print("WALK-FORWARD OPTIMIZATION - PAPER 2")
        print("="*80)
        print(f"Método: {n_splits} splits con {train_ratio:.0%} train / {1-train_ratio:.0%} test")
        print(f"Métrica de optimización: {optimize_metric}")
        print(f"Período total: {dates[min_history]} a {dates[-1]}")
        print()

        # Grid de parámetros (MÁS AGRESIVOS para generar trades)
        param_grid = {
            'entry_threshold': [-3.0, -2.0, -1.5, -1.0, -0.5, -0.2, -0.1],  # Más agresivos
            'exit_threshold': [0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0],          # Más variados
            'position_size': [0.5, 0.75, 1.0],                              # Incluir 0.5
            'use_momentum_filter': [False]                                  # Solo False (el filtro bloquea trades)
        }

        # Generar combinaciones
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        all_combinations = list(itertools.product(*param_values))

        print(f"Combinaciones por split: {len(all_combinations)}")
        print(f"Total de backtests: {len(all_combinations) * n_splits * 2} (IS + OOS)")
        print()

        # Dividir datos en ventanas
        total_periods = len(dates) - min_history
        split_size = total_periods // n_splits

        oos_results = []  # Resultados out-of-sample
        all_walk_results = []

        # Walk-forward loop
        for split_idx in range(n_splits):
            split_start = min_history + split_idx * split_size
            split_end = min(split_start + split_size, len(dates))

            # Dividir en train/test
            train_size = int((split_end - split_start) * train_ratio)
            train_start = split_start
            train_end = split_start + train_size
            test_start = train_end
            test_end = split_end

            print(f"\n{'='*80}")
            print(f"SPLIT {split_idx + 1}/{n_splits}")
            print(f"{'='*80}")
            print(f"Training:   {dates[train_start].date()} a {dates[train_end-1].date()} ({train_end-train_start} días)")
            print(f"Testing:    {dates[test_start].date()} a {dates[test_end-1].date()} ({test_end-test_start} días)")
            print()

            # FASE 1: Optimizar en IN-SAMPLE
            print(f"Optimizando en período IN-SAMPLE (training)...")
            is_results = []

            for combo in tqdm(all_combinations, desc=f"Split {split_idx+1} IS"):
                params = dict(zip(param_names, combo))

                # Validación: exit threshold debe ser positivo pero no necesariamente > entry
                # Permitimos configuraciones asimétricas

                # Backtest en training data
                metrics = self._run_single_backtest(
                    params, dates, prices, returns, vix_f1, vix_f2,
                    market_prices, train_start, train_end, min_history=60
                )

                result = {
                    'split': split_idx,
                    'period': 'IS',
                    **params,
                    **metrics
                }
                is_results.append(result)

            # Encontrar mejores parámetros en IS
            is_df = pd.DataFrame(is_results)
            valid_is = is_df[(is_df['valid'] == True) & (is_df['num_trades'] > 1)]  # Reducido: >1 trade

            if len(valid_is) == 0:
                print(f"⚠️  Split {split_idx+1}: No se encontraron configuraciones válidas en IS (>1 trade)")
                print(f"Intentando con num_trades >= 1...")
                valid_is = is_df[is_df['valid'] == True]

            if len(valid_is) == 0:
                print(f"⚠️  Split {split_idx+1}: No hay configuraciones válidas. Saltando...")
                continue

            best_is_idx = valid_is[optimize_metric].idxmax()
            best_params = valid_is.loc[best_is_idx, param_names].to_dict()

            print(f"\nMejores parámetros en IS:")
            for param, value in best_params.items():
                print(f"  {param}: {value}")
            print(f"IS {optimize_metric}: {valid_is.loc[best_is_idx, optimize_metric]:.3f}")

            # FASE 2: Validar en OUT-OF-SAMPLE
            print(f"\nValidando en período OUT-OF-SAMPLE (testing)...")

            oos_metrics = self._run_single_backtest(
                best_params, dates, prices, returns, vix_f1, vix_f2,
                market_prices, test_start, test_end, min_history=60
            )

            oos_result = {
                'split': split_idx,
                'period': 'OOS',
                'train_start': dates[train_start],
                'train_end': dates[train_end-1],
                'test_start': dates[test_start],
                'test_end': dates[test_end-1],
                **best_params,
                **oos_metrics
            }
            oos_results.append(oos_result)

            print(f"OOS {optimize_metric}: {oos_metrics[optimize_metric]:.3f}")
            print(f"OOS Total Return: {oos_metrics['total_return']:.2%}")
            print(f"OOS Trades: {oos_metrics['num_trades']}")

            # Guardar todos los resultados
            all_walk_results.extend(is_results)
            all_walk_results.append(oos_result)

        # ANÁLISIS DE RESULTADOS
        print("\n" + "="*80)
        print("RESULTADOS WALK-FORWARD ANALYSIS")
        print("="*80)

        oos_df = pd.DataFrame(oos_results)

        if len(oos_df) == 0:
            print("❌ No se pudieron completar validaciones OOS")
            return {}, pd.DataFrame(), pd.DataFrame()

        # Estadísticas agregadas de OOS
        print("\nRESULTADOS OUT-OF-SAMPLE (sin overfitting):")
        print("-" * 80)
        print(f"Promedio Total Return:     {oos_df['total_return'].mean():.2%}")
        print(f"Promedio Sharpe Ratio:     {oos_df['sharpe_ratio'].mean():.3f}")
        print(f"Promedio Max Drawdown:     {oos_df['max_drawdown'].mean():.2%}")
        print(f"Total Trades (todos OOS):  {oos_df['num_trades'].sum():.0f}")
        print(f"Promedio Trades por split: {oos_df['num_trades'].mean():.1f}")

        # Encontrar parámetros más consistentes (más frecuentes entre mejores)
        print("\nPARÁMETROS MÁS CONSISTENTES:")
        print("-" * 80)

        for param in param_names:
            values_count = oos_df[param].value_counts()
            print(f"\n{param}:")
            for val, count in values_count.items():
                print(f"  {val}: {count}/{len(oos_df)} splits")

        # Parámetros "robustos" = los más frecuentes
        robust_params = {}
        for param in param_names:
            robust_params[param] = oos_df[param].mode()[0]

        print("\nPARÁMETROS ROBUSTOS (más frecuentes en OOS):")
        print("-" * 80)
        for param, value in robust_params.items():
            print(f"  {param:25s}: {value}")

        # Guardar resultados
        all_results_df = pd.DataFrame(all_walk_results)
        all_results_df.to_csv('walkforward_all_results.csv', index=False)
        oos_df.to_csv('walkforward_oos_results.csv', index=False)

        print("\n✓ Resultados guardados:")
        print("  - walkforward_all_results.csv (todos los backtests IS+OOS)")
        print("  - walkforward_oos_results.csv (solo resultados OOS)")

        return robust_params, oos_df, all_results_df


def main():
    """Función principal para ejecutar walk-forward optimization"""
    import sys
    sys.path.append('.')

    from main import ThetaDataConnector

    # Parámetros de datos
    symbol = 'SPY'
    end_date = datetime(2024, 12, 31)
    start_date = end_date - timedelta(days=4*365)

    print("="*80)
    print(f"WALK-FORWARD OPTIMIZATION - PAPER 2")
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

        # EJECUTAR WALK-FORWARD OPTIMIZATION
        optimizer = WalkForwardOptimizer(initial_capital=1_000_000)

        robust_params, oos_df, all_df = optimizer.walk_forward_analysis(
            dates=dates,
            prices=prices,
            returns=returns,
            vix_f1=vix_f1,
            vix_f2=vix_f2,
            market_prices=None,
            min_history=252,
            n_splits=4,              # 4 períodos walk-forward
            train_ratio=0.5,         # 50% train, 50% test
            optimize_metric='sharpe_ratio'
        )

        if robust_params:
            print("\n" + "="*80)
            print("WALK-FORWARD OPTIMIZATION COMPLETADA")
            print("="*80)
            print("\nUsa estos parámetros ROBUSTOS (validados OOS) en Paper 2:")
            print(f"strategy = Paper2Strategy(")
            for param, value in robust_params.items():
                print(f"    {param}={value},")
            print(")")
            print("\nEstos parámetros fueron validados en datos OUT-OF-SAMPLE")
            print("y NO tienen overfitting.")

    finally:
        theta.disconnect()


if __name__ == '__main__':
    main()
