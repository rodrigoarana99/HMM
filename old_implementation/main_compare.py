#cspell:disable
"""
Comparación de las 3 Estrategias de los Papers
Ejecuta backtest separado para cada una y compara resultados
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import sys

# Importar estrategias
from Hm_p_1 import Paper1Strategy
from momemtum_p_2 import Paper2Strategy
from switching_p_3 import Paper3Strategy

# Importar módulos originales
from signals import IntegratedSignalGenerator
from backtester import Backtest
from evaluation import PerformanceMetrics, print_metrics_report


class StrategyBacktester:
    """
    Backtester genérico que funciona con cualquiera de las 3 estrategias
    """
    
    def __init__(self, strategy, signal_generator, initial_capital: float = 1_000_000):
        self.strategy = strategy
        self.signal_generator = signal_generator
        self.backtest = Backtest(initial_capital=initial_capital)
        self.strategy_name = strategy.__class__.__name__
        
    def run(self,
            dates: List[datetime],
            prices: Dict[str, np.ndarray],
            returns: np.ndarray,
            vix_f1: np.ndarray,
            vix_f2: np.ndarray,
            market_prices=None,
            min_history: int = 252) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Ejecuta backtest para la estrategia específica
        """
        from signals import calculate_vix_term_structure
        
        vix_term_struct = calculate_vix_term_structure(vix_f1, vix_f2)
        main_asset = list(prices.keys())[0]
        main_prices = prices[main_asset]
        
        print(f"\n{'='*70}")
        print(f"BACKTESTING: {self.strategy_name}")
        print(f"{'='*70}")
        print(f"Period: {dates[min_history]} to {dates[-1]}")
        print(f"Initial capital: ${self.backtest.initial_capital:,.2f}\n")
        
        # Loop principal
        for i in range(min_history, len(dates)):
            current_date = dates[i]
            
            # Generar señales
            hist_returns = returns[:i+1]
            hist_vix_ts = vix_term_struct[:i+1]
            hist_prices = main_prices[:i+1]
            hist_market = market_prices[:i+1] if market_prices is not None else None
            
            signals = self.signal_generator.generate_signals(
                prices=hist_prices,
                returns=hist_returns,
                vix_term_structure=hist_vix_ts,
                market_prices=hist_market
            )
            
            # Decisión de la estrategia
            decision = self.strategy.calculate_position(signals)
            
            # Precios actuales
            current_prices = {asset: price_array[i] for asset, price_array in prices.items()}
            
            # Ejecutar trades según la estrategia
            action = decision.get('action', 'HOLD')
            
            if action in ['BUY', 'SELL', 'CLOSE', 'ADJUST']:
                target_size = decision.get('target_size', 0.0)
                portfolio_value = self.backtest.update_portfolio_value(
                    current_date, current_prices, signals['regime']
                )
                target_value = portfolio_value * target_size
                
                self.backtest.execute_trade(
                    timestamp=current_date,
                    action=action,
                    asset=main_asset,
                    price=current_prices[main_asset],
                    target_size=target_value,
                    reasoning=decision.get('reasoning', []),
                    regime=signals['regime']
                )
            
            # Actualizar portfolio
            self.backtest.update_portfolio_value(
                current_date, current_prices, signals['regime']
            )
            
            # Progress
            if i % 252 == 0:
                summary = self.backtest.get_portfolio_summary()
                print(f"{current_date.date()}: ${summary['total_value']:,.2f} | "
                      f"Return={summary['total_return']:.2%} | "
                      f"Regime={signals['regime']}")
        
        # Resultados
        portfolio_df = self.backtest.get_portfolio_history_df()
        trades_df = self.backtest.get_trades_df()
        summary = self.backtest.get_portfolio_summary()
        
        print(f"\n{'='*70}")
        print(f"RESULTS - {self.strategy_name}")
        print(f"{'='*70}")
        print(f"Final Value: ${summary['total_value']:,.2f}")
        print(f"Total Return: {summary['total_return']:.2%}")
        print(f"Trades: {summary['num_trades']}")
        print(f"Commission: ${summary['total_commission']:,.2f}")
        print(f"Slippage: ${summary['total_slippage']:,.2f}")
        
        return portfolio_df, trades_df, summary


def run_all_strategies(dates, prices, returns, vix_f1, vix_f2, market_prices=None):
    """
    Ejecuta las 3 estrategias y compara resultados
    """
    signal_generator = IntegratedSignalGenerator()
    initial_capital = 1_000_000
    min_history = min(252, len(dates) // 2)
    
    results = {}
    
    # ESTRATEGIA 1: HMM Básico (3 estados)
    print("\n" + "="*80)
    print("PAPER 1: HMM BÁSICO (3 Estados)")
    print("="*80)
    
    strategy1 = Paper1Strategy(
        confidence_threshold=0.80,
        position_size=1.0
    )
    
    backtester1 = StrategyBacktester(strategy1, signal_generator, initial_capital)
    portfolio1, trades1, summary1 = backtester1.run(
        dates, prices, returns, vix_f1, vix_f2, market_prices, min_history
    )
    
    # Calcular métricas
    spy_final = prices['SPY'][-1]
    spy_initial = prices['SPY'][min_history]
    bh_return = (spy_final - spy_initial) / spy_initial
    benchmark_returns = np.diff(prices['SPY'][min_history:]) / prices['SPY'][min_history:-1]
    
    metrics1 = PerformanceMetrics.calculate_all_metrics(
        portfolio1, benchmark_returns=benchmark_returns, risk_free_rate=0.02
    )
    
    results['Paper1_HMM'] = {
        'portfolio': portfolio1,
        'trades': trades1,
        'summary': summary1,
        'metrics': metrics1
    }
    
    # ESTRATEGIA 2: Mean Reversion
    print("\n" + "="*80)
    print("PAPER 2: MEAN REVERSION (Buy Losers)")
    print("="*80)
    
    strategy2 = Paper2Strategy(
        entry_threshold=-0.3,
        exit_threshold=0.3,
        position_size=1.0,
        use_momentum_filter=True
    )
    
    signal_generator2 = IntegratedSignalGenerator()
    backtester2 = StrategyBacktester(strategy2, signal_generator2, initial_capital)
    portfolio2, trades2, summary2 = backtester2.run(
        dates, prices, returns, vix_f1, vix_f2, market_prices, min_history
    )
    
    metrics2 = PerformanceMetrics.calculate_all_metrics(
        portfolio2, benchmark_returns=benchmark_returns, risk_free_rate=0.02
    )
    
    results['Paper2_MeanReversion'] = {
        'portfolio': portfolio2,
        'trades': trades2,
        'summary': summary2,
        'metrics': metrics2
    }
    
    # ESTRATEGIA 3: Regime Switching
    print("\n" + "="*80)
    print("PAPER 3: REGIME SWITCHING (Stock/Treasury)")
    print("="*80)
    
    strategy3 = Paper3Strategy(
        bear_threshold=0.90,
        bull_threshold=0.70,
        use_leverage=False
    )
    
    signal_generator3 = IntegratedSignalGenerator()
    backtester3 = StrategyBacktester(strategy3, signal_generator3, initial_capital)
    portfolio3, trades3, summary3 = backtester3.run(
        dates, prices, returns, vix_f1, vix_f2, market_prices, min_history
    )
    
    metrics3 = PerformanceMetrics.calculate_all_metrics(
        portfolio3, benchmark_returns=benchmark_returns, risk_free_rate=0.02
    )
    
    results['Paper3_RegimeSwitching'] = {
        'portfolio': portfolio3,
        'trades': trades3,
        'summary': summary3,
        'metrics': metrics3
    }
    
    # COMPARACIÓN FINAL
    print("\n" + "="*80)
    print("COMPARATIVE RESULTS")
    print("="*80)
    
    comparison_data = {
        'Strategy': ['Buy&Hold', 'Paper1_HMM', 'Paper2_MeanRev', 'Paper3_RegimeSw'],
        'Total Return': [
            bh_return,
            metrics1['total_return'],
            metrics2['total_return'],
            metrics3['total_return']
        ],
        'Ann. Return': [
            (1 + bh_return) ** (252 / len(benchmark_returns)) - 1,
            metrics1['annualized_return'],
            metrics2['annualized_return'],
            metrics3['annualized_return']
        ],
        'Volatility': [
            np.std(benchmark_returns) * np.sqrt(252),
            metrics1['annualized_volatility'],
            metrics2['annualized_volatility'],
            metrics3['annualized_volatility']
        ],
        'Sharpe': [
            (((1 + bh_return) ** (252 / len(benchmark_returns)) - 1) - 0.02) / (np.std(benchmark_returns) * np.sqrt(252)),
            metrics1['sharpe_ratio'],
            metrics2['sharpe_ratio'],
            metrics3['sharpe_ratio']
        ],
        'Max DD': [
            PerformanceMetrics.max_drawdown(prices['SPY'][min_history:])[0],
            metrics1['max_drawdown'],
            metrics2['max_drawdown'],
            metrics3['max_drawdown']
        ],
        'Num Trades': [
            0,
            summary1['num_trades'],
            summary2['num_trades'],
            summary3['num_trades']
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\n")
    print(comparison_df.to_string(index=False))
    print("\n")
    
    # Guardar resultados
    for name, data in results.items():
        data['portfolio'].to_csv(f'{name}_portfolio.csv', index=False)
        data['trades'].to_csv(f'{name}_trades.csv', index=False)
    
    comparison_df.to_csv('strategy_comparison.csv', index=False)
    
    print("Files saved:")
    print("  - Paper1_HMM_portfolio.csv / Paper1_HMM_trades.csv")
    print("  - Paper2_MeanReversion_portfolio.csv / Paper2_MeanReversion_trades.csv")
    print("  - Paper3_RegimeSwitching_portfolio.csv / Paper3_RegimeSwitching_trades.csv")
    print("  - strategy_comparison.csv")
    
    return results, comparison_df


if __name__ == '__main__':
    # Este script debería ser llamado desde main.py después de cargar los datos
    print("Use main.py con --compare flag para ejecutar todas las estrategias")