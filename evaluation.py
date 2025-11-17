#cspell:disable
"""
Módulo de Evaluación
Calcula métricas de performance del backtest
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats


class PerformanceMetrics:
    """
    Calcula métricas de performance como en los papers
    """
    
    @staticmethod
    def calculate_returns(portfolio_values: np.ndarray) -> np.ndarray:
        """Calcula returns diarios"""
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        return returns
    
    @staticmethod
    def annualized_return(returns: np.ndarray, periods_per_year: int = 252) -> float:
        """
        Calcula return anualizado
        
        Papers usan: (final_value / initial_value) ^ (252 / n_days) - 1
        """
        if len(returns) == 0:
            return 0.0
        
        total_return = np.prod(1 + returns)
        n_periods = len(returns)
        
        if n_periods < periods_per_year:
            # No anualizar si hay menos de 1 año
            return total_return - 1
        
        annualized = total_return ** (periods_per_year / n_periods) - 1
        return annualized
    
    @staticmethod
    def annualized_volatility(returns: np.ndarray, periods_per_year: int = 252) -> float:
        """Calcula volatilidad anualizada"""
        if len(returns) == 0:
            return 0.0
        return np.std(returns) * np.sqrt(periods_per_year)
    
    @staticmethod
    def sharpe_ratio(returns: np.ndarray, 
                    risk_free_rate: float = 0.02,
                    periods_per_year: int = 252) -> float:
        """
        Calcula Sharpe Ratio como en los papers
        Sharpe = (Return - RiskFree) / Volatility
        """
        if len(returns) == 0:
            return 0.0
        
        ann_return = PerformanceMetrics.annualized_return(returns, periods_per_year)
        ann_vol = PerformanceMetrics.annualized_volatility(returns, periods_per_year)
        
        if ann_vol == 0:
            return 0.0
        
        sharpe = (ann_return - risk_free_rate) / ann_vol
        return sharpe
    
    @staticmethod
    def max_drawdown(portfolio_values: np.ndarray) -> Tuple[float, int, int]:
        """
        Calcula maximum drawdown
        
        Returns:
            (max_dd, start_idx, end_idx)
        """
        if len(portfolio_values) == 0:
            return 0.0, 0, 0
        
        # Calcular running maximum
        running_max = np.maximum.accumulate(portfolio_values)
        
        # Calcular drawdown
        drawdown = (portfolio_values - running_max) / running_max
        
        # Encontrar máximo drawdown
        max_dd = np.min(drawdown)
        end_idx = np.argmin(drawdown)
        
        # Encontrar inicio (último peak antes del máximo drawdown)
        start_idx = np.argmax(portfolio_values[:end_idx+1]) if end_idx > 0 else 0
        
        return abs(max_dd), start_idx, end_idx
    
    @staticmethod
    def calmar_ratio(returns: np.ndarray, 
                    portfolio_values: np.ndarray,
                    periods_per_year: int = 252) -> float:
        """
        Calmar Ratio = Annualized Return / Max Drawdown
        """
        ann_return = PerformanceMetrics.annualized_return(returns, periods_per_year)
        max_dd, _, _ = PerformanceMetrics.max_drawdown(portfolio_values)
        
        if max_dd == 0:
            return np.inf if ann_return > 0 else 0.0
        
        return ann_return / max_dd
    
    @staticmethod
    def sortino_ratio(returns: np.ndarray,
                     risk_free_rate: float = 0.02,
                     periods_per_year: int = 252) -> float:
        """
        Sortino Ratio - similar a Sharpe pero solo considera downside volatility
        """
        if len(returns) == 0:
            return 0.0
        
        ann_return = PerformanceMetrics.annualized_return(returns, periods_per_year)
        
        # Downside deviation (solo returns negativos)
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0:
            return np.inf if ann_return > 0 else 0.0
        
        downside_vol = np.std(negative_returns) * np.sqrt(periods_per_year)
        
        if downside_vol == 0:
            return np.inf if ann_return > 0 else 0.0
        
        return (ann_return - risk_free_rate) / downside_vol
    
    @staticmethod
    def win_rate(returns: np.ndarray) -> float:
        """Porcentaje de días positivos"""
        if len(returns) == 0:
            return 0.0
        return np.sum(returns > 0) / len(returns)
    
    @staticmethod
    def profit_factor(returns: np.ndarray) -> float:
        """
        Profit Factor = Sum(Gains) / Sum(Losses)
        """
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        
        if losses == 0:
            return np.inf if gains > 0 else 0.0
        
        return gains / losses
    
    @staticmethod
    def calculate_all_metrics(portfolio_df: pd.DataFrame,
                             benchmark_returns: Optional[np.ndarray] = None,
                             risk_free_rate: float = 0.02) -> Dict:
        """
        Calcula todas las métricas de performance
        
        Args:
            portfolio_df: DataFrame con columna 'total_value'
            benchmark_returns: Returns del benchmark (e.g., Buy&Hold)
            risk_free_rate: Tasa libre de riesgo anualizada
            
        Returns:
            Dict con todas las métricas
        """
        if len(portfolio_df) == 0:
            return {}
        
        values = portfolio_df['total_value'].values
        returns = PerformanceMetrics.calculate_returns(values)
        
        # Métricas básicas
        metrics = {
            'initial_value': values[0],
            'final_value': values[-1],
            'total_return': (values[-1] - values[0]) / values[0],
            'annualized_return': PerformanceMetrics.annualized_return(returns),
            'annualized_volatility': PerformanceMetrics.annualized_volatility(returns),
            'sharpe_ratio': PerformanceMetrics.sharpe_ratio(returns, risk_free_rate),
            'sortino_ratio': PerformanceMetrics.sortino_ratio(returns, risk_free_rate),
            'win_rate': PerformanceMetrics.win_rate(returns),
            'profit_factor': PerformanceMetrics.profit_factor(returns)
        }
        
        # Maximum drawdown
        max_dd, dd_start, dd_end = PerformanceMetrics.max_drawdown(values)
        metrics['max_drawdown'] = max_dd
        metrics['max_drawdown_start'] = portfolio_df.iloc[dd_start]['timestamp'] if dd_start < len(portfolio_df) else None
        metrics['max_drawdown_end'] = portfolio_df.iloc[dd_end]['timestamp'] if dd_end < len(portfolio_df) else None
        
        # Calmar ratio
        metrics['calmar_ratio'] = PerformanceMetrics.calmar_ratio(returns, values)
        
        # Comparación con benchmark
        if benchmark_returns is not None:
            # Asegurar que tengan la misma longitud
            min_len = min(len(returns), len(benchmark_returns))
            strategy_ret = returns[:min_len]
            bench_ret = benchmark_returns[:min_len]
            
            # Alpha y Beta
            covariance = np.cov(strategy_ret, bench_ret)[0, 1]
            benchmark_variance = np.var(bench_ret)
            
            if benchmark_variance > 0:
                beta = covariance / benchmark_variance
                
                strategy_ann_ret = PerformanceMetrics.annualized_return(strategy_ret)
                benchmark_ann_ret = PerformanceMetrics.annualized_return(bench_ret)
                
                alpha = strategy_ann_ret - (risk_free_rate + beta * (benchmark_ann_ret - risk_free_rate))
                
                metrics['beta'] = beta
                metrics['alpha'] = alpha
                
                # Information Ratio
                active_returns = strategy_ret - bench_ret
                tracking_error = np.std(active_returns) * np.sqrt(252)
                
                if tracking_error > 0:
                    metrics['information_ratio'] = (strategy_ann_ret - benchmark_ann_ret) / tracking_error
                else:
                    metrics['information_ratio'] = 0.0
            
            # Comparación simple
            metrics['benchmark_return'] = PerformanceMetrics.annualized_return(bench_ret)
            metrics['excess_return'] = metrics['annualized_return'] - metrics['benchmark_return']
        
        return metrics


class RegimeAnalysis:
    """
    Análisis de performance por régimen de mercado
    """
    
    @staticmethod
    def analyze_by_regime(portfolio_df: pd.DataFrame,
                         trades_df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Analiza performance separado por régimen
        
        Returns:
            Dict[regime] -> metrics
        """
        if 'regime' not in portfolio_df.columns:
            return {}
        
        results = {}
        
        for regime in portfolio_df['regime'].unique():
            regime_data = portfolio_df[portfolio_df['regime'] == regime].copy()
            
            if len(regime_data) < 2:
                continue
            
            values = regime_data['total_value'].values
            returns = PerformanceMetrics.calculate_returns(values)
            
            # Calcular métricas para este régimen
            metrics = {
                'days': len(regime_data),
                'total_return': (values[-1] - values[0]) / values[0],
                'avg_daily_return': np.mean(returns),
                'volatility': np.std(returns),
                'sharpe': PerformanceMetrics.sharpe_ratio(returns) if len(returns) > 10 else 0.0,
                'max_drawdown': PerformanceMetrics.max_drawdown(values)[0],
                'win_rate': PerformanceMetrics.win_rate(returns)
            }
            
            # Trades en este régimen
            if len(trades_df) > 0 and 'regime' in trades_df.columns:
                regime_trades = trades_df[trades_df['regime'] == regime]
                metrics['num_trades'] = len(regime_trades)
                
                if len(regime_trades) > 0:
                    buy_trades = regime_trades[regime_trades['action'] == 'BUY']
                    sell_trades = regime_trades[regime_trades['action'].isin(['SELL', 'CLOSE'])]
                    metrics['num_buys'] = len(buy_trades)
                    metrics['num_sells'] = len(sell_trades)
            
            results[regime] = metrics
        
        return results
    
    @staticmethod
    def regime_transitions(portfolio_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analiza transiciones entre regímenes
        """
        if 'regime' not in portfolio_df.columns or len(portfolio_df) < 2:
            return pd.DataFrame()
        
        # Detectar cambios de régimen
        regime_changes = portfolio_df['regime'] != portfolio_df['regime'].shift(1)
        transitions = portfolio_df[regime_changes].copy()
        
        # Agregar régimen previo
        transitions['prev_regime'] = portfolio_df['regime'].shift(1)[regime_changes]
        
        return transitions[['timestamp', 'prev_regime', 'regime', 'total_value']]


class ComparativeAnalysis:
    """
    Compara múltiples estrategias
    """
    
    @staticmethod
    def compare_strategies(results: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compara múltiples estrategias lado a lado
        
        Args:
            results: Dict[strategy_name] -> metrics_dict
            
        Returns:
            DataFrame comparativo
        """
        if not results:
            return pd.DataFrame()
        
        # Crear DataFrame
        df = pd.DataFrame(results).T
        
        # Ordenar por Sharpe Ratio
        if 'sharpe_ratio' in df.columns:
            df = df.sort_values('sharpe_ratio', ascending=False)
        
        return df
    
    @staticmethod
    def plot_comparison(portfolio_dfs: Dict[str, pd.DataFrame],
                       normalize: bool = True) -> None:
        """
        Grafica comparación de portfolios
        
        Args:
            portfolio_dfs: Dict[strategy_name] -> portfolio_df
            normalize: Si normalizar a 1.0 al inicio
        """
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 6))
            
            for name, df in portfolio_dfs.items():
                if len(df) == 0:
                    continue
                
                values = df['total_value'].values
                dates = pd.to_datetime(df['timestamp'])
                
                if normalize:
                    values = values / values[0]
                
                plt.plot(dates, values, label=name, linewidth=2)
            
            plt.xlabel('Date')
            plt.ylabel('Portfolio Value' + (' (Normalized)' if normalize else ''))
            plt.title('Strategy Comparison')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("matplotlib not available for plotting")


def print_metrics_report(metrics: Dict, title: str = "Performance Metrics") -> None:
    """
    Imprime reporte de métricas de forma legible
    """
    print("\n" + "="*70)
    print(f"{title:^70}")
    print("="*70)
    
    # Formato para diferentes tipos de métricas
    formatters = {
        'total_return': '{:.2%}',
        'annualized_return': '{:.2%}',
        'annualized_volatility': '{:.2%}',
        'max_drawdown': '{:.2%}',
        'win_rate': '{:.2%}',
        'sharpe_ratio': '{:.3f}',
        'sortino_ratio': '{:.3f}',
        'calmar_ratio': '{:.3f}',
        'profit_factor': '{:.3f}',
        'information_ratio': '{:.3f}',
        'alpha': '{:.2%}',
        'beta': '{:.3f}',
        'excess_return': '{:.2%}',
        'initial_value': '${:,.2f}',
        'final_value': '${:,.2f}'
    }
    
    for key, value in metrics.items():
        if pd.isna(value) or value is None:
            continue
        
        # Formatear según el tipo de métrica
        if key in formatters:
            formatted_value = formatters[key].format(value)
        elif isinstance(value, (int, np.integer)):
            formatted_value = f'{value:,}'
        elif isinstance(value, (float, np.floating)):
            formatted_value = f'{value:.4f}'
        else:
            formatted_value = str(value)
        
        # Imprimir con buen formato
        key_display = key.replace('_', ' ').title()
        print(f"{key_display:.<40} {formatted_value:>25}")
    
    print("="*70 + "\n")