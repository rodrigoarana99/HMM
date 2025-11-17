"""
Script alternativo para ejecutar backtest usando Yahoo Finance directamente
Útil cuando Thetadata no está disponible
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import warnings

from signals import IntegratedSignalGenerator, calculate_vix_term_structure
from strategy import TradingStrategy
from backtester import BacktestEngine
from evaluation import PerformanceMetrics, print_metrics_report


def download_data_yfinance(symbol: str = 'SPY',
                           start_date: datetime = None,
                           end_date: datetime = None):
    """
    Descarga datos usando Yahoo Finance
    """
    if end_date is None:
        end_date = datetime.now()
    if start_date is None:
        start_date = end_date - timedelta(days=4*365)

    print("="*70)
    print(f"DESCARGANDO DATOS DE YAHOO FINANCE PARA {symbol}")
    print("="*70)
    print(f"Período: {start_date.date()} a {end_date.date()}")
    print()

    # 1. Descargar SPY
    print(f"Descargando {symbol}...")
    spy = yf.download(symbol, start=start_date, end=end_date, progress=False)

    if spy.empty:
        print(f"[!] No se pudieron descargar datos de {symbol}")
        return None

    # Resetear índice y seleccionar columna Close
    spy_data = spy[['Close']].copy()
    spy_data.columns = ['close']
    spy_data = spy_data.reset_index()
    spy_data.columns = ['date', 'close']

    spy_data['returns'] = spy_data['close'].pct_change()
    spy_data = spy_data.dropna(subset=['returns'])

    print(f"  OK {len(spy_data)} dias de {symbol}")

    # 2. Descargar VIX
    print("Descargando ^VIX...")
    vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)

    if vix.empty:
        print("[!] No se pudieron descargar datos de VIX")
        return None

    vix_data = vix[['Close']].copy()
    vix_data.columns = ['vix_f1']
    vix_data = vix_data.reset_index()
    vix_data.columns = ['date', 'vix_f1']

    print(f"  OK {len(vix_data)} dias de VIX")

    # 3. Descargar VIX3M (VXV)
    print("Descargando ^VIX3M...")
    vxv = yf.download('^VIX3M', start=start_date, end=end_date, progress=False)

    if vxv.empty:
        print("[!] No se pudieron descargar datos de VIX3M")
        # Usar aproximación: VXV ≈ VIX * 1.05 (contango típico)
        print("  Usando aproximacion: VXV = VIX * 1.05")
        vxv_data = pd.DataFrame({
            'date': vix_data['date'],
            'vix_f2': vix_data['vix_f1'] * 1.05
        })
    else:
        vxv_data = vxv[['Close']].copy()
        vxv_data.columns = ['vix_f2']
        vxv_data = vxv_data.reset_index()
        vxv_data.columns = ['date', 'vix_f2']
        print(f"  OK {len(vxv_data)} dias de VIX3M")

    # 4. Merge de datos
    print("\nAlineando datos...")
    merged_data = spy_data.merge(vix_data, on='date', how='left')
    merged_data = merged_data.merge(vxv_data, on='date', how='left')

    # Forward fill para fines de semana/feriados
    merged_data['vix_f1'] = merged_data['vix_f1'].ffill(limit=5)
    merged_data['vix_f2'] = merged_data['vix_f2'].ffill(limit=5)

    # Eliminar NaNs
    merged_data = merged_data.dropna(subset=['vix_f1', 'vix_f2', 'close', 'returns'])

    if merged_data.empty:
        print("[!] No quedaron datos después del alineamiento")
        return None

    print(f"  OK {len(merged_data)} días alineados")
    print("="*70)

    return merged_data


def run_backtest_yfinance(symbol: str = 'SPY',
                          start_date: datetime = None,
                          end_date: datetime = None):
    """
    Ejecuta backtest usando datos de Yahoo Finance
    """
    # Descargar datos
    merged_data = download_data_yfinance(symbol, start_date, end_date)

    if merged_data is None:
        print("No se pudieron obtener los datos")
        return None, None, None

    # Preparar datos para backtest
    dates = merged_data['date'].tolist()
    prices = {symbol: merged_data['close'].values}
    returns = merged_data['returns'].values
    vix_f1 = merged_data['vix_f1'].values
    vix_f2 = merged_data['vix_f2'].values

    # Inicializar estrategia
    print("\nInicializando estrategia...")
    signal_generator = IntegratedSignalGenerator()
    strategy = TradingStrategy(
        risk_threshold=0.90,
        position_size=1.0,
        use_leverage=False
    )

    # Ejecutar backtest
    print("\n" + "="*70)
    print("EJECUTANDO BACKTEST")
    print("="*70)

    engine = BacktestEngine(
        strategy=strategy,
        signal_generator=signal_generator,
        initial_capital=1_000_000
    )

    min_history = min(252, len(dates) // 2)

    portfolio_df, trades_df, summary = engine.run(
        dates=dates,
        prices=prices,
        returns=returns,
        vix_f1=vix_f1,
        vix_f2=vix_f2,
        market_prices=None,
        min_history=min_history
    )

    # Calcular métricas
    spy_final = prices[symbol][-1]
    spy_initial = prices[symbol][min_history]
    bh_return = (spy_final - spy_initial) / spy_initial

    benchmark_returns = np.diff(prices[symbol][min_history:]) / prices[symbol][min_history:-1]

    metrics = PerformanceMetrics.calculate_all_metrics(
        portfolio_df,
        benchmark_returns=benchmark_returns,
        risk_free_rate=0.02
    )

    # Imprimir resultados
    print_metrics_report(metrics, f"ESTRATEGIA CON YAHOO FINANCE - {symbol}")

    print(f"\nCOMPARACION CON BUY & HOLD DE {symbol}:")
    print(f"{'Strategy Total Return':<40} {metrics['total_return']:>25.2%}")
    print(f"{'Buy & Hold Total Return':<40} {bh_return:>25.2%}")
    print(f"{'Excess Return':<40} {metrics['total_return'] - bh_return:>25.2%}")
    print(f"{'Strategy Sharpe Ratio':<40} {metrics['sharpe_ratio']:>25.3f}")

    # Guardar resultados
    portfolio_df.to_csv(f'portfolio_history_{symbol}_yfinance.csv', index=False)
    trades_df.to_csv(f'trades_history_{symbol}_yfinance.csv', index=False)

    print(f"\nArchivos guardados:")
    print(f"  - portfolio_history_{symbol}_yfinance.csv")
    print(f"  - trades_history_{symbol}_yfinance.csv")

    return portfolio_df, trades_df, metrics


if __name__ == '__main__':
    import sys

    symbol = 'SPY'
    if len(sys.argv) > 1:
        symbol = sys.argv[1]

    print("\nSISTEMA DE TRADING HMM + MEAN REVERSION")
    print("Modo: YAHOO FINANCE\n")

    results = run_backtest_yfinance(symbol=symbol)

    if results and results[0] is not None:
        print("\n" + "="*70)
        print("BACKTEST COMPLETADO")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("EJECUCION ABORTADA")
        print("="*70)
