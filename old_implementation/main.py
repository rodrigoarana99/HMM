#cspell:disable
"""
Main - Orquestación del Sistema de Trading
VersiÃ³n 3.0 - Comparación de 3 Estrategias Separadas

NUEVO: Usa --compare para ejecutar las 3 estrategias y compararlas
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import sys
import warnings
from io import StringIO
import json

# Dependencias externas
import requests
try:
    import yfinance as yf
except ImportError:
    print("ADVERTENCIA: 'yfinance' no está instalado. El fallback de CBOE/Yahoo no funcionará.")
    yf = None

# Módulos del proyecto
from signals import IntegratedSignalGenerator, calculate_vix_term_structure
from evaluation import PerformanceMetrics, print_metrics_report


class ThetaDataConnector:
    """Conector para Thetadata (igual que antes)"""
    
    def __init__(self, host: str = 'localhost', port: int = 25510):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}/v2"
        self.session = requests.Session()
        self.timeout = 60
        
    def connect(self):
        print(f"--- Asumiendo conexión a Thetadata en {self.host}:{self.port} ---")
        return True
    
    def _parse_response(self, text: str, symbol: str, date_str: str) -> Optional[pd.DataFrame]:
        """Parse JSON o CSV con detección de errores"""
        if not text or not text.strip():
            return None
        
        text = text.strip()
        
        # Intentar JSON
        if text.startswith('{'):
            try:
                data = json.loads(text)
                if 'error' in data or 'error_type' in data:
                    error_msg = data.get('error', data.get('error_msg', 'Unknown'))
                    warnings.warn(f"{symbol} {date_str}: JSON error - {error_msg}")
                    return None
                
                if 'header' in data and 'response' in data:
                    df = pd.DataFrame(data['response'], columns=data['header']['format'])
                    return df
            except Exception as e:
                warnings.warn(f"{symbol} {date_str}: JSON parse failed - {e}")
        
        # Intentar CSV
        try:
            df = pd.read_csv(StringIO(text))
            
            if 'error_type' in df.columns or 'error_msg' in df.columns:
                error_type = df['error_type'].iloc[0] if 'error_type' in df.columns else 'UNKNOWN'
                error_msg = df['error_msg'].iloc[0] if 'error_msg' in df.columns else 'Unknown'
                warnings.warn(f"{symbol} {date_str}: CSV error - {error_type}: {error_msg}")
                return None
            
            if df.empty or 'date' not in df.columns:
                 warnings.warn(f"{symbol} {date_str}: CSV parseado pero inválido o vacío")
                 return None

            return df
        except Exception as e:
            warnings.warn(f"{symbol} {date_str}: CSV parse error - {e}")
            return None
    
    def get_market_data_eod(self, symbol: str, start_date: datetime, 
                           end_date: datetime) -> pd.DataFrame:
        """Obtiene datos EOD"""
        url = f"{self.base_url}/hist/stock/eod"
        start_str = start_date.strftime('%Y%m%d')
        end_str = end_date.strftime('%Y%m%d')
        params = {
            'root': symbol,
            'start_date': start_str,
            'end_date': end_str,
            'use_csv': 'true'
        }
        
        try:
            print(f"Descargando {symbol} (EOD) desde {start_str} hasta {end_str}...")
            response = self.session.get(url, params=params, timeout=self.timeout)
            
            if response.status_code != 200:
                warnings.warn(f"{symbol} (EOD): HTTP {response.status_code}")
                return pd.DataFrame()
            
            df = self._parse_response(response.text, symbol, f"{start_str}-{end_str}")
            
            if df is None or df.empty:
                warnings.warn(f"{symbol} (EOD): No valid data after parsing")
                return pd.DataFrame()
            
            column_mapping = {'close': 'close', 'date': 'date'}
            df = df.rename(columns=column_mapping)

            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
            else:
                 warnings.warn(f"{symbol} (EOD): 'date' column missing")
                 return pd.DataFrame()
            
            df = df[['date', 'close']]
            print(f"✓ Descargados {len(df)} períodos para {symbol}")
            return df

        except Exception as e:
            warnings.warn(f"Error obteniendo datos (EOD) para {symbol}: {e}")
            return pd.DataFrame()

    def get_market_fallback(self, start_date: datetime, 
                           end_date: datetime) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fallback a CBOE si Thetadata falla"""
        print("\n--- FALLBACK (CBOE/YAHOO) ---")
        if yf is None:
            warnings.warn("yfinance no está instalado")
            return pd.DataFrame(), pd.DataFrame()
            
        try:
            # VIX desde CBOE
            print("Fallback: Descargando VIX desde CBOE...")
            vix_url = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
            response_vix = self.session.get(vix_url, timeout=30)
            vix_df = pd.read_csv(StringIO(response_vix.text))
            vix_df['date'] = pd.to_datetime(vix_df['DATE'])
            vix_df = vix_df.rename(columns={'CLOSE': 'close'})
            vix_df = vix_df[(vix_df['date'] >= start_date) & (vix_df['date'] <= end_date)]
            vix_df = vix_df[['date', 'close']]

            # VXV (VIX3M) desde CBOE
            print("Fallback: Descargando VXV desde CBOE...")
            vxv_url = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX3M_History.csv"
            response_vxv = self.session.get(vxv_url, timeout=30)
            vxv_df = pd.read_csv(StringIO(response_vxv.text))
            vxv_df['date'] = pd.to_datetime(vxv_df['DATE'])
            vxv_df = vxv_df.rename(columns={'CLOSE': 'close'})
            vxv_df = vxv_df[(vxv_df['date'] >= start_date) & (vxv_df['date'] <= end_date)]
            vxv_df = vxv_df[['date', 'close']]
            
            if vix_df.empty or vxv_df.empty:
                warnings.warn("Fallback de CBOE falló")
                return pd.DataFrame(), pd.DataFrame()
            
            print(f"✓ Fallback: {len(vix_df)} días de VIX, {len(vxv_df)} días de VXV")
            return vix_df, vxv_df
            
        except Exception as e:
            warnings.warn(f"Fallback (CBOE/Yahoo) falló: {e}")
            return pd.DataFrame(), pd.DataFrame()

    def get_vix_futures(self, start_date: datetime, 
                       end_date: datetime) -> Dict[str, pd.DataFrame]:
        """Obtiene VIX y VXV con fallback"""
        print("\n" + "="*70)
        print("OBTENIENDO VOLATILIDAD (VIX y VXV)")
        print("="*70)
        
        # Intentar Thetadata
        vix_f1_data = self.get_market_data_eod('VIX', start_date, end_date)
        vix_f2_data = self.get_market_data_eod('VXV', start_date, end_date)
        
        # Fallback si necesario
        if vix_f1_data.empty or vix_f2_data.empty:
            warnings.warn("Usando Fallback para VIX/VXV...")
            vix_f1_data, vix_f2_data = self.get_market_fallback(start_date, end_date)
        
        if vix_f1_data.empty:
            warnings.warn("ERROR: No se pudieron obtener datos de VIX (F1)")
        if vix_f2_data.empty:
            warnings.warn("ERROR: No se pudieron obtener datos de VXV (F2)")
        
        print("="*70 + "\n")
        
        return {'F1': vix_f1_data, 'F2': vix_f2_data}
    
    def disconnect(self):
        self.session.close()


def run_comparison_mode(symbol: str = 'SPY',
                       start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None):
    """
    NUEVO: Ejecuta las 3 estrategias y compara resultados
    """
    if end_date is None:
        end_date = datetime(2024, 12, 31)
    if start_date is None:
        start_date = end_date - timedelta(days=4*365)
    
    print("="*80)
    print(f"COMPARACIÓN DE 3 ESTRATEGIAS - {symbol}")
    print("="*80)
    print(f"Período: {start_date.date()} a {end_date.date()}")
    print()
    
    # Obtener datos
    theta = ThetaDataConnector()
    if not theta.connect():
        print("No se pudo conectar a Thetadata. Abortando.")
        return None
    
    try:
        # Obtener SPY
        print(f"\nDescargando {symbol}...")
        stock_data = theta.get_market_data_eod(symbol, start_date, end_date)
        
        if stock_data.empty:
            print(f"No se pudieron obtener datos para {symbol}. Abortando.")
            return None
        
        stock_data['returns'] = stock_data['close'].pct_change()
        stock_data = stock_data.dropna(subset=['returns'])
        print(f"✓ Datos de {symbol}: {len(stock_data)} días")
        
        # Obtener VIX/VXV
        vix_futures = theta.get_vix_futures(start_date, end_date)
        vix_f1_data = vix_futures['F1']
        vix_f2_data = vix_futures['F2']
        
        if vix_f1_data.empty or vix_f2_data.empty:
            print("No se pudieron obtener datos de VIX/VXV. Abortando.")
            return None

        # Alinear datos
        print("Alineando datos...")
        vix_f1_data = vix_f1_data[['date', 'close']].rename(columns={'close': 'vix_f1'})
        vix_f2_data = vix_f2_data[['date', 'close']].rename(columns={'close': 'vix_f2'})
        
        merged_data = stock_data.merge(vix_f1_data, on='date', how='left')
        merged_data = merged_data.merge(vix_f2_data, on='date', how='left')
        
        merged_data['vix_f1'] = merged_data['vix_f1'].fillna(method='ffill', limit=5)
        merged_data['vix_f2'] = merged_data['vix_f2'].fillna(method='ffill', limit=5)
        merged_data = merged_data.dropna(subset=['vix_f1', 'vix_f2', 'close', 'returns'])
        
        if merged_data.empty:
            print("No quedaron datos después de alinear. Abortando.")
            return None

        print(f"✓ Datos alineados: {len(merged_data)} días\n")
        
        # Preparar datos para backtest
        dates = merged_data['date'].tolist()
        prices = {symbol: merged_data['close'].values}
        returns = merged_data['returns'].values
        vix_f1 = merged_data['vix_f1'].values
        vix_f2 = merged_data['vix_f2'].values
        
        # EJECUTAR COMPARACIÓN
        from main_compare import run_all_strategies
        
        results, comparison_df = run_all_strategies(
            dates, prices, returns, vix_f1, vix_f2, market_prices=None
        )
        
        return results, comparison_df
        
    finally:
        theta.disconnect()


def main():
    """Función principal"""
    print("\nSISTEMA DE TRADING - COMPARACIÓN DE ESTRATEGIAS")
    print("Versión 3.0 - Papers Separados\n")
    
    # Parsear argumentos
    if '--compare' in sys.argv:
        symbol = 'SPY'
        if len(sys.argv) > 2:
            symbol = sys.argv[2]
        
        print(f"Modo COMPARACIÓN: Ejecutando las 3 estrategias para {symbol}\n")
        results = run_comparison_mode(symbol=symbol)
        
        if results is None:
            print("\n" + "="*70)
            print("EJECUCIÓN ABORTADA")
            print("="*70)
        else:
            print("\n" + "="*70)
            print("COMPARACIÓN COMPLETADA")
            print("="*70)
    else:
        print("Uso:")
        print("  python main.py --compare [SYMBOL]")
        print("\nEjemplo:")
        print("  python main.py --compare SPY")
        print("\nEsto ejecutará las 3 estrategias y comparará resultados.")


if __name__ == '__main__':
    main()