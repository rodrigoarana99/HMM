"""
Módulo de Backtesting
Framework para probar estrategias con datos históricos
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field


@dataclass
class Trade:
    """Registro de un trade"""
    timestamp: datetime
    action: str  # 'BUY', 'SELL', 'CLOSE'
    asset: str
    price: float
    size: float
    value: float
    reasoning: List[str] = field(default_factory=list)
    regime: str = ''
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'action': self.action,
            'asset': self.asset,
            'price': self.price,
            'size': self.size,
            'value': self.value,
            'reasoning': '; '.join(self.reasoning),
            'regime': self.regime
        }


@dataclass
class PortfolioSnapshot:
    """Snapshot del portfolio en un momento"""
    timestamp: datetime
    total_value: float
    cash: float
    positions: Dict[str, float]  # asset -> quantity
    position_values: Dict[str, float]  # asset -> value
    returns: float = 0.0
    regime: str = ''
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'total_value': self.total_value,
            'cash': self.cash,
            'returns': self.returns,
            'regime': self.regime,
            **{f'pos_{k}': v for k, v in self.positions.items()},
            **{f'val_{k}': v for k, v in self.position_values.items()}
        }


class Backtest:
    """
    Motor de backtesting que simula trading con datos históricos
    """
    
    def __init__(self, 
                 initial_capital: float = 1_000_000,
                 commission: float = 0.0002,  # 2 cents per share típico
                 slippage: float = 0.0005):    # 0.05% slippage
        """
        Args:
            initial_capital: Capital inicial
            commission: Comisión por trade ($ por acción o % del valor)
            slippage: Slippage estimado (%)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
        # Estado del portfolio
        self.cash = initial_capital
        self.positions = {}  # asset -> quantity
        self.position_values = {}
        
        # Historia
        self.trades: List[Trade] = []
        self.portfolio_history: List[PortfolioSnapshot] = []
        
        # Tracking
        self.total_commission = 0.0
        self.total_slippage = 0.0
        
    def execute_trade(self, 
                     timestamp: datetime,
                     action: str,
                     asset: str,
                     price: float,
                     target_size: float,
                     reasoning: List[str],
                     regime: str = '') -> bool:
        """
        Ejecuta un trade
        
        Args:
            timestamp: Momento del trade
            action: 'BUY', 'SELL', 'CLOSE'
            asset: Nombre del activo
            price: Precio del activo
            target_size: Tamaño objetivo (en quantity o $ dependiendo)
            reasoning: Razones del trade
            regime: Régimen de mercado
            
        Returns:
            True si el trade se ejecutó, False si no había fondos
        """
        current_position = self.positions.get(asset, 0.0)
        
        # Determinar quantity a tradear
        if action == 'BUY':
            # Comprar hasta target_size
            value_to_buy = target_size * price if target_size < 100 else target_size
            quantity = value_to_buy / price
            
            # Aplicar slippage (compramos más caro)
            execution_price = price * (1 + self.slippage)
            cost = quantity * execution_price
            commission = quantity * self.commission
            
            total_cost = cost + commission
            
            if total_cost > self.cash:
                # No hay suficiente efectivo
                return False
            
            # Ejecutar
            self.cash -= total_cost
            self.positions[asset] = self.positions.get(asset, 0) + quantity
            self.total_commission += commission
            self.total_slippage += quantity * price * self.slippage
            
            # Registrar trade
            self.trades.append(Trade(
                timestamp=timestamp,
                action='BUY',
                asset=asset,
                price=execution_price,
                size=quantity,
                value=cost,
                reasoning=reasoning,
                regime=regime
            ))
            
        elif action in ['SELL', 'CLOSE']:
            # Vender parte o toda la posición
            if asset not in self.positions or self.positions[asset] <= 0:
                return False  # No hay posición para vender
            
            if action == 'CLOSE':
                quantity = self.positions[asset]
            else:
                # Vender diferencia
                quantity = current_position - target_size
                quantity = max(0, quantity)  # No vender más de lo que tenemos
            
            # Aplicar slippage (vendemos más barato)
            execution_price = price * (1 - self.slippage)
            proceeds = quantity * execution_price
            commission = quantity * self.commission
            
            net_proceeds = proceeds - commission
            
            # Ejecutar
            self.cash += net_proceeds
            self.positions[asset] -= quantity
            
            if self.positions[asset] < 0.01:  # Evitar posiciones residuales
                self.positions[asset] = 0
            
            self.total_commission += commission
            self.total_slippage += quantity * price * self.slippage
            
            # Registrar trade
            self.trades.append(Trade(
                timestamp=timestamp,
                action='SELL' if action == 'SELL' else 'CLOSE',
                asset=asset,
                price=execution_price,
                size=quantity,
                value=proceeds,
                reasoning=reasoning,
                regime=regime
            ))
        
        return True
    
    def update_portfolio_value(self, 
                              timestamp: datetime,
                              prices: Dict[str, float],
                              regime: str = '') -> float:
        """
        Actualiza el valor del portfolio
        
        Args:
            timestamp: Momento actual
            prices: Diccionario asset -> precio actual
            regime: Régimen actual
            
        Returns:
            Valor total del portfolio
        """
        # Calcular valor de posiciones
        position_values = {}
        for asset, quantity in self.positions.items():
            if quantity > 0 and asset in prices:
                position_values[asset] = quantity * prices[asset]
            else:
                position_values[asset] = 0.0
        
        total_value = self.cash + sum(position_values.values())
        
        # Calcular returns
        if len(self.portfolio_history) > 0:
            prev_value = self.portfolio_history[-1].total_value
            returns = (total_value - prev_value) / prev_value if prev_value > 0 else 0.0
        else:
            returns = (total_value - self.initial_capital) / self.initial_capital
        
        # Guardar snapshot
        snapshot = PortfolioSnapshot(
            timestamp=timestamp,
            total_value=total_value,
            cash=self.cash,
            positions=self.positions.copy(),
            position_values=position_values,
            returns=returns,
            regime=regime
        )
        self.portfolio_history.append(snapshot)
        
        self.position_values = position_values
        return total_value
    
    def get_current_position(self, asset: str) -> float:
        """Obtiene la posición actual en un activo"""
        return self.positions.get(asset, 0.0)
    
    def get_portfolio_summary(self) -> Dict:
        """Resumen del portfolio"""
        if len(self.portfolio_history) == 0:
            return {}
        
        latest = self.portfolio_history[-1]
        
        return {
            'total_value': latest.total_value,
            'cash': latest.cash,
            'positions': latest.positions,
            'position_values': latest.position_values,
            'total_return': (latest.total_value - self.initial_capital) / self.initial_capital,
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage,
            'num_trades': len(self.trades)
        }
    
    def get_trades_df(self) -> pd.DataFrame:
        """Obtiene DataFrame de trades"""
        if len(self.trades) == 0:
            return pd.DataFrame()
        return pd.DataFrame([t.to_dict() for t in self.trades])
    
    def get_portfolio_history_df(self) -> pd.DataFrame:
        """Obtiene DataFrame de historia del portfolio"""
        if len(self.portfolio_history) == 0:
            return pd.DataFrame()
        return pd.DataFrame([s.to_dict() for s in self.portfolio_history])


class BacktestEngine:
    """
    Engine que coordina strategy, signals y backtester
    """
    
    def __init__(self, 
                 strategy,
                 signal_generator,
                 initial_capital: float = 1_000_000):
        """
        Args:
            strategy: Instancia de TradingStrategy o MultiAssetStrategy
            signal_generator: Instancia de IntegratedSignalGenerator
            initial_capital: Capital inicial
        """
        self.strategy = strategy
        self.signal_generator = signal_generator
        self.backtest = Backtest(initial_capital=initial_capital)
        
    def run(self,
            dates: List[datetime],
            prices: Dict[str, np.ndarray],  # asset -> array of prices
            returns: np.ndarray,
            vix_f1: np.ndarray,
            vix_f2: np.ndarray,
            market_prices: Optional[np.ndarray] = None,
            min_history: int = 504) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Ejecuta backtest completo
        
        Args:
            dates: Lista de fechas
            prices: Dict con arrays de precios por activo
            returns: Array de returns del activo principal
            vix_f1: Precios del VIX F1
            vix_f2: Precios del VIX F2
            market_prices: Precios del mercado/industria (opcional)
            min_history: Mínimo de días antes de empezar a tradear
            
        Returns:
            (portfolio_history_df, trades_df, summary)
        """
        from signals import calculate_vix_term_structure
        
        # Calcular VIX term structure
        vix_term_struct = calculate_vix_term_structure(vix_f1, vix_f2)
        
        # Obtener asset principal
        main_asset = list(prices.keys())[0]
        main_prices = prices[main_asset]
        
        print(f"Running backtest from {dates[min_history]} to {dates[-1]}")
        print(f"Initial capital: ${self.backtest.initial_capital:,.2f}")
        
        # Loop principal
        for i in range(min_history, len(dates)):
            current_date = dates[i]
            
            # Datos hasta este momento
            hist_returns = returns[:i+1]
            hist_vix_ts = vix_term_struct[:i+1]
            hist_prices = main_prices[:i+1]
            hist_market = market_prices[:i+1] if market_prices is not None else None
            
            # Generar señales
            signals = self.signal_generator.generate_signals(
                prices=hist_prices,
                returns=hist_returns,
                vix_term_structure=hist_vix_ts,
                market_prices=hist_market
            )
            
            # Decisión de trading
            decision = self.strategy.calculate_position(signals)
            
            # Precios actuales para valoración
            current_prices = {asset: price_array[i] for asset, price_array in prices.items()}
            
            # Ejecutar trades si necesario
            if decision['action'] in ['BUY', 'SELL', 'CLOSE', 'ADJUST']:
                target_size = decision['target_size']
                
                # Convertir target_size (ratio) a valor en $
                portfolio_value = self.backtest.update_portfolio_value(
                    current_date, current_prices, signals['regime']
                )
                target_value = portfolio_value * target_size
                
                self.backtest.execute_trade(
                    timestamp=current_date,
                    action=decision['action'],
                    asset=main_asset,
                    price=current_prices[main_asset],
                    target_size=target_value,
                    reasoning=decision['reasoning'],
                    regime=signals['regime']
                )
            
            # Actualizar valor del portfolio
            self.backtest.update_portfolio_value(
                current_date, current_prices, signals['regime']
            )
            
            # Progress cada 252 días (1 año)
            if i % 252 == 0:
                summary = self.backtest.get_portfolio_summary()
                print(f"{current_date.date()}: Portfolio=${summary['total_value']:,.2f} "
                      f"Return={summary['total_return']:.2%} Regime={signals['regime']}")
        
        # Resultados finales
        portfolio_df = self.backtest.get_portfolio_history_df()
        trades_df = self.backtest.get_trades_df()
        summary = self.backtest.get_portfolio_summary()
        
        print("\n" + "="*60)
        print("BACKTEST COMPLETE")
        print("="*60)
        print(f"Final Portfolio Value: ${summary['total_value']:,.2f}")
        print(f"Total Return: {summary['total_return']:.2%}")
        print(f"Number of Trades: {summary['num_trades']}")
        print(f"Total Commission: ${summary['total_commission']:,.2f}")
        print(f"Total Slippage: ${summary['total_slippage']:,.2f}")
        
        return portfolio_df, trades_df, summary