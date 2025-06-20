"""
Backtesting Engine for TIAS

Production-grade backtesting engine with realistic slippage, commission modeling,
and comprehensive performance metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from .data_loader import DataLoader
from ..risk_management.risk_manager import RiskManager, Position

logger = logging.getLogger(__name__)

@dataclass
class Trade:
    """Represents a completed trade"""
    symbol: str
    side: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    commission: float
    slippage: float
    pnl: float
    pnl_pct: float
    duration: timedelta
    exit_reason: str

@dataclass
class BacktestResult:
    """Container for backtest results and performance metrics"""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_pct: float
    
    # Performance metrics
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # Trade details
    trades: List[Trade]
    equity_curve: pd.DataFrame

class BacktestEngine:
    """Production-grade backtesting engine."""
    
    def __init__(self,
                 initial_capital: float = 100000.0,
                 commission_rate: float = 0.001,  # 0.1% commission
                 slippage_rate: float = 0.0005):   # 0.05% slippage
        
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        
        # Backtest state
        self.current_time: Optional[datetime] = None
        self.price_data: Optional[pd.DataFrame] = None
        self.signal_data: Optional[pd.DataFrame] = None
        self.risk_manager: Optional[RiskManager] = None
        
        # Results tracking
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        
        logger.info(f"BacktestEngine initialized with {initial_capital} capital")
    
    def run_backtest(self,
                    symbols: List[str],
                    start_date: datetime,
                    end_date: datetime,
                    data_loader: DataLoader,
                    risk_manager: RiskManager = None,
                    signal_threshold: float = 0.7) -> BacktestResult:
        """Run complete backtest simulation."""
        
        logger.info(f"Starting backtest from {start_date} to {end_date}")
        
        # Initialize risk manager if not provided
        if risk_manager is None:
            risk_manager = RiskManager(initial_capital=self.initial_capital)
        self.risk_manager = risk_manager
        
        # Load data
        self.price_data = data_loader.load_price_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date
        )
        
        self.signal_data = data_loader.load_signal_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date
        )
        
        if self.price_data.empty or self.signal_data.empty:
            raise ValueError("No data available for backtest period")
        
        # Run simulation
        self._run_simulation(signal_threshold)
        
        # Calculate performance metrics
        result = self._calculate_performance_metrics(start_date, end_date)
        
        logger.info(f"Backtest completed: {len(self.trades)} trades, "
                   f"{result.total_return_pct:.2%} return")
        
        return result
    
    def _run_simulation(self, signal_threshold: float):
        """Run the main simulation loop"""
        # Get unique timestamps and sort
        all_timestamps = set()
        all_timestamps.update(self.price_data.index)
        all_timestamps.update(self.signal_data.index)
        sorted_timestamps = sorted(all_timestamps)
        
        logger.info(f"Simulating {len(sorted_timestamps)} time steps")
        
        for timestamp in sorted_timestamps:
            self.current_time = timestamp
            
            # Update current prices for all symbols
            current_prices = self._get_current_prices(timestamp)
            if current_prices:
                self.risk_manager.update_positions(current_prices, timestamp)
            
            # Process new signals
            signals = self._get_signals_at_time(timestamp)
            for signal in signals:
                self._process_signal(signal, current_prices, signal_threshold)
            
            # Record equity curve
            equity = self.risk_manager.get_total_equity()
            self.equity_curve.append((timestamp, equity))
    
    def _get_current_prices(self, timestamp: datetime) -> Dict[str, float]:
        """Get current prices for all symbols at given timestamp"""
        current_prices = {}
        
        if 'symbol' in self.price_data.columns:
            # Multi-symbol data
            price_slice = self.price_data[self.price_data.index == timestamp]
            for _, row in price_slice.iterrows():
                current_prices[row['symbol']] = row['close']
        else:
            # Single symbol data
            if timestamp in self.price_data.index:
                price = self.price_data.loc[timestamp, 'close']
                current_prices['DEFAULT'] = price
        
        return current_prices
    
    def _get_signals_at_time(self, timestamp: datetime) -> List[Dict]:
        """Get all signals at given timestamp"""
        signals = []
        
        if timestamp in self.signal_data.index:
            signal_slice = self.signal_data[self.signal_data.index == timestamp]
            
            for _, row in signal_slice.iterrows():
                signal_dict = row.to_dict()
                signal_dict['timestamp'] = timestamp
                signals.append(signal_dict)
        
        return signals
    
    def _process_signal(self, 
                       signal: Dict, 
                       current_prices: Dict[str, float],
                       signal_threshold: float):
        """Process a trading signal"""
        symbol = signal.get('symbol', 'DEFAULT')
        signal_type = signal.get('signal', 'hold')
        confidence = signal.get('confidence', 0.0)
        
        # Skip if signal doesn't meet threshold
        if confidence < signal_threshold:
            return
        
        # Skip if no price data available
        if symbol not in current_prices:
            return
        
        current_price = current_prices[symbol]
        
        # Determine trading action
        if signal_type == 'buy' and symbol not in self.risk_manager.positions:
            self._execute_buy(symbol, current_price, confidence)
        elif signal_type == 'sell' and symbol in self.risk_manager.positions:
            self._execute_sell(symbol, current_price, "signal")
    
    def _execute_buy(self, symbol: str, price: float, confidence: float):
        """Execute a buy order with slippage and commission"""
        # Calculate position size
        position_size = self.risk_manager.calculate_position_size(
            symbol=symbol,
            signal_strength=confidence,
            price=price
        )
        
        if position_size <= 0:
            return
        
        # Apply slippage
        execution_price = price * (1 + self.slippage_rate)
        
        # Open position
        success = self.risk_manager.open_position(
            symbol=symbol,
            side='long',
            quantity=position_size,
            price=execution_price,
            timestamp=self.current_time
        )
        
        if success:
            logger.debug(f"Bought {position_size} shares of {symbol} at {execution_price}")
    
    def _execute_sell(self, symbol: str, price: float, reason: str):
        """Execute a sell order with slippage and commission"""
        if symbol not in self.risk_manager.positions:
            return
        
        position = self.risk_manager.positions[symbol]
        
        # Apply slippage
        execution_price = price * (1 - self.slippage_rate)
        
        # Close position and record trade
        success = self.risk_manager.close_position(
            symbol=symbol,
            price=execution_price,
            timestamp=self.current_time,
            reason=reason
        )
        
        if success:
            self._record_trade(position, execution_price, reason)
            logger.debug(f"Sold {position.quantity} shares of {symbol} at {execution_price}")
    
    def _record_trade(self, position: Position, exit_price: float, exit_reason: str):
        """Record a completed trade"""
        entry_time = position.timestamp
        exit_time = self.current_time
        
        # Calculate commission
        total_commission = position.quantity * (position.entry_price + exit_price) * self.commission_rate
        
        # Calculate P&L
        if position.side == 'long':
            gross_pnl = position.quantity * (exit_price - position.entry_price)
        else:
            gross_pnl = position.quantity * (position.entry_price - exit_price)
        
        net_pnl = gross_pnl - total_commission
        
        # Calculate percentage return
        cost_basis = position.quantity * position.entry_price
        pnl_pct = net_pnl / cost_basis if cost_basis > 0 else 0
        
        trade = Trade(
            symbol=position.symbol,
            side=position.side,
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
            commission=total_commission,
            slippage=position.quantity * (position.entry_price + exit_price) * self.slippage_rate,
            pnl=net_pnl,
            pnl_pct=pnl_pct,
            duration=exit_time - entry_time,
            exit_reason=exit_reason
        )
        
        self.trades.append(trade)
    
    def _calculate_performance_metrics(self, 
                                     start_date: datetime, 
                                     end_date: datetime) -> BacktestResult:
        """Calculate comprehensive performance metrics"""
        
        # Basic metrics
        final_capital = self.risk_manager.get_total_equity()
        total_return = final_capital - self.initial_capital
        total_return_pct = total_return / self.initial_capital
        
        # Create equity curve DataFrame
        equity_df = pd.DataFrame(self.equity_curve, columns=['timestamp', 'equity'])
        equity_df.set_index('timestamp', inplace=True)
        
        # Calculate returns
        equity_df['returns'] = equity_df['equity'].pct_change()
        daily_returns = equity_df['returns'].dropna()
        
        # Performance metrics
        sharpe_ratio = self._calculate_sharpe_ratio(daily_returns)
        max_drawdown = self._calculate_max_drawdown(equity_df['equity'])
        volatility = daily_returns.std() * np.sqrt(252)  # Annualized
        
        # Trade statistics
        total_trades = len(self.trades)
        if total_trades > 0:
            winning_trades = len([t for t in self.trades if t.pnl > 0])
            win_rate = winning_trades / total_trades
            
            wins = [t.pnl for t in self.trades if t.pnl > 0]
            losses = [t.pnl for t in self.trades if t.pnl < 0]
            
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
            
            gross_profit = sum(wins)
            gross_loss = abs(sum(losses))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        else:
            winning_trades = 0
            win_rate = avg_win = avg_loss = profit_factor = 0
        
        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            total_return_pct=total_return_pct,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            volatility=volatility,
            total_trades=total_trades,
            winning_trades=winning_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            trades=self.trades,
            equity_curve=equity_df
        )
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0.0
        
        mean_return = returns.mean()
        std_return = returns.std()
        
        if std_return == 0:
            return 0.0
        
        return (mean_return / std_return) * np.sqrt(252)  # Annualized
    
    def _calculate_max_drawdown(self, equity: pd.Series) -> float:
        """Calculate maximum drawdown"""
        if len(equity) < 2:
            return 0.0
        
        # Calculate running maximum
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak
        
        return drawdown.min() 