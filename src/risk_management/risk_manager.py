"""
Risk Manager - Core risk management functionality

Handles position sizing, risk controls, and portfolio-level risk management.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk level classification"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class RiskMetrics:
    """Container for risk metrics"""
    portfolio_value: float
    total_exposure: float
    leverage: float
    var_1d: float  # 1-day Value at Risk
    max_drawdown: float
    sharpe_ratio: float
    volatility: float
    risk_level: RiskLevel

@dataclass
class Position:
    """Represents a trading position"""
    symbol: str
    side: str  # 'long' or 'short'
    quantity: float
    entry_price: float
    current_price: float
    timestamp: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    @property
    def market_value(self) -> float:
        """Current market value of position"""
        return self.quantity * self.current_price
    
    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss"""
        if self.side == 'long':
            return self.quantity * (self.current_price - self.entry_price)
        else:
            return self.quantity * (self.entry_price - self.current_price)

class PositionSizer(ABC):
    """Abstract base class for position sizing strategies"""
    
    @abstractmethod
    def calculate_position_size(self, 
                              signal_strength: float,
                              account_balance: float,
                              price: float,
                              volatility: float,
                              **kwargs) -> float:
        """Calculate position size in shares/units"""
        pass

class RiskManager:
    """Main risk management system."""
    
    def __init__(self,
                 initial_capital: float = 100000.0,
                 max_position_size: float = 0.05,  # 5% max position
                 stop_loss_pct: float = 0.02,      # 2% stop loss
                 take_profit_pct: float = 0.06,    # 6% take profit
                 max_drawdown_limit: float = 0.15):  # 15% max drawdown
        
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_drawdown_limit = max_drawdown_limit
        
        # Portfolio state
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        self.cash = initial_capital
        self.equity_curve: List[Tuple[datetime, float]] = []
        
        # Risk tracking
        self.peak_equity = initial_capital
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        
        logger.info(f"RiskManager initialized with {initial_capital} capital")
    
    def calculate_position_size(self,
                              symbol: str,
                              signal_strength: float,
                              price: float,
                              volatility: float = 0.02) -> float:
        """Calculate optimal position size for a new trade."""
        
        # Check if we can open new positions
        if not self._can_open_position(symbol):
            return 0.0
        
        # Simple fixed fractional sizing
        portfolio_value = self.get_total_equity()
        max_position_value = portfolio_value * self.max_position_size * signal_strength
        position_size = max_position_value / price
        
        # Apply risk limits
        position_size = self._apply_risk_limits(symbol, position_size, price)
        
        logger.debug(f"Calculated position size for {symbol}: {position_size} shares")
        return position_size
    
    def open_position(self,
                     symbol: str,
                     side: str,
                     quantity: float,
                     price: float,
                     timestamp: datetime = None) -> bool:
        """Open a new position."""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Calculate stop loss and take profit
        stop_loss = self._calculate_stop_loss(price, side)
        take_profit = self._calculate_take_profit(price, side)
        
        # Create position
        position = Position(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=price,
            current_price=price,
            timestamp=timestamp,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        # Update portfolio
        self.positions[symbol] = position
        self.cash -= quantity * price
        
        # Update equity curve
        self._update_equity_curve(timestamp)
        
        logger.info(f"Opened {side} position for {symbol}: {quantity} shares at {price}")
        return True
    
    def close_position(self,
                      symbol: str,
                      price: float,
                      timestamp: datetime = None,
                      reason: str = "manual") -> bool:
        """Close an existing position"""
        if symbol not in self.positions:
            return False
        
        if timestamp is None:
            timestamp = datetime.now()
        
        position = self.positions[symbol]
        
        # Update cash
        self.cash += position.quantity * price
        
        # Move to closed positions
        position.current_price = price
        self.closed_positions.append(position)
        del self.positions[symbol]
        
        # Update equity curve
        self._update_equity_curve(timestamp)
        
        logger.info(f"Closed position for {symbol}: reason = {reason}")
        return True
    
    def update_positions(self, prices: Dict[str, float], timestamp: datetime = None):
        """Update current prices for all positions"""
        if timestamp is None:
            timestamp = datetime.now()
        
        for symbol, position in self.positions.items():
            if symbol in prices:
                position.current_price = prices[symbol]
                
                # Check stop loss and take profit
                self._check_risk_controls(symbol, position, timestamp)
        
        # Update equity curve
        self._update_equity_curve(timestamp)
    
    def get_total_equity(self) -> float:
        """Get total portfolio equity"""
        position_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + position_value
    
    def get_risk_metrics(self) -> RiskMetrics:
        """Calculate current risk metrics"""
        portfolio_value = self.get_total_equity()
        total_exposure = sum(pos.market_value for pos in self.positions.values())
        
        leverage = total_exposure / portfolio_value if portfolio_value > 0 else 0
        
        # Calculate VaR (simplified)
        returns = self._calculate_returns()
        var_1d = np.percentile(returns, 5) if len(returns) > 0 else 0
        
        # Calculate Sharpe ratio
        sharpe_ratio = self._calculate_sharpe_ratio()
        
        # Calculate volatility
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
        
        # Determine risk level
        risk_level = self._determine_risk_level(leverage, abs(var_1d), self.current_drawdown)
        
        return RiskMetrics(
            portfolio_value=portfolio_value,
            total_exposure=total_exposure,
            leverage=leverage,
            var_1d=var_1d,
            max_drawdown=self.max_drawdown,
            sharpe_ratio=sharpe_ratio,
            volatility=volatility,
            risk_level=risk_level
        )
    
    def _can_open_position(self, symbol: str) -> bool:
        """Check basic constraints for opening positions"""
        if self.current_drawdown >= self.max_drawdown_limit:
            return False
        if symbol in self.positions:
            return False
        return True
    
    def _apply_risk_limits(self, symbol: str, position_size: float, price: float) -> float:
        """Apply risk limits to position size"""
        # Limit by available cash
        max_cash_shares = self.cash / price
        position_size = min(position_size, max_cash_shares)
        return max(0, position_size)
    
    def _calculate_stop_loss(self, price: float, side: str) -> float:
        """Calculate stop loss price"""
        if side == 'long':
            return price * (1 - self.stop_loss_pct)
        else:
            return price * (1 + self.stop_loss_pct)
    
    def _calculate_take_profit(self, price: float, side: str) -> float:
        """Calculate take profit price"""
        if side == 'long':
            return price * (1 + self.take_profit_pct)
        else:
            return price * (1 - self.take_profit_pct)
    
    def _check_risk_controls(self, symbol: str, position: Position, timestamp: datetime):
        """Check stop loss and take profit conditions"""
        if position.side == 'long':
            if position.stop_loss and position.current_price <= position.stop_loss:
                self.close_position(symbol, position.current_price, timestamp, "stop_loss")
            elif position.take_profit and position.current_price >= position.take_profit:
                self.close_position(symbol, position.current_price, timestamp, "take_profit")
        else:
            if position.stop_loss and position.current_price >= position.stop_loss:
                self.close_position(symbol, position.current_price, timestamp, "stop_loss")
            elif position.take_profit and position.current_price <= position.take_profit:
                self.close_position(symbol, position.current_price, timestamp, "take_profit")
    
    def _update_equity_curve(self, timestamp: datetime):
        """Update equity curve with current portfolio value"""
        current_equity = self.get_total_equity()
        self.equity_curve.append((timestamp, current_equity))
        
        # Update drawdown tracking
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
    
    def _calculate_returns(self) -> np.ndarray:
        """Calculate daily returns from equity curve"""
        if len(self.equity_curve) < 2:
            return np.array([])
        
        values = [equity for _, equity in self.equity_curve]
        returns = np.diff(values) / values[:-1]
        return returns
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio"""
        returns = self._calculate_returns()
        if len(returns) < 2:
            return 0.0
        
        return np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0.0
    
    def _determine_risk_level(self, leverage: float, var: float, drawdown: float) -> RiskLevel:
        """Determine overall risk level"""
        if leverage > 2.0 or var > 0.05 or drawdown > 0.15:
            return RiskLevel.CRITICAL
        elif leverage > 1.5 or var > 0.03 or drawdown > 0.10:
            return RiskLevel.HIGH
        elif leverage > 1.0 or var > 0.02 or drawdown > 0.05:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW 