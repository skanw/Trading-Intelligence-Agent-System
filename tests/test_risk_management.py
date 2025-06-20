"""
Unit tests for risk management module
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from risk_management.risk_manager import RiskManager, RiskLevel


class TestRiskManager:
    """Test cases for RiskManager class"""
    
    def test_init(self):
        """Test RiskManager initialization"""
        initial_capital = 100000.0
        rm = RiskManager(initial_capital=initial_capital)
        
        assert rm.initial_capital == initial_capital
        assert rm.current_capital == initial_capital
        assert rm.cash == initial_capital
        assert len(rm.positions) == 0
        assert rm.get_total_equity() == initial_capital
    
    def test_position_sizing(self):
        """Test position sizing calculation"""
        rm = RiskManager(initial_capital=100000.0)
        
        position_size = rm.calculate_position_size(
            symbol="AAPL",
            signal_strength=0.8,
            price=150.0
        )
        
        assert position_size >= 0
        assert position_size <= rm.cash / 150.0  # Can't exceed available cash
    
    def test_open_position(self):
        """Test opening a position"""
        rm = RiskManager(initial_capital=100000.0)
        
        success = rm.open_position(
            symbol="AAPL",
            side="long",
            quantity=100,
            price=150.0,
            timestamp=datetime.now()
        )
        
        assert success == True
        assert "AAPL" in rm.positions
        assert rm.positions["AAPL"].quantity == 100
        assert rm.positions["AAPL"].entry_price == 150.0
        assert rm.cash == 100000.0 - (100 * 150.0)
    
    def test_close_position(self):
        """Test closing a position"""
        rm = RiskManager(initial_capital=100000.0)
        
        # Open position first
        rm.open_position("AAPL", "long", 100, 150.0, datetime.now())
        
        # Close position
        success = rm.close_position(
            symbol="AAPL",
            price=160.0,
            timestamp=datetime.now(),
            reason="test"
        )
        
        assert success == True
        assert "AAPL" not in rm.positions
        assert len(rm.closed_positions) == 1
    
    def test_update_positions(self):
        """Test updating position prices"""
        rm = RiskManager(initial_capital=100000.0)
        
        # Open position
        rm.open_position("AAPL", "long", 100, 150.0, datetime.now())
        
        # Update prices
        new_prices = {"AAPL": 160.0}
        rm.update_positions(new_prices, datetime.now())
        
        assert rm.positions["AAPL"].current_price == 160.0
        assert rm.positions["AAPL"].unrealized_pnl == 100 * (160.0 - 150.0)
    
    def test_stop_loss(self):
        """Test stop loss functionality"""
        rm = RiskManager(initial_capital=100000.0, stop_loss_pct=0.05)
        
        # Open position
        rm.open_position("AAPL", "long", 100, 150.0, datetime.now())
        
        # Price drops triggering stop loss
        stop_price = 150.0 * (1 - 0.05)  # 5% stop loss
        new_prices = {"AAPL": stop_price - 1.0}  # Below stop loss
        rm.update_positions(new_prices, datetime.now())
        
        # Position should be closed
        assert "AAPL" not in rm.positions
        assert len(rm.closed_positions) == 1
    
    def test_take_profit(self):
        """Test take profit functionality"""
        rm = RiskManager(initial_capital=100000.0, take_profit_pct=0.10)
        
        # Open position
        rm.open_position("AAPL", "long", 100, 150.0, datetime.now())
        
        # Price rises triggering take profit
        take_profit_price = 150.0 * (1 + 0.10)  # 10% take profit
        new_prices = {"AAPL": take_profit_price + 1.0}  # Above take profit
        rm.update_positions(new_prices, datetime.now())
        
        # Position should be closed
        assert "AAPL" not in rm.positions
        assert len(rm.closed_positions) == 1
    
    def test_risk_metrics(self):
        """Test risk metrics calculation"""
        rm = RiskManager(initial_capital=100000.0)
        
        # Add some equity curve data
        rm.equity_curve = [
            (datetime.now() - timedelta(days=5), 100000),
            (datetime.now() - timedelta(days=4), 102000),
            (datetime.now() - timedelta(days=3), 98000),
            (datetime.now() - timedelta(days=2), 101000),
            (datetime.now() - timedelta(days=1), 99000),
            (datetime.now(), 103000)
        ]
        
        risk_metrics = rm.get_risk_metrics()
        
        assert hasattr(risk_metrics, 'portfolio_value')
        assert hasattr(risk_metrics, 'leverage')
        assert hasattr(risk_metrics, 'sharpe_ratio')
        assert hasattr(risk_metrics, 'max_drawdown')
        assert isinstance(risk_metrics.risk_level, RiskLevel)
    
    def test_drawdown_calculation(self):
        """Test drawdown calculation"""
        rm = RiskManager(initial_capital=100000.0)
        
        # Simulate equity curve with drawdown
        rm.equity_curve = [
            (datetime.now() - timedelta(days=3), 100000),  # Peak
            (datetime.now() - timedelta(days=2), 95000),   # Down 5%
            (datetime.now() - timedelta(days=1), 90000),   # Down 10%
            (datetime.now(), 95000)                        # Recovering
        ]
        
        # Update peak and drawdown manually for test
        rm.peak_equity = 100000
        rm.current_drawdown = 0.05  # 5% current drawdown
        rm.max_drawdown = 0.10      # 10% max drawdown
        
        risk_metrics = rm.get_risk_metrics()
        assert risk_metrics.max_drawdown == 0.10
    
    def test_multiple_positions(self):
        """Test handling multiple positions"""
        rm = RiskManager(initial_capital=100000.0)
        
        # Open multiple positions
        rm.open_position("AAPL", "long", 100, 150.0, datetime.now())
        rm.open_position("MSFT", "long", 50, 200.0, datetime.now())
        
        assert len(rm.positions) == 2
        assert "AAPL" in rm.positions
        assert "MSFT" in rm.positions
        
        # Total position value
        total_value = (100 * 150.0) + (50 * 200.0)
        remaining_cash = 100000.0 - total_value
        assert rm.cash == remaining_cash
    
    def test_insufficient_cash(self):
        """Test handling insufficient cash"""
        rm = RiskManager(initial_capital=1000.0)  # Small capital
        
        # Try to open large position
        position_size = rm.calculate_position_size(
            symbol="AAPL",
            signal_strength=1.0,
            price=150.0
        )
        
        # Should be limited by available cash
        max_affordable = rm.cash / 150.0
        assert position_size <= max_affordable


if __name__ == "__main__":
    pytest.main([__file__]) 