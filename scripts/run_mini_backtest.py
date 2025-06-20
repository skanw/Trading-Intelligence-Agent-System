#!/usr/bin/env python3
"""
Mini Backtest Script for CI/CD Validation

Runs a quick backtest to validate system functionality during CI/CD pipeline.
"""

import sys
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from backtesting import DataLoader, BacktestEngine
from risk_management import RiskManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_mini_backtest():
    """Run a mini backtest for CI/CD validation"""
    
    logger.info("🚀 Starting mini backtest for CI/CD validation")
    
    try:
        # Initialize components
        data_loader = DataLoader()
        initial_capital = 50000.0
        risk_manager = RiskManager(initial_capital=initial_capital)
        backtest_engine = BacktestEngine(initial_capital=initial_capital)
        
        # Define test parameters
        symbols = ['AAPL', 'MSFT']
        start_date = datetime.now() - timedelta(days=7)  # 1 week backtest
        end_date = datetime.now() - timedelta(days=1)
        signal_threshold = 0.7
        
        logger.info(f"Running backtest for {symbols} from {start_date.date()} to {end_date.date()}")
        
        # Run backtest
        result = backtest_engine.run_backtest(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            data_loader=data_loader,
            risk_manager=risk_manager,
            signal_threshold=signal_threshold
        )
        
        # Validate results
        assert result is not None, "Backtest result should not be None"
        assert result.initial_capital == initial_capital, "Initial capital mismatch"
        assert result.final_capital > 0, "Final capital should be positive"
        assert result.total_trades >= 0, "Trade count should be non-negative"
        assert hasattr(result, 'equity_curve'), "Result should have equity curve"
        
        # Log results
        logger.info(f"✅ Mini backtest completed successfully!")
        logger.info(f"📊 Results Summary:")
        logger.info(f"   - Initial Capital: ${result.initial_capital:,.2f}")
        logger.info(f"   - Final Capital: ${result.final_capital:,.2f}")
        logger.info(f"   - Total Return: {result.total_return_pct:.2%}")
        logger.info(f"   - Total Trades: {result.total_trades}")
        logger.info(f"   - Sharpe Ratio: {result.sharpe_ratio:.2f}")
        logger.info(f"   - Max Drawdown: {result.max_drawdown:.2%}")
        logger.info(f"   - Win Rate: {result.win_rate:.2%}")
        
        # Basic validation checks
        if result.total_trades > 0:
            logger.info("✅ Trading system generated signals and executed trades")
        else:
            logger.warning("⚠️ No trades executed during backtest period")
        
        if abs(result.total_return_pct) < 0.5:  # Less than 50% change
            logger.info("✅ Returns within reasonable range")
        else:
            logger.warning(f"⚠️ Large return magnitude: {result.total_return_pct:.2%}")
        
        if result.sharpe_ratio > -2.0:  # Not extremely negative
            logger.info("✅ Sharpe ratio within acceptable range")
        else:
            logger.warning(f"⚠️ Poor Sharpe ratio: {result.sharpe_ratio:.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Mini backtest failed: {str(e)}")
        return False

def test_system_components():
    """Test individual system components"""
    
    logger.info("🔧 Testing system components")
    
    try:
        # Test DataLoader
        logger.info("Testing DataLoader...")
        data_loader = DataLoader()
        
        # Test sample data generation
        symbols = ['AAPL']
        start_date = datetime.now() - timedelta(days=2)
        end_date = datetime.now() - timedelta(days=1)
        
        price_data = data_loader.load_price_data(symbols, start_date, end_date)
        signal_data = data_loader.load_signal_data(symbols, start_date, end_date)
        
        assert not price_data.empty, "Price data should not be empty"
        assert not signal_data.empty, "Signal data should not be empty"
        logger.info("✅ DataLoader working correctly")
        
        # Test RiskManager
        logger.info("Testing RiskManager...")
        risk_manager = RiskManager(initial_capital=10000)
        
        # Test basic functionality
        equity = risk_manager.get_total_equity()
        assert equity == 10000, f"Initial equity should be 10000, got {equity}"
        
        risk_metrics = risk_manager.get_risk_metrics()
        assert risk_metrics.portfolio_value == 10000, "Portfolio value mismatch"
        
        logger.info("✅ RiskManager working correctly")
        
        # Test BacktestEngine
        logger.info("Testing BacktestEngine...")
        engine = BacktestEngine(initial_capital=10000)
        assert engine.initial_capital == 10000, "Engine capital mismatch"
        
        logger.info("✅ BacktestEngine working correctly")
        
        logger.info("✅ All system components tested successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Component testing failed: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("TIAS Mini Backtest - CI/CD Validation")
    logger.info("=" * 60)
    
    # Test components first
    components_ok = test_system_components()
    
    if not components_ok:
        logger.error("❌ Component tests failed - aborting backtest")
        sys.exit(1)
    
    # Run mini backtest
    backtest_ok = run_mini_backtest()
    
    if backtest_ok:
        logger.info("🎉 Mini backtest validation completed successfully!")
        sys.exit(0)
    else:
        logger.error("💥 Mini backtest validation failed!")
        sys.exit(1) 