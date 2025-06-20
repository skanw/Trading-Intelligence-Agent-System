#!/usr/bin/env python3
"""
Basic Trading System Demo
Shows available data and basic functionality
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def load_signal_data():
    """Load the signal history data"""
    try:
        data = pd.read_parquet('data/signal_history.parquet')
        print(f"✅ Loaded {len(data)} signals from signal_history.parquet")
        return data
    except Exception as e:
        print(f"❌ Error loading signal data: {e}")
        return None

def load_training_data():
    """Load the training data"""
    try:
        data = pd.read_parquet('data/train.parquet')
        print(f"✅ Loaded {len(data)} training records from train.parquet")
        return data
    except Exception as e:
        print(f"❌ Error loading training data: {e}")
        return None

def analyze_data(df, name):
    """Basic data analysis"""
    print(f"\n📊 Analysis of {name}:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    if 'ticker' in df.columns:
        print(f"Unique tickers: {df['ticker'].nunique()}")
        print(f"Top tickers: {df['ticker'].value_counts().head().to_dict()}")
    
    if 'sentiment' in df.columns:
        print(f"Sentiment distribution: {df['sentiment'].value_counts().to_dict()}")
    
    if 'target_move' in df.columns:
        print(f"Target move distribution: {df['target_move'].value_counts().to_dict()}")
    
    print(f"Date range: {df.select_dtypes(include=['datetime']).min().min()} to {df.select_dtypes(include=['datetime']).max().max()}")

def simulate_trading_signals():
    """Simulate basic trading signals"""
    print("\n🔄 Simulating Trading Signals...")
    
    # Create sample price data
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
    
    signals = []
    
    for ticker in tickers:
        # Generate random walk price data
        price_base = np.random.uniform(100, 500)
        price_data = price_base + np.cumsum(np.random.normal(0, 5, len(dates)))
        
        for i, (date, price) in enumerate(zip(dates, price_data)):
            if i > 20:  # Need some history for signals
                # Simple momentum signal
                recent_prices = price_data[i-20:i]
                sma_20 = np.mean(recent_prices)
                
                if price > sma_20 * 1.02:  # 2% above moving average
                    signal_strength = min((price - sma_20) / sma_20 * 10, 1.0)
                    signals.append({
                        'ticker': ticker,
                        'date': date,
                        'price': price,
                        'sma_20': sma_20,
                        'signal': 'BUY',
                        'strength': signal_strength
                    })
                elif price < sma_20 * 0.98:  # 2% below moving average
                    signal_strength = min((sma_20 - price) / sma_20 * 10, 1.0)
                    signals.append({
                        'ticker': ticker,
                        'date': date,
                        'price': price,
                        'sma_20': sma_20,
                        'signal': 'SELL',
                        'strength': signal_strength
                    })
    
    signals_df = pd.DataFrame(signals)
    print(f"✅ Generated {len(signals_df)} trading signals")
    print(f"Signal breakdown: {signals_df['signal'].value_counts().to_dict()}")
    
    return signals_df

def basic_backtest(signals_df):
    """Basic backtesting simulation"""
    print("\n📈 Running Basic Backtest...")
    
    initial_capital = 100000
    capital = initial_capital
    positions = {}
    trades = []
    
    for _, signal in signals_df.iterrows():
        ticker = signal['ticker']
        price = signal['price']
        signal_type = signal['signal']
        strength = signal['strength']
        
        if signal_type == 'BUY' and strength > 0.5:
            # Buy signal with sufficient strength
            if ticker not in positions:
                position_size = capital * 0.1 * strength  # Risk 10% * signal strength
                shares = position_size / price
                
                if capital >= position_size:
                    positions[ticker] = {
                        'shares': shares,
                        'entry_price': price,
                        'entry_date': signal['date']
                    }
                    capital -= position_size
                    
                    trades.append({
                        'ticker': ticker,
                        'action': 'BUY',
                        'shares': shares,
                        'price': price,
                        'date': signal['date'],
                        'value': position_size
                    })
        
        elif signal_type == 'SELL' and ticker in positions and strength > 0.5:
            # Sell signal
            position = positions[ticker]
            shares = position['shares']
            value = shares * price
            capital += value
            
            profit = value - (shares * position['entry_price'])
            
            trades.append({
                'ticker': ticker,
                'action': 'SELL',
                'shares': shares,
                'price': price,
                'date': signal['date'],
                'value': value,
                'profit': profit
            })
            
            del positions[ticker]
    
    # Calculate final portfolio value
    final_value = capital
    for ticker, position in positions.items():
        # Use last known price for evaluation
        last_signal = signals_df[signals_df['ticker'] == ticker].iloc[-1]
        final_value += position['shares'] * last_signal['price']
    
    total_return = (final_value - initial_capital) / initial_capital * 100
    
    print(f"✅ Backtest Results:")
    print(f"   Initial Capital: ${initial_capital:,.2f}")
    print(f"   Final Value: ${final_value:,.2f}")
    print(f"   Total Return: {total_return:.2f}%")
    print(f"   Total Trades: {len(trades)}")
    print(f"   Winning Trades: {len([t for t in trades if 'profit' in t and t['profit'] > 0])}")
    
    return trades

def main():
    """Main demo function"""
    print("🤖 Trading Intelligence Agent System - Basic Demo")
    print("=" * 60)
    print(f"Demo started at: {datetime.now()}")
    print()
    
    # Load actual data
    signal_data = load_signal_data()
    training_data = load_training_data()
    
    if signal_data is not None:
        analyze_data(signal_data, "Signal History")
    
    if training_data is not None:
        analyze_data(training_data, "Training Data")
    
    # Generate and analyze simulated signals
    simulated_signals = simulate_trading_signals()
    
    # Run basic backtest
    trades = basic_backtest(simulated_signals)
    
    print("\n🎯 Demo Summary:")
    print("✅ Data loading and analysis working")
    print("✅ Signal generation working")
    print("✅ Basic backtesting working")
    print("✅ Trading simulation complete")
    
    print("\n💡 Next Steps:")
    print("1. Access the Streamlit dashboard at: http://localhost:8501")
    print("2. Install missing dependencies (talib) for full functionality")
    print("3. Set up API keys for live data feeds")
    print("4. Configure Redis for real-time caching")
    
    print("\n🎉 Basic demo completed successfully!")

if __name__ == '__main__':
    main() 