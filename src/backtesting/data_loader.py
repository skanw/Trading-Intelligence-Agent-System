"""
Data Loader for Backtesting Framework

Handles loading and preprocessing of price data and signals from Parquet stores.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

@dataclass
class PriceData:
    """Container for OHLCV price data"""
    symbol: str
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: int
    
    @property
    def typical_price(self) -> float:
        """Calculate typical price (HLC/3)"""
        return (self.high + self.low + self.close) / 3
    
    @property
    def weighted_price(self) -> float:
        """Calculate volume-weighted price"""
        return (self.high + self.low + self.close * 2) / 4

@dataclass
class SignalData:
    """Container for trading signals"""
    symbol: str
    timestamp: pd.Timestamp
    signal: str  # 'buy', 'sell', 'hold'
    confidence: float
    score: float
    sentiment: str
    headline: str
    source: str

class DataLoader:
    """
    Production data loader for backtesting framework.
    
    Loads price data and signals from Parquet files with proper alignment
    and preprocessing for backtesting engine.
    """
    
    def __init__(self, 
                 data_dir: Union[str, Path] = "data",
                 price_file: str = "price_data.parquet",
                 signal_file: str = "signal_history.parquet"):
        """
        Initialize DataLoader.
        
        Args:
            data_dir: Directory containing data files
            price_file: Name of price data parquet file
            signal_file: Name of signal data parquet file
        """
        self.data_dir = Path(data_dir)
        self.price_file = self.data_dir / price_file
        self.signal_file = self.data_dir / signal_file
        
        # Cache for loaded data
        self._price_cache: Optional[pd.DataFrame] = None
        self._signal_cache: Optional[pd.DataFrame] = None
        
    def load_price_data(self, 
                       symbols: Optional[List[str]] = None,
                       start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Load OHLCV price data from Parquet file.
        
        Args:
            symbols: List of symbols to load (None for all)
            start_date: Start date for data
            end_date: End date for data
            timeframe: Timeframe for data aggregation
            
        Returns:
            DataFrame with price data
        """
        try:
            # Load from cache or file
            if self._price_cache is None:
                if not self.price_file.exists():
                    logger.warning(f"Price file {self.price_file} not found, generating sample data")
                    return self._generate_sample_price_data(symbols, start_date, end_date)
                
                logger.info(f"Loading price data from {self.price_file}")
                self._price_cache = pd.read_parquet(self.price_file)
                
                # Ensure timestamp column is datetime
                if 'timestamp' in self._price_cache.columns:
                    self._price_cache['timestamp'] = pd.to_datetime(self._price_cache['timestamp'])
                    self._price_cache.set_index('timestamp', inplace=True)
            
            df = self._price_cache.copy()
            
            # Filter by symbols
            if symbols:
                if 'symbol' in df.columns:
                    df = df[df['symbol'].isin(symbols)]
                else:
                    logger.warning("No symbol column found in price data")
            
            # Filter by date range
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
            
            # Resample if needed
            if timeframe != '1min':
                df = self._resample_price_data(df, timeframe)
            
            logger.info(f"Loaded {len(df)} price records")
            return df
            
        except Exception as e:
            logger.error(f"Error loading price data: {e}")
            return self._generate_sample_price_data(symbols, start_date, end_date)
    
    def load_signal_data(self,
                        symbols: Optional[List[str]] = None,
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Load trading signals from Parquet file.
        
        Args:
            symbols: List of symbols to filter
            start_date: Start date for signals
            end_date: End date for signals
            
        Returns:
            DataFrame with signal data
        """
        try:
            # Load from cache or file
            if self._signal_cache is None:
                if not self.signal_file.exists():
                    logger.warning(f"Signal file {self.signal_file} not found, generating sample data")
                    return self._generate_sample_signal_data(symbols, start_date, end_date)
                
                logger.info(f"Loading signal data from {self.signal_file}")
                self._signal_cache = pd.read_parquet(self.signal_file)
                
                # Ensure timestamp column is datetime
                if 'published_at' in self._signal_cache.columns:
                    self._signal_cache['timestamp'] = pd.to_datetime(self._signal_cache['published_at'])
                elif 'timestamp' in self._signal_cache.columns:
                    self._signal_cache['timestamp'] = pd.to_datetime(self._signal_cache['timestamp'])
                
                self._signal_cache.set_index('timestamp', inplace=True)
            
            df = self._signal_cache.copy()
            
            # Filter by symbols
            if symbols:
                if 'symbol' in df.columns:
                    df = df[df['symbol'].isin(symbols)]
            
            # Filter by date range
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
            
            logger.info(f"Loaded {len(df)} signal records")
            return df
            
        except Exception as e:
            logger.error(f"Error loading signal data: {e}")
            return self._generate_sample_signal_data(symbols, start_date, end_date)
    
    def align_data(self, 
                   price_data: pd.DataFrame, 
                   signal_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Align price and signal data by timestamp and symbol.
        
        Args:
            price_data: OHLCV price data
            signal_data: Trading signals
            
        Returns:
            Tuple of aligned (price_data, signal_data)
        """
        # Find common time range
        price_start = price_data.index.min()
        price_end = price_data.index.max()
        signal_start = signal_data.index.min()
        signal_end = signal_data.index.max()
        
        common_start = max(price_start, signal_start)
        common_end = min(price_end, signal_end)
        
        # Filter to common time range
        aligned_price = price_data[(price_data.index >= common_start) & 
                                  (price_data.index <= common_end)]
        aligned_signals = signal_data[(signal_data.index >= common_start) & 
                                     (signal_data.index <= common_end)]
        
        logger.info(f"Aligned data: {len(aligned_price)} price records, "
                   f"{len(aligned_signals)} signal records")
        
        return aligned_price, aligned_signals
    
    def get_universe(self, 
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None) -> List[str]:
        """
        Get list of available symbols in the dataset.
        
        Args:
            start_date: Start date filter
            end_date: End date filter
            
        Returns:
            List of available symbols
        """
        price_data = self.load_price_data(start_date=start_date, end_date=end_date)
        
        if 'symbol' in price_data.columns:
            return price_data['symbol'].unique().tolist()
        else:
            # If no symbol column, assume single symbol dataset
            return ['DEFAULT_SYMBOL']
    
    def _resample_price_data(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample price data to different timeframe"""
        if 'symbol' in df.columns:
            # Group by symbol and resample
            resampled = df.groupby('symbol').resample(timeframe).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min', 
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            return resampled.reset_index()
        else:
            # Single symbol resampling
            return df.resample(timeframe).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
    
    def _generate_sample_price_data(self, 
                                   symbols: Optional[List[str]] = None,
                                   start_date: Optional[datetime] = None,
                                   end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Generate sample price data for demo purposes"""
        if not symbols:
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
        
        if not start_date:
            start_date = datetime.now() - timedelta(days=30)
        if not end_date:
            end_date = datetime.now()
        
        logger.info("Generating sample price data for demo")
        
        # Generate minute-level data
        date_range = pd.date_range(start=start_date, end=end_date, freq='1min')
        
        all_data = []
        for symbol in symbols:
            # Initialize price with random walk
            np.random.seed(hash(symbol) % 2**32)  # Deterministic randomness per symbol
            initial_price = np.random.uniform(100, 300)
            
            prices = [initial_price]
            for _ in range(len(date_range) - 1):
                # Random walk with slight upward bias
                change = np.random.normal(0.0001, 0.01)  # Small drift, 1% volatility
                new_price = prices[-1] * (1 + change)
                prices.append(max(new_price, 1.0))  # Ensure positive prices
            
            for i, timestamp in enumerate(date_range):
                base_price = prices[i]
                
                # Generate OHLC from base price
                noise = np.random.uniform(-0.005, 0.005, 4)  # 0.5% noise
                open_price = base_price * (1 + noise[0])
                high_price = base_price * (1 + abs(noise[1]))
                low_price = base_price * (1 - abs(noise[2]))
                close_price = base_price * (1 + noise[3])
                
                # Ensure HLOC ordering
                high_price = max(high_price, open_price, low_price, close_price)
                low_price = min(low_price, open_price, high_price, close_price)
                
                volume = int(np.random.exponential(1000))  # Exponential distribution for volume
                
                all_data.append({
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'open': round(open_price, 2),
                    'high': round(high_price, 2),
                    'low': round(low_price, 2),
                    'close': round(close_price, 2),
                    'volume': volume
                })
        
        df = pd.DataFrame(all_data)
        df.set_index('timestamp', inplace=True)
        return df
    
    def _generate_sample_signal_data(self,
                                    symbols: Optional[List[str]] = None,
                                    start_date: Optional[datetime] = None,
                                    end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Generate sample signal data for demo purposes"""
        if not symbols:
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
        
        if not start_date:
            start_date = datetime.now() - timedelta(days=30)
        if not end_date:
            end_date = datetime.now()
        
        logger.info("Generating sample signal data for demo")
        
        # Generate signals every 5-15 minutes randomly
        all_signals = []
        
        for symbol in symbols:
            current_time = start_date
            signal_id = 0
            
            while current_time < end_date:
                # Random interval between signals (5-15 minutes)
                interval = np.random.randint(5, 16)
                current_time += timedelta(minutes=interval)
                
                if current_time >= end_date:
                    break
                
                # Generate signal
                signal_types = ['buy', 'sell', 'hold']
                signal_weights = [0.35, 0.35, 0.30]  # Slightly favor buy/sell
                signal = np.random.choice(signal_types, p=signal_weights)
                
                confidence = np.random.uniform(0.6, 0.95)
                score = np.random.uniform(-1.0, 1.0)
                
                sentiments = ['positive', 'negative', 'neutral']
                sentiment = np.random.choice(sentiments)
                
                headlines = [
                    f"{symbol} earnings beat expectations",
                    f"{symbol} announces new product launch",
                    f"Analyst upgrade for {symbol}",
                    f"{symbol} reports strong quarterly results",
                    f"Market volatility affects {symbol}",
                    f"{symbol} CEO announces strategic initiative"
                ]
                headline = np.random.choice(headlines)
                
                all_signals.append({
                    'timestamp': current_time,
                    'symbol': symbol,
                    'signal': signal,
                    'confidence': round(confidence, 3),
                    'score': round(score, 3),
                    'sentiment': sentiment,
                    'headline': headline,
                    'source': 'sample_generator',
                    'signal_id': f"{symbol}_{signal_id}"
                })
                
                signal_id += 1
        
        df = pd.DataFrame(all_signals)
        df.set_index('timestamp', inplace=True)
        return df 