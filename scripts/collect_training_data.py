"""
Collect historical headlines and price data for model training.
This script:
1. Downloads historical headlines from NewsAPI
2. Fetches corresponding price data from yfinance
3. Computes target labels based on price moves
4. Saves the dataset as a parquet file
"""
import asyncio
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from newsapi import NewsApiClient
from pathlib import Path
from tqdm import tqdm
import json

from src.config import settings
from src.features.feature_builder import build_features

async def collect_headlines(api_key: str, tickers: list, days_back: int = 30) -> pd.DataFrame:
    """Collect historical headlines for each ticker."""
    client = NewsApiClient(api_key=api_key)
    all_articles = []
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    # Format dates as YYYY-MM-DDTHH:MM:SS
    start_str = start_date.strftime("%Y-%m-%dT%H:%M:%S")
    end_str = end_date.strftime("%Y-%m-%dT%H:%M:%S")
    
    for ticker in tqdm(tickers, desc="Collecting headlines"):
        try:
            articles = client.get_everything(
                q=ticker,
                from_param=start_str,
                to=end_str,
                language='en',
                sort_by='publishedAt',
            )
            
            for article in articles['articles']:
                features = build_features(json.dumps(article))
                if features['ticker'].iloc[0] == ticker:  # Only keep if ticker match
                    all_articles.append(features)
        except Exception as e:
            print(f"Error fetching articles for {ticker}: {e}")
    
    return pd.concat(all_articles, ignore_index=True) if all_articles else pd.DataFrame()

def get_price_data(tickers: list, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Download minute-level price data for each ticker."""
    all_prices = []
    for ticker in tqdm(tickers, desc="Fetching prices"):
        try:
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval='1m'
            )
            all_prices.append(df)
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
    
    if not all_prices:
        return pd.DataFrame()
        
    # Combine all price data
    prices_df = pd.concat(all_prices, axis=1)
    
    # Debug info about the structure
    print("\nPrice DataFrame Structure:")
    print("Column levels:", prices_df.columns.nlevels)
    print("Column names:", prices_df.columns.values)
    
    return prices_df

def compute_target(headlines_df: pd.DataFrame,
                  prices_df: pd.DataFrame,
                  horizon: str = "30min") -> pd.DataFrame:
    """
    Compute target variable based on forward returns.
    
    Args:
        headlines_df: DataFrame with columns ['ticker','published_at', ...] (UTC, sorted)
        prices_df: yfinance download with MultiIndex columns
        horizon: forward window used to build the label
    """
    if headlines_df.empty or prices_df.empty:
        return pd.DataFrame()
    
    # Debug info
    print("\nHeadlines DataFrame Info:")
    print(headlines_df.info())
    print("\nPrices DataFrame Structure:")
    print("Column levels:", prices_df.columns.nlevels)
    print("Column names:", prices_df.columns.values)
    
    # --- 1️⃣ Get Close prices in long format ----------
    # Reset index to make Datetime a column
    prices_df = prices_df.reset_index()
    
    # Keep only Close prices and reshape to long format
    close_prices = []
    for ticker in headlines_df['ticker'].unique():
        try:
            # Extract close prices for this ticker
            df = pd.DataFrame({
                'Datetime': prices_df[('Datetime', '')],  # Access the Datetime column
                'ticker': ticker,
                'close': prices_df[('Close', ticker)]  # Access multi-index column
            })
            close_prices.append(df)
        except Exception as e:
            print(f"Warning: Error processing {ticker}: {e}")
            continue
    
    if not close_prices:
        print("No valid price data found for any ticker")
        return pd.DataFrame()
        
    close_long = pd.concat(close_prices, ignore_index=True)
    close_long = close_long.sort_values(['ticker', 'Datetime'])
    
    # --- 2️⃣ Forward return over `horizon` -------------------------
    step = int(pd.Timedelta(horizon) / pd.Timedelta("1min"))  # rows = 1-min bars
    close_long["future_close"] = close_long.groupby("ticker")["close"].shift(-step)
    close_long["forward_return"] = (
        close_long["future_close"] - close_long["close"]
    ) / close_long["close"]
    
    # Ensure datetime columns are in the right format
    headlines_df['published_at'] = pd.to_datetime(headlines_df['published_at'])
    close_long['Datetime'] = pd.to_datetime(close_long['Datetime'])

    # --- 3️⃣ Merge headlines ↔ price snapshot just **after** headline ----
    merged = pd.merge_asof(
        headlines_df.sort_values("published_at"),
        close_long.sort_values("Datetime"),
        left_on="published_at",
        right_on="Datetime",
        by="ticker",
        direction="forward",          # first bar *after* the headline
        tolerance=pd.Timedelta("2h"), # optional: skip if no bar within 2 h
    )
    
    if merged.empty:
        print("\nNo matches found in merge_asof!")
        return pd.DataFrame()
    
    # Drop rows with missing price data
    merged = merged.dropna(subset=['close', 'forward_return'])
    
    if len(merged) == 0:
        print("No valid data after dropping missing values")
        return pd.DataFrame()
    
    # Convert to classification target (-1, 0, 1) based on percentile thresholds
    # Use 40th and 60th percentiles for a more balanced distribution
    lower_threshold = merged['forward_return'].quantile(0.40)
    upper_threshold = merged['forward_return'].quantile(0.60)
    
    merged['target_move'] = 0
    merged.loc[merged['forward_return'] > upper_threshold, 'target_move'] = 1
    merged.loc[merged['forward_return'] < lower_threshold, 'target_move'] = -1
    
    # Debug info about the results
    print("\nMerged Data Info:")
    print(merged.info())
    print("\nTarget Distribution:")
    print(merged['target_move'].value_counts(normalize=True))
    print("\nForward Return Stats:")
    print(merged['forward_return'].describe())
    print("\nThresholds:")
    print(f"Lower (40th percentile): {lower_threshold:.6f}")
    print(f"Upper (60th percentile): {upper_threshold:.6f}")
    
    return merged

def main():
    output_path = Path('data/train.parquet')
    tickers = settings.TRACKED_TICKERS.split(',')
    
    print("Starting data collection...")
    
    # Collect data
    headlines_df = asyncio.run(collect_headlines(
        settings.NEWS_API_KEY,
        tickers
    ))
    
    if headlines_df.empty:
        print("No headlines found. Exiting.")
        return
        
    print(f"Collected {len(headlines_df)} headlines")
    
    start_date = headlines_df['published_at'].min()
    end_date = headlines_df['published_at'].max()
    
    prices_df = get_price_data(tickers, start_date, end_date)
    
    if prices_df.empty:
        print("No price data found. Exiting.")
        return
        
    print(f"Collected price data from {start_date} to {end_date}")
    
    # Compute targets and save
    final_df = compute_target(headlines_df, prices_df)
    
    if not final_df.empty:
        # Select relevant columns for training
        cols_to_save = [
            'ticker', 'published_at', 'source', 'sentiment', 'sentiment_conf',
            'headline_len', 'num_vals', 'close', 'forward_return', 'target_move'
        ]
        final_df[cols_to_save].to_parquet(output_path)
        
        print(f"\nSaved training dataset to {output_path}")
        print(f"Shape: {final_df.shape}")
    else:
        print("No data to save after processing.")

if __name__ == '__main__':
    main() 