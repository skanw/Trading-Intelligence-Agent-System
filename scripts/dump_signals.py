"""
Dump all signals from Redis to a parquet file for analysis and backtesting.
"""
import pandas as pd
from redis import Redis
from pathlib import Path

def main():
    r = Redis.from_url("redis://localhost:6379/0", decode_responses=True)
    
    # Dump all entries from the 'signals' stream
    entries = r.xrange("signals", min='-', max='+')
    
    if not entries:
        print("No signals found in Redis")
        return
    
    rows = [e[1] for e in entries]
    df = pd.DataFrame(rows)
    
    # Convert data types
    df["published_at"] = pd.to_datetime(df["published_at"])
    df["score"] = df["score"].astype(float)
    df["sentiment_conf"] = df["sentiment_conf"].astype(float)
    df["headline_len"] = df["headline_len"].astype(int)
    df["num_vals"] = df["num_vals"].astype(int)
    
    # Ensure data directory exists
    Path("data").mkdir(exist_ok=True)
    
    # Save to parquet
    output_file = "data/signal_history.parquet"
    df.to_parquet(output_file, index=False)
    
    print(f"Wrote {len(df)} signals to {output_file}")
    print("\nSignal summary:")
    print(df.describe())
    print("\nSentiment distribution:")
    print(df["sentiment"].value_counts())

if __name__ == "__main__":
    main() 