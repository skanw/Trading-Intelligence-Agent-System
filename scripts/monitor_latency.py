"""
Monitor signal latency and alert if signals are stale.
"""
from redis import Redis
import pandas as pd
import datetime
import sys

def main():
    r = Redis.from_url("redis://localhost:6379/0", decode_responses=True)
    
    # Get the last entry in signals
    last = r.xrevrange("signals", count=1)
    
    if not last:
        print("ALERT: no signals in Redis!", file=sys.stderr)
        sys.exit(1)
    
    _, data = last[0]
    pub_ts = pd.to_datetime(data["published_at"])
    now = pd.Timestamp.utcnow()
    age = (now - pub_ts).total_seconds() / 60  # minutes
    
    print(f"Latest signal age: {age:.1f} min")
    
    if age > 5:
        print("ALERT: signal latency > 5 min", file=sys.stderr)
        sys.exit(1)
    
    print("Signal latency OK")

if __name__ == "__main__":
    main() 