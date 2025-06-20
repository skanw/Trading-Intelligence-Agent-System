"""
Async workers:
  • XREAD raw headline → build features
  • Predict impact with LGBM → push to 'signals' stream
  • Streamlit/Slack read 'signals'
Run with:  uvicorn src.realtime.orchestrator:app
"""
import asyncio, joblib, json, pandas as pd
import numpy as np
from redis.asyncio import Redis
from src.config import settings
from src.features.feature_builder import build_features

model = joblib.load("models/lgbm_headline.pkl")
REQUIRED_FEATURES = ['ticker', 'source', 'sentiment', 'sentiment_conf', 'headline_len', 
                    'num_vals', 'close', 'forward_return', 'sentiment_strength', 'price_volume']

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare features in the same format as training data."""
    X = pd.DataFrame()
    
    # Categorical features
    X['ticker'] = pd.Categorical(df['ticker']).codes
    X['source'] = pd.Categorical(df['source']).codes
    X['sentiment'] = pd.Categorical(df['sentiment']).codes
    
    # Numerical features (use dummy values for live prediction)
    X['sentiment_conf'] = df['sentiment_conf']
    X['headline_len'] = df['headline_len']
    X['num_vals'] = df['num_vals']
    X['close'] = 0.0  # Dummy value
    X['forward_return'] = 0.0  # Dummy value
    
    # Interaction features
    X['sentiment_strength'] = X['sentiment'] * X['sentiment_conf']
    X['price_volume'] = X['close'] * X['num_vals']
    
    return X[REQUIRED_FEATURES]

async def worker():
    redis = Redis.from_url(settings.REDIS_URL, decode_responses=True)
    last_id = "0-0"
    while True:
        res = await redis.xread({"news_raw": last_id}, block=0, count=10)
        for stream, msgs in res:
            for msg_id, data in msgs:
                last_id = msg_id
                # Handle nested data structure
                if "data" in data:
                    data = json.loads(data["data"])
                feats_df = build_features(json.dumps(data))
                if not feats_df.empty:
                    # Prepare features in the same format as training
                    X = prepare_features(feats_df)
                    proba = model.predict_proba(X)[0, 1]
                    # Convert datetime to string for Redis
                    feats_dict = feats_df.iloc[0].to_dict()
                    feats_dict['published_at'] = feats_dict['published_at'].isoformat()
                    await redis.xadd("signals", {"score": proba, **feats_dict})

if __name__ == "__main__":
    asyncio.run(worker()) 