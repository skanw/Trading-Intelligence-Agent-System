import streamlit as st, pandas as pd
from redis import Redis
from src.config import settings

redis = Redis.from_url(settings.REDIS_URL, decode_responses=True)

st.title("⚡ EventPulse – Live Signal Feed")

msgs = redis.xrevrange("signals", count=50)
df = pd.DataFrame([dict(m[1]) for m in msgs])
if not df.empty:
    st.dataframe(df.sort_values("published_at", ascending=False))
    st.altair_chart(
        df.groupby("sentiment")["score"].mean()
        .reset_index()
        .rename(columns={"score": "avg_prob"})
        .set_index("sentiment"),
        use_container_width=True,
    )
else:
    st.info("No signals yet – keep the worker running!") 