import json, logging, re
from datetime import datetime
import pandas as pd
from src.nlp.sentiment import score as sent_score

NUM_RE = re.compile(r"(-?\d[\d,\.]*)")

def build_features(raw_json: str) -> pd.DataFrame:
    art = json.loads(raw_json)
    sent = sent_score(art["title"] + ". " + (art["description"] or ""))
    feats = {
        "ticker": extract_ticker(art["title"]),
        "published_at": pd.to_datetime(art["publishedAt"]),
        "source": art["source"]["name"],
        "sentiment": sent["label"],
        "sentiment_conf": sent["score"],
        "headline_len": len(art["title"]),
        "num_vals": len(NUM_RE.findall(art["title"])),
    }
    return pd.DataFrame([feats])

def extract_ticker(text: str):
    # naive: $TSLA or (TSLA) pattern
    m = re.search(r"\$?([A-Z]{2,5})\b", text)
    return m.group(1) if m else None 