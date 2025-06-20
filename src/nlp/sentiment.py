from textblob import TextBlob

def score(text: str) -> dict:
    """Return {'label': 'POSITIVE', 'score': 0.92} style dict."""
    analysis = TextBlob(text)
    # Convert polarity (-1 to 1) to confidence score (0 to 1)
    score = (analysis.sentiment.polarity + 1) / 2
    
    label = "POSITIVE" if score > 0.6 else "NEGATIVE" if score < 0.4 else "NEUTRAL"
    return {"label": label, "score": score} 