#!/usr/bin/env bash
# one-liner: curl -sSL https://raw.githubusercontent.com/youruser/eventpulse-ai/main/bootstrap.sh | bash

set -e
python -m venv .venv
source .venv/bin/activate

# Core libs
pip install --upgrade pip wheel
pip install -r requirements.txt

# spaCy model
python -m spacy download en_core_web_sm

# Pre-pull FinBERT so first inference is fast
python - <<'PY'
from transformers import pipeline
pipeline("sentiment-analysis", model="ProsusAI/finbert", tokenizer="ProsusAI/finbert")
PY

echo "✅  Environment ready. Activate with: source .venv/bin/activate" 