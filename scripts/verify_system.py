"""
System verification script to check all EventPulse AI components.
"""
import sys
from pathlib import Path
import subprocess
import json
import datetime

def check_redis():
    """Check if Redis is accessible."""
    try:
        from redis import Redis
        r = Redis.from_url("redis://localhost:6379/0", decode_responses=True)
        r.ping()
        print("✅ Redis: Connected")
        return True
    except Exception as e:
        print(f"❌ Redis: Failed - {e}")
        return False

def check_model():
    """Check if model file exists and is loadable."""
    try:
        import joblib
        model_path = Path("models/lgbm_headline.pkl")
        if not model_path.exists():
            print("❌ Model: File not found")
            return False
        
        model = joblib.load(model_path)
        print(f"✅ Model: Loaded ({model_path.stat().st_size} bytes)")
        print(f"   Features: {len(model.feature_name_)} expected")
        return True
    except Exception as e:
        print(f"❌ Model: Failed to load - {e}")
        return False

def check_environment():
    """Check if virtual environment and dependencies are set up."""
    try:
        import pandas, redis, lightgbm, streamlit, yfinance
        print("✅ Dependencies: All core packages available")
        return True
    except ImportError as e:
        print(f"❌ Dependencies: Missing package - {e}")
        return False

def check_data():
    """Check if training data exists."""
    data_path = Path("data/train.parquet")
    if data_path.exists():
        import pandas as pd
        df = pd.read_parquet(data_path)
        print(f"✅ Training Data: {len(df)} samples")
        return True
    else:
        print("⚠️  Training Data: Not found (run collect_training_data.py)")
        return False

def test_feature_pipeline():
    """Test the feature building pipeline."""
    try:
        from src.features.feature_builder import build_features
        
        test_article = {
            "title": "AAPL reports strong Q2 earnings",
            "description": "Apple beats estimates with revenue of $90B",
            "publishedAt": datetime.datetime.utcnow().isoformat() + "Z",
            "source": {"name": "TestSource"}
        }
        
        features = build_features(json.dumps(test_article))
        if not features.empty:
            print(f"✅ Feature Pipeline: Generated {len(features.columns)} features")
            return True
        else:
            print("❌ Feature Pipeline: No features generated")
            return False
    except Exception as e:
        print(f"❌ Feature Pipeline: Failed - {e}")
        return False

def check_signals():
    """Check if signals are being generated."""
    try:
        from redis import Redis
        r = Redis.from_url("redis://localhost:6379/0", decode_responses=True)
        
        signals = r.xrevrange("signals", count=1)
        if signals:
            latest = signals[0][1]
            print(f"✅ Signals: Latest score = {latest.get('score', 'N/A')}")
            return True
        else:
            print("⚠️  Signals: No signals in Redis (orchestrator may not be running)")
            return False
    except Exception as e:
        print(f"❌ Signals: Failed to check - {e}")
        return False

def main():
    """Run all system checks."""
    print("EventPulse AI - System Verification")
    print("=" * 40)
    
    checks = [
        ("Environment", check_environment),
        ("Redis", check_redis),
        ("Model", check_model),
        ("Training Data", check_data),
        ("Feature Pipeline", test_feature_pipeline),
        ("Signals", check_signals),
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\nChecking {name}...")
        results.append(check_func())
    
    print("\n" + "=" * 40)
    print("SYSTEM STATUS SUMMARY")
    print("=" * 40)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 ALL CHECKS PASSED ({passed}/{total})")
        print("\nSystem is ready for production!")
    else:
        print(f"⚠️  {passed}/{total} checks passed")
        print("\nSome components need attention.")
    
    print("\nNext steps:")
    print("1. Start orchestrator: python -m src.realtime.orchestrator")
    print("2. Start dashboard: streamlit run src/dashboard/app.py")
    print("3. Monitor signals: python scripts/monitor_latency.py")

if __name__ == "__main__":
    main() 