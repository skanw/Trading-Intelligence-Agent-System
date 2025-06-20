"""
Train a LightGBM classifier to predict price movement direction from news headlines.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
import shap
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def prepare_features(df: pd.DataFrame) -> tuple:
    """Prepare features and target for training."""
    # Print available columns for debugging
    print("\nAvailable columns:", df.columns.tolist())
    
    # Create features
    X = pd.DataFrame()
    
    # Categorical features
    X['ticker'] = pd.Categorical(df['ticker']).codes
    X['source'] = pd.Categorical(df['source']).codes
    X['sentiment'] = pd.Categorical(df['sentiment']).codes
    
    # Numerical features
    numerical_features = [
        'sentiment_conf',
        'headline_len',
        'num_vals',
        'close',
        'forward_return'
    ]
    
    # Scale numerical features
    scaler = StandardScaler()
    X[numerical_features] = scaler.fit_transform(df[numerical_features])
    
    # Interaction features
    X['sentiment_strength'] = X['sentiment'] * X['sentiment_conf']
    X['price_volume'] = X['close'] * X['num_vals']
    
    # Target variable
    y = df['target_move']
    
    print("\nFeature correlations with target:")
    correlations = pd.DataFrame({
        'feature': X.columns,
        'correlation': [abs(X[col].corr(y)) for col in X.columns]
    }).sort_values('correlation', ascending=False)
    print(correlations)
    
    return X, y

def train_model(X: pd.DataFrame, y: pd.Series) -> lgb.LGBMClassifier:
    """Train a LightGBM classifier using leave-one-out cross-validation."""
    # Initialize model with parameters suitable for small dataset
    model = lgb.LGBMClassifier(
        n_estimators=50,
        learning_rate=0.1,
        num_leaves=4,  # Prevent overfitting
        min_child_samples=2,  # Allow small leaf sizes
        subsample=0.8,
        colsample_bytree=0.8,  # Feature subsampling
        random_state=42,
        verbose=-1
    )
    
    # Perform leave-one-out cross-validation
    cv = LeaveOneOut()
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    
    print("\nCross-validation results:")
    print(f"Mean accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
    print("Individual fold accuracies:", scores)
    
    # Train final model on all data
    model.fit(X, y)
    
    # Print feature importance
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(importance)
    
    return model

def plot_shap_summary(model: lgb.LGBMClassifier, X: pd.DataFrame):
    """Generate and save SHAP summary plot."""
    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # Create summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig('models/shap_summary.png')
    plt.close()

def main():
    # Create models directory if it doesn't exist
    Path('models').mkdir(exist_ok=True)
    
    # Load data
    df = pd.read_parquet('data/train.parquet')
    print("Loaded dataset shape:", df.shape)
    
    # Prepare features
    X, y = prepare_features(df)
    print("\nFeatures:", X.columns.tolist())
    
    # Train model
    model = train_model(X, y)
    
    # Generate SHAP plot
    plot_shap_summary(model, X)
    
    # Save model
    model_path = 'models/lgbm_headline.pkl'
    pd.to_pickle(model, model_path)
    print(f"\nModel saved to {model_path}")

if __name__ == '__main__':
    main() 