import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from features.engineer import load_features_df
from models.predictor import train_xgboost, run_predictions

print("Loading features...")
df = load_features_df(lookback_days=400)
print(f"Loaded {len(df)} rows")

print("\nTraining XGBoost...")
train_xgboost(df)

print("\nRunning predictions...")
results = run_predictions(df)

print("\nNext-day predictions:")
print(results.to_string(index=False))