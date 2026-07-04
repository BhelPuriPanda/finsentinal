import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from features.engineer import load_features_df
from models.anomaly import train_isolation_forest, run_anomaly_detection

print("Loading features...")
df = load_features_df(lookback_days=400)
print(f"Loaded {len(df)} rows")

print("\nTraining Isolation Forest...")
train_isolation_forest(df)

print("\nRunning anomaly detection...")
results = run_anomaly_detection(df)

anomalies = results[results["is_anomaly"] == True]
print(f"\nTotal rows scored : {len(results)}")
print(f"Anomalies flagged : {len(anomalies)}")
print(f"\nTop anomalies:")
print(anomalies.sort_values("score").head(10).to_string(index=False))