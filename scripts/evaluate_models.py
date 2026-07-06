import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from typing import cast
import json
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score
)
from db.session import get_session
from db.models import AnomalyAlert, Prediction
from features.engineer import load_features_df
from models.anomaly import _prep, MODEL_FEATURES as IF_FEATURES
from models.predictor import _prep as xgb_prep, MODEL_FEATURES as XGB_FEATURES
import joblib
from pathlib import Path

MODEL_DIR = Path("models/artifacts")

print("=" * 60)
print("FINSENTINEL — Week 2 Model Evaluation")
print("=" * 60)

df = load_features_df(lookback_days=400)
print(f"\nFeature rows loaded : {len(df)}")

# ── Isolation Forest ──────────────────────────────────────────
print("\n[ Isolation Forest ]")

scaler = joblib.load(MODEL_DIR / "if_scaler.pkl")
model_if = joblib.load(MODEL_DIR / "isolation_forest.pkl")

clean_if = _prep(df)
X_if = scaler.transform(clean_if[IF_FEATURES].values)
scores = model_if.decision_function(X_if)
preds  = model_if.predict(X_if)

n_anomalies = (preds == -1).sum()
print(f"  Rows scored       : {len(preds)}")
print(f"  Anomalies flagged : {n_anomalies} ({n_anomalies/len(preds)*100:.1f}%)")
print(f"  Score range       : {scores.min():.4f} → {scores.max():.4f}")
print(f"  Score mean        : {scores.mean():.4f}")

# Anomalies per ticker
with get_session() as session:
    rows = session.query(AnomalyAlert).filter(
        AnomalyAlert.is_anomaly == True
    ).all()
    anomaly_records = [{
        "ticker": r.ticker, "date": r.date,
        "score": r.anomaly_score, "top_feature": r.top_features
    } for r in rows]
anomaly_df = pd.DataFrame(anomaly_records)

print(f"\n  Anomalies per ticker:")
for ticker, group in anomaly_df.groupby("ticker"):
    print(f"    {ticker:5s} : {len(group)} anomalies")

print(f"\n  Top contributing features:")
all_features = []
for tf in anomaly_df["top_feature"]:
    try:
        all_features.extend(json.loads(tf))
    except:
        pass
feat_counts = pd.Series(all_features).value_counts()
for feat, count in feat_counts.items():
    print(f"    {feat:30s} : {count}")

# ── XGBoost ───────────────────────────────────────────────────
print("\n[ XGBoost Classifier ]")

model_xgb = joblib.load(MODEL_DIR / "xgboost.pkl")
clean_xgb = xgb_prep(df).sort_values(["ticker", "date"])

X = clean_xgb[XGB_FEATURES].values
y = clean_xgb["next_day_up"].astype(int).values

split = int(len(X) * 0.8)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

val_preds  = np.asarray(model_xgb.predict(X_val))
val_proba  = np.asarray(model_xgb.predict_proba(X_val)[:, 1])
y_val_arr  = np.asarray(y_val)

acc    = accuracy_score(y_val_arr, val_preds)
try:
    auc = roc_auc_score(y_val_arr, val_proba)
except:
    auc = float("nan")

print(f"  Train rows        : {len(X_train)}")
print(f"  Val rows          : {len(X_val)}")
print(f"  Val accuracy      : {acc:.3f}")
print(f"  Val ROC-AUC       : {auc:.3f}")

print(f"\n  Classification report (val):")
report = cast(str, classification_report(y_val_arr, val_preds, target_names=["DOWN", "UP"]))
for line in report.split("\n"):
    print(f"    {line}")

print(f"\n  Confusion matrix (val):")
cm = confusion_matrix(y_val_arr, val_preds)
print(f"    Predicted →    DOWN   UP")
print(f"    Actual DOWN  : {cm[0][0]:5d}  {cm[0][1]:5d}")
print(f"    Actual UP    : {cm[1][0]:5d}  {cm[1][1]:5d}")

print(f"\n  Feature importances:")
importances = dict(zip(XGB_FEATURES, model_xgb.feature_importances_))
for feat, imp in sorted(importances.items(), key=lambda x: x[1], reverse=True):
    bar = "█" * int(imp * 200)
    print(f"    {feat:30s} : {imp:.4f} {bar}")

# ── Today's predictions ───────────────────────────────────────
print("\n[ Today's Predictions ]")
with get_session() as session:
    preds_today = session.query(Prediction).all()
    pred_records = [{
        "Ticker": r.ticker,
        "Direction": r.direction,
        "Confidence": f"{r.confidence:.1%}",
        "P(Up)": f"{r.prob_up:.1%}",
        "P(Down)": f"{r.prob_down:.1%}",
    } for r in preds_today]
pred_df = pd.DataFrame(pred_records)
print(pred_df.to_string(index=False))

# ── Assertions ────────────────────────────────────────────────
print("\n[ Assertions ]")
checks = [
    (n_anomalies > 0,          "Isolation Forest flagged at least 1 anomaly"),
    (n_anomalies < len(preds) * 0.15, "Anomaly rate < 15%"),
    (acc > 0.45,               f"XGBoost val accuracy > 45% (got {acc:.3f})"),
    (auc > 0.45,               f"ROC-AUC > 45% (got {auc:.3f})"),
    (len(pred_df) == 5,        "Predictions exist for all 5 tickers"),
    (Path(MODEL_DIR / "xgboost.pkl").exists(),        "XGBoost artifact exists"),
    (Path(MODEL_DIR / "isolation_forest.pkl").exists(),"IF artifact exists"),
]

all_passed = True
for passed, label in checks:
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {label}")
    if not passed:
        all_passed = False

print("\n" + "=" * 60)
print("  ALL CHECKS PASSED" if all_passed else "  SOME CHECKS FAILED")
print("=" * 60)
