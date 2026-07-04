import os
import json
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from loguru import logger

from db.models import AnomalyAlert
from db.session import get_session
from features.engineer import load_features_df

MODEL_DIR = Path("models/artifacts")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FEATURES = [
    "returns_1d", "returns_5d",
    "rsi_14", "macd", "macd_hist", "bb_pct", "atr_14",
    "sentiment_compound_mean", "news_count",
    "fed_funds_rate", "vix",
]


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sentiment_compound_mean"] = df["sentiment_compound_mean"].fillna(0.0)
    df["news_count"]              = df["news_count"].fillna(0)
    df["vix"]                     = df["vix"].ffill().bfill()
    df["fed_funds_rate"]          = df["fed_funds_rate"].ffill().bfill()
    return df.dropna(subset=MODEL_FEATURES)


def train_isolation_forest(df: pd.DataFrame, contamination: float = 0.05) -> IsolationForest:
    clean = _prep(df)
    X = clean[MODEL_FEATURES].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    joblib.dump(scaler, MODEL_DIR / "if_scaler.pkl")
    joblib.dump(model,  MODEL_DIR / "isolation_forest.pkl")
    logger.info(f"Isolation Forest trained on {len(clean)} rows, saved to {MODEL_DIR}")
    return model


def run_anomaly_detection(df: pd.DataFrame) -> pd.DataFrame:
    scaler: StandardScaler = joblib.load(MODEL_DIR / "if_scaler.pkl")
    model: IsolationForest  = joblib.load(MODEL_DIR / "isolation_forest.pkl")

    clean = _prep(df).copy()
    X_scaled = scaler.transform(clean[MODEL_FEATURES].values)

    clean["if_pred"]  = model.predict(X_scaled)        # 1=normal, -1=anomaly
    clean["if_score"] = model.decision_function(X_scaled)

    # Top contributing feature per row
    feature_means = X_scaled.mean(axis=0)
    deviations    = np.abs(X_scaled - feature_means)
    top_feat_idx  = np.argmax(deviations, axis=1)
    clean["top_feature"] = [MODEL_FEATURES[i] for i in top_feat_idx]

    results = []
    with get_session() as session:
        for _, row in clean.iterrows():
            ticker     = row["ticker"]
            row_date   = row["date"].date() if hasattr(row["date"], "date") else row["date"]
            is_anomaly = int(row["if_pred"]) == -1

            existing = session.query(AnomalyAlert).filter_by(
                ticker=ticker, date=row_date
            ).first()

            if existing:
                setattr(existing, "anomaly_score", float(row["if_score"]))
                setattr(existing, "is_anomaly", is_anomaly)
                setattr(existing, "top_features", json.dumps([row["top_feature"]]))
            else:
                session.add(AnomalyAlert(
                    ticker=ticker,
                    date=row_date,
                    anomaly_score=float(row["if_score"]),
                    is_anomaly=is_anomaly,
                    top_features=json.dumps([row["top_feature"]]),
                ))
            results.append({
                "ticker": ticker,
                "date": row_date,
                "is_anomaly": is_anomaly,
                "score": round(float(row["if_score"]), 4),
                "top_feature": row["top_feature"],
            })

    return pd.DataFrame(results)