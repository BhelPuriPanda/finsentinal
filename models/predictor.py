import os
import json
from datetime import date
from pathlib import Path
from typing import cast
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from loguru import logger

from db.models import Prediction
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
    df = df.dropna(subset=MODEL_FEATURES)
    df = df.dropna(subset=["next_day_up"])
    return df


def train_xgboost(df: pd.DataFrame) -> XGBClassifier:
    clean = _prep(df).sort_values(["ticker", "date"])

    X = clean[MODEL_FEATURES].values
    y = clean["next_day_up"].astype(int).values

    # Time-based split — no shuffle to prevent data leakage
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    val_preds = model.predict(X_val)
    report = cast(dict, classification_report(
        np.asarray(y_val),
        np.asarray(val_preds),
        output_dict=True
    ))
    logger.info(
        f"XGBoost trained on {len(X_train)} rows | "
        f"val_accuracy={report['accuracy']:.3f} | "
        f"F1_up={report['1']['f1-score']:.3f} | "
        f"F1_down={report['0']['f1-score']:.3f}"
    )

    # Feature importance
    importances = dict(zip(MODEL_FEATURES, model.feature_importances_))
    top = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
    logger.info(f"Top features: {top}")

    joblib.dump(model, MODEL_DIR / "xgboost.pkl")
    logger.info(f"XGBoost saved to {MODEL_DIR}")
    return model


def run_predictions(df: pd.DataFrame, model_version: str = "v1") -> pd.DataFrame:
    model: XGBClassifier = joblib.load(MODEL_DIR / "xgboost.pkl")

    clean = _prep(df).copy()

    # Latest row per ticker only
    latest = clean.sort_values("date").groupby("ticker").tail(1)

    results = []
    today = date.today()

    with get_session() as session:
        for _, row in latest.iterrows():
            ticker = row["ticker"]
            X = np.array([[row[f] for f in MODEL_FEATURES]])
            proba     = model.predict_proba(X)[0]
            prob_down = float(proba[0])
            prob_up   = float(proba[1])
            direction  = "UP" if prob_up >= 0.5 else "DOWN"
            confidence = max(prob_up, prob_down)

            existing = session.query(Prediction).filter_by(
                ticker=ticker, prediction_date=today
            ).first()

            if existing:
                setattr(existing, "direction", direction)
                setattr(existing, "confidence", confidence)
                setattr(existing, "prob_up", prob_up)
                setattr(existing, "prob_down", prob_down)
                setattr(existing, "model_version", model_version)
            else:
                session.add(Prediction(
                    ticker=ticker,
                    prediction_date=today,
                    target_date=today,
                    direction=direction,
                    confidence=confidence,
                    prob_up=prob_up,
                    prob_down=prob_down,
                    model_version=model_version,
                ))

            results.append({
                "ticker":     ticker,
                "direction":  direction,
                "confidence": round(confidence, 4),
                "prob_up":    round(prob_up, 4),
                "prob_down":  round(prob_down, 4),
            })

    return pd.DataFrame(results)
