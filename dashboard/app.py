import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
from datetime import date, timedelta
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from db.session import init_db, get_session
from db.models import RawPrice, RawNews, Features, AnomalyAlert, Prediction
from sqlalchemy import select

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="FinSentinel",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

TICKERS = os.getenv("TICKERS", "AAPL,TSLA,GOOGL,MSFT,NVDA").split(",")

PALETTE = {
    "bg":    "#0e1117",
    "card":  "#1c1e26",
    "green": "#00d084",
    "red":   "#ff4d4d",
    "blue":  "#4da6ff",
    "gold":  "#ffd700",
    "text":  "#e0e0e0",
    "muted": "#888888",
}

# ── Init DB ───────────────────────────────────────────────────
init_db()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 FinSentinel")
    st.markdown("*Sentiment-Augmented Financial Analytics*")
    st.divider()

    ticker   = st.selectbox("Ticker", TICKERS)
    lookback = st.slider("Lookback (days)", 30, 365, 90)

    st.divider()
    st.markdown("**Data Coverage**")
    with get_session() as session:
        price_count = session.query(RawPrice).filter(
            RawPrice.ticker == ticker
        ).count()
        news_count = session.query(RawNews).filter(
            RawNews.ticker == ticker
        ).count()
        feature_count = session.query(Features).filter(
            Features.ticker == ticker
        ).count()

    st.metric("Price rows",   price_count)
    st.metric("News rows",    news_count)
    st.metric("Feature rows", feature_count)

    st.divider()
    st.caption(f"Last updated: {date.today().isoformat()}")
    st.caption("Pipeline: daily at 4:30 PM ET")

# ── Data loaders ──────────────────────────────────────────────
@st.cache_data(ttl=900)
def load_prices(ticker: str, days: int):
    import pandas as pd
    start = date.today() - timedelta(days=days)
    with get_session() as session:
        rows = session.execute(
            select(RawPrice)
            .where(RawPrice.ticker == ticker, RawPrice.date >= start)
            .order_by(RawPrice.date)
        ).scalars().all()
        # Convert inside session while still attached
        records = [{
            "date": r.date, "open": r.open, "high": r.high,
            "low": r.low, "close": r.close, "volume": r.volume,
        } for r in rows]
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


@st.cache_data(ttl=900)
def load_features(ticker: str, days: int):
    import pandas as pd
    start = date.today() - timedelta(days=days)
    with get_session() as session:
        rows = session.execute(
            select(Features)
            .where(Features.ticker == ticker, Features.date >= start)
            .order_by(Features.date)
        ).scalars().all()
        records = [{
            "date": r.date, "close": r.close,
            "rsi_14": r.rsi_14, "macd": r.macd,
            "macd_signal": r.macd_signal, "macd_hist": r.macd_hist,
            "bb_upper": r.bb_upper, "bb_lower": r.bb_lower, "bb_mid": r.bb_mid,
            "sentiment_compound_mean": r.sentiment_compound_mean,
            "news_count": r.news_count, "vix": r.vix,
        } for r in rows]
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


@st.cache_data(ttl=900)
def load_anomalies(days: int):
    import pandas as pd
    start = date.today() - timedelta(days=days)
    with get_session() as session:
        rows = session.execute(
            select(AnomalyAlert)
            .where(AnomalyAlert.is_anomaly == True,
                   AnomalyAlert.date >= start)
            .order_by(AnomalyAlert.date.desc())
        ).scalars().all()
        records = [{
            "Date":          r.date,
            "Ticker":        r.ticker,
            "Anomaly Score": round(r.anomaly_score, 4) if r.anomaly_score is not None else None,# type: ignore
            "Top Feature":   r.top_features,
        } for r in rows]
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


@st.cache_data(ttl=900)
def load_predictions():
    import pandas as pd
    with get_session() as session:
        rows = session.execute(
            select(Prediction)
            .order_by(Prediction.prediction_date.desc())
        ).scalars().all()
        seen, unique = set(), []
        for r in rows:
            if r.ticker not in seen:
                seen.add(r.ticker)
                unique.append({
                    "Ticker":     r.ticker,
                    "Direction":  r.direction,
                    "Confidence": r.confidence,
                    "P(Up)":      r.prob_up,
                    "P(Down)":    r.prob_down,
                    "Date":       r.prediction_date,
                })
    if not unique:
        return pd.DataFrame()
    return pd.DataFrame(unique)

# ── Main area placeholder ─────────────────────────────────────
prices   = load_prices(ticker, lookback)
features = load_features(ticker, lookback)

st.title(f"FinSentinel — {ticker}")
st.caption(f"Showing {lookback} days of data")

col1, col2, col3, col4 = st.columns(4)
with col1:
    if not prices.empty:
        latest = prices["close"].iloc[-1]
        prev   = prices["close"].iloc[-2] if len(prices) > 1 else latest
        st.metric("Latest Close", f"${latest:.2f}", f"{((latest-prev)/prev)*100:.2f}%")

with col2:
    if not features.empty:
        rsi = features["rsi_14"].iloc[-1]
        st.metric("RSI (14)", f"{rsi:.1f}", "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral")

with col3:
    if not features.empty:
        vix = features["vix"].iloc[-1]
        st.metric("VIX", f"{vix:.2f}")

with col4:
    preds = load_predictions()
    if not preds.empty:
        row = preds[preds["Ticker"] == ticker]
        if not row.empty:
            direction = row["Direction"].iloc[0]
            conf      = row["Confidence"].iloc[0]
            st.metric("Prediction", direction, f"{conf:.1%} confidence")

st.divider()
st.info("📊 Panels loading in Days 12–15. Shell is live.")