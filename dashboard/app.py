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

# ── Panel 1: Candlestick + Technicals ─────────────────────────
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.subheader(f"📊 Price & Technicals")

if prices.empty:
    st.info("No price data. Run the pipeline first.")
else:
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.20],
        vertical_spacing=0.04,
        subplot_titles=("Price + Bollinger Bands", "RSI (14)", "MACD"),
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=prices.index,
        open=prices["open"],
        high=prices["high"],
        low=prices["low"],
        close=prices["close"],
        increasing_line_color=PALETTE["green"],
        decreasing_line_color=PALETTE["red"],
        name="OHLC",
    ), row=1, col=1)

    # Bollinger Bands
    if not features.empty and "bb_upper" in features.columns:
        fig.add_trace(go.Scatter(
            x=features.index, y=features["bb_upper"],
            line=dict(color=PALETTE["blue"], width=1, dash="dot"),
            name="BB Upper",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=features.index, y=features["bb_lower"],
            line=dict(color=PALETTE["blue"], width=1, dash="dot"),
            fill="tonexty",
            fillcolor="rgba(77,166,255,0.07)",
            name="BB Lower",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=features.index, y=features["bb_mid"],
            line=dict(color=PALETTE["muted"], width=1),
            name="BB Mid",
        ), row=1, col=1)

    # RSI
    if not features.empty and "rsi_14" in features.columns:
        fig.add_trace(go.Scatter(
            x=features.index, y=features["rsi_14"],
            line=dict(color=PALETTE["gold"], width=1.5),
            name="RSI 14",
        ), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot",
                      line_color=PALETTE["red"],   row=2, col=1)
        fig.add_hline(y=30, line_dash="dot",
                      line_color=PALETTE["green"], row=2, col=1)

    # MACD
    if not features.empty and "macd" in features.columns:
        colors = [
            PALETTE["green"] if v >= 0 else PALETTE["red"]
            for v in features["macd_hist"].fillna(0)
        ]
        fig.add_trace(go.Bar(
            x=features.index, y=features["macd_hist"],
            marker_color=colors, name="MACD Hist", opacity=0.7,
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=features.index, y=features["macd"],
            line=dict(color=PALETTE["blue"], width=1.5),
            name="MACD",
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=features.index, y=features["macd_signal"],
            line=dict(color=PALETTE["gold"], width=1.5),
            name="Signal",
        ), row=3, col=1)

    fig.update_layout(
        height=620,
        paper_bgcolor=PALETTE["bg"],
        plot_bgcolor=PALETTE["bg"],
        font_color=PALETTE["text"],
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    fig.update_xaxes(gridcolor="#2a2d3a", showgrid=True)
    fig.update_yaxes(gridcolor="#2a2d3a", showgrid=True)

    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Panel 2: Sentiment Timeline ───────────────────────────────
st.subheader(f"🗞️ News Sentiment vs. Price")

if features.empty or "sentiment_compound_mean" not in features.columns:
    st.info("No sentiment data available.")
else:
    sent = features["sentiment_compound_mean"].dropna()

    if sent.empty:
        st.info("No sentiment data for this period.")
    else:
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])

        # Price line
        fig2.add_trace(go.Scatter(
            x=prices.index,
            y=prices["close"],
            line=dict(color=PALETTE["blue"], width=2),
            name="Close Price",
        ), secondary_y=False)

        # Sentiment bars
        bar_colors = [
            PALETTE["green"] if v >= 0 else PALETTE["red"]
            for v in sent
        ]
        fig2.add_trace(go.Bar(
            x=sent.index,
            y=sent,
            marker_color=bar_colors,
            name="Sentiment (VADER)",
            opacity=0.6,
        ), secondary_y=True)

        # Zero line on sentiment axis
        fig2.add_hline(
            y=0, line_dash="dot",
            line_color=PALETTE["muted"],
            secondary_y=True,
        )

        fig2.update_layout(
            height=380,
            paper_bgcolor=PALETTE["bg"],
            plot_bgcolor=PALETTE["bg"],
            font_color=PALETTE["text"],
            margin=dict(l=40, r=40, t=40, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        fig2.update_xaxes(gridcolor="#2a2d3a", showgrid=True)
        fig2.update_yaxes(
            title_text="Price ($)",
            gridcolor="#2a2d3a",
            secondary_y=False,
        )
        fig2.update_yaxes(
            title_text="Sentiment Score",
            range=[-1, 1],
            gridcolor="#2a2d3a",
            secondary_y=True,
        )

        st.plotly_chart(fig2, use_container_width=True)

        # Summary stats below chart
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Sentiment", f"{sent.mean():+.3f}")
        with col2:
            st.metric("Latest Sentiment", f"{sent.iloc[-1]:+.3f}")
        with col3:
            positive_days = (sent > 0).sum()
            st.metric("Positive Days", f"{positive_days}/{len(sent)}")

st.divider()

# ── Panel 3: Anomaly Alerts ───────────────────────────────────
st.subheader("🚨 Anomaly Alerts")

anomalies = load_anomalies(lookback)

if anomalies.empty:
    st.success("No anomalies detected in this period.")
else:
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Anomalies", len(anomalies))
    with col2:
        most_common = anomalies["Ticker"].value_counts().idxmax()
        st.metric("Most Flagged", most_common)
    with col3:
        latest = anomalies["Date"].max()
        st.metric("Latest Alert", str(latest))

    st.markdown("---")

    # Colour map per ticker
    ticker_colors = {
        "AAPL":  "#4da6ff",
        "TSLA":  "#ff4d4d",
        "GOOGL": "#ffd700",
        "MSFT":  "#00d084",
        "NVDA":  "#c084fc",
    }

    # Render each anomaly as a card
    for _, row in anomalies.iterrows():
        color = ticker_colors.get(row["Ticker"], "#888")
        score = row["Anomaly Score"]
        severity = "HIGH" if score < -0.10 else "MED" if score < -0.05 else "LOW"
        sev_color = "#ff4d4d" if severity == "HIGH" else "#ffd700" if severity == "MED" else "#00d084"

        st.markdown(f"""
<div style="
    background: #1c1e26;
    border-left: 4px solid {color};
    border-radius: 6px;
    padding: 10px 16px;
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
">
    <div>
        <span style="color:{color};font-weight:700;font-size:1rem">{row['Ticker']}</span>
        <span style="color:#888;margin-left:12px;font-size:0.85rem">{row['Date']}</span>
        <span style="color:#aaa;margin-left:12px;font-size:0.82rem">
            Top feature: <b style="color:#e0e0e0">{row['Top Feature']}</b>
        </span>
    </div>
    <div style="text-align:right">
        <span style="
            background:{sev_color}22;
            color:{sev_color};
            border:1px solid {sev_color}55;
            border-radius:4px;
            padding:2px 8px;
            font-size:0.78rem;
            font-weight:700;
        ">{severity}</span>
        <span style="color:#888;font-size:0.82rem;margin-left:10px">score: {score}</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Panel 4: Next-Day Predictions ─────────────────────────────
st.subheader(f"🤖 Next-Day Predictions — {date.today().isoformat()}")

preds = load_predictions()

if preds.empty:
    st.info("No predictions yet. Run the pipeline.")
else:
    cols = st.columns(len(preds))
    for col, (_, row) in zip(cols, preds.iterrows()):
        direction = row["Direction"]
        confidence = row["Confidence"]
        prob_up = row["P(Up)"]
        prob_down = row["P(Down)"]
        color = PALETTE["green"] if direction == "UP" else PALETTE["red"]
        arrow = "⬆" if direction == "UP" else "⬇"

        with col:
            st.markdown(f"""
<div style="
    background:{PALETTE['card']};
    border:1px solid {color}55;
    border-radius:10px;
    padding:18px 12px;
    text-align:center;
">
    <div style="font-size:1rem;font-weight:700;color:{PALETTE['text']}">{row['Ticker']}</div>
    <div style="font-size:2.2rem;color:{color};margin:8px 0">{arrow}</div>
    <div style="font-size:1.1rem;font-weight:700;color:{color}">{direction}</div>
    <div style="color:{PALETTE['muted']};font-size:0.82rem;margin-top:6px">
        Confidence: <b style="color:{PALETTE['text']}">{confidence:.1%}</b>
    </div>
    <div style="margin-top:8px">
        <span style="color:{PALETTE['green']};font-size:0.78rem">▲ {prob_up:.1%}</span>
        <span style="color:{PALETTE['muted']};font-size:0.78rem"> / </span>
        <span style="color:{PALETTE['red']};font-size:0.78rem">▼ {prob_down:.1%}</span>
    </div>
    <div style="color:{PALETTE['muted']};font-size:0.72rem;margin-top:6px">{row['Date']}</div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.caption("⚠️ For educational and portfolio purposes only. Not financial advice.")