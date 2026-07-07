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
import pandas as pd

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="FinSentinel",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom typography injection
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"], .stApp {
        font-family: 'Inter', sans-serif !important;
    }

    /* Slightly increased line spacing */
    p, span, div, label, li {
        line-height: 1.55 !important;
    }

    /* Page Title: bold */
    h1 {
        font-weight: 700 !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Section Titles: semi-bold */
    h2, h3, h4, h5, h6 {
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Metric Container Card Redesign */
    [data-testid="stMetric"] {
        background-color: #1c1e26 !important;
        border: 1px solid #2a2d3a !important;
        border-radius: 16px !important;
        padding: 20px 24px !important;
        min-height: 120px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.15), 0 2px 4px -1px rgba(0, 0, 0, 0.1) !important;
        transition: transform 0.2s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    [data-testid="stMetric"]:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 12px 20px -3px rgba(0, 0, 0, 0.4), 0 4px 8px -2px rgba(0, 0, 0, 0.15) !important;
        border-color: #4da6ff44 !important;
    }

    /* Metric Label: smaller and medium weight */
    [data-testid="stMetricLabel"] {
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        color: #888888 !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Metric Value: larger and bold */
    [data-testid="stMetricValue"] {
        font-size: 2.3rem !important;
        font-weight: 700 !important;
        color: #e0e0e0 !important;
        font-family: 'Inter', sans-serif !important;
        margin-top: 6px !important;
    }

    /* Labels: medium weight */
    label, .stWidgetLabel {
        font-weight: 500 !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Anomaly Alert Card Design */
    .anomaly-alert-card {
        background-color: #1c1e26 !important;
        border: 1px solid #2a2d3a !important;
        border-radius: 12px !important;
        padding: 14px 20px !important;
        margin-bottom: 12px !important;
        display: flex !important;
        justify-content: space-between !important;
        align-items: center !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
        transition: transform 0.2s cubic-bezier(0.4, 0, 0.2, 1), border-color 0.2s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    .anomaly-alert-card:hover {
        transform: translateX(4px) !important;
        border-color: #4da6ff44 !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3), 0 4px 6px -2px rgba(0, 0, 0, 0.1) !important;
    }

    /* Button Hover & Transitions */
    .stButton>button {
        transition: background-color 0.2s cubic-bezier(0.4, 0, 0.2, 1), border-color 0.2s cubic-bezier(0.4, 0, 0.2, 1), color 0.2s cubic-bezier(0.4, 0, 0.2, 1), transform 0.2s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .stButton>button:hover {
        border-color: #4da6ffdd !important;
        color: #4da6ffdd !important;
        background-color: #1c1e26 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15) !important;
    }
    .stButton>button:active {
        transform: translateY(0) !important;
    }

</style>
""", unsafe_allow_html=True)

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
    # SaaS-style Header
    st.markdown("""
    <div style="display: flex; flex-direction: column; gap: 4px; margin-bottom: 12px; padding-top: 8px;">
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="font-size: 1.5rem; font-weight: 800; letter-spacing: -0.03em;">📈 FinSentinel</span>
        </div>
        <span style="font-size: 0.82rem; color: #888888; font-weight: 500;">Sentiment-Augmented Financial Analytics</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<p style='font-size: 0.72rem; font-weight: 800; color: #888888; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 2px;'>Configuration</p>", unsafe_allow_html=True)
    ticker   = st.selectbox("Ticker", TICKERS, label_visibility="collapsed")
    
    st.markdown("<p style='font-size: 0.72rem; font-weight: 800; color: #888888; letter-spacing: 0.1em; text-transform: uppercase; margin-top: 14px; margin-bottom: 2px;'>Analysis Window</p>", unsafe_allow_html=True)
    lookback = st.slider("Lookback (days)", 30, 365, 90, label_visibility="collapsed")

    # Data Coverage
    st.markdown("<p style='font-size: 0.72rem; font-weight: 800; color: #888888; letter-spacing: 0.1em; text-transform: uppercase; margin-top: 18px; margin-bottom: 10px;'>Database Status</p>", unsafe_allow_html=True)
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

    st.markdown(f"""
    <div style="background-color: #1c1e26; border: 1px solid #2a2d3a; border-radius: 12px; padding: 14px 16px; display: flex; flex-direction: column; gap: 8px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.15); margin-bottom: 12px;">
        <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.8rem;">
            <span style="color: #888888; font-weight: 500;">Price Records</span>
            <span style="color: #e0e0e0; font-weight: 700; font-family: monospace;">{price_count:,}</span>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.8rem;">
            <span style="color: #888888; font-weight: 500;">News Articles</span>
            <span style="color: #e0e0e0; font-weight: 700; font-family: monospace;">{news_count:,}</span>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.8rem;">
            <span style="color: #888888; font-weight: 500;">Features Computed</span>
            <span style="color: #e0e0e0; font-weight: 700; font-family: monospace;">{feature_count:,}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("▶ Run Pipeline Now", width='stretch'):
        with st.spinner("Running pipeline..."):
            from pipeline.runner import run_pipeline
            run_pipeline(backfill_days=2)
        st.cache_data.clear()
        st.success("Pipeline complete!")
        st.rerun()

    st.markdown(f"""
    <div style="margin-top: 14px; padding-top: 12px; border-top: 1px solid #2a2d3a; display: flex; flex-direction: column; gap: 2px;">
        <span style="font-size: 0.72rem; color: #888888; font-weight: 500;">Last updated: {date.today().isoformat()}</span>
        <span style="font-size: 0.72rem; color: #888888; font-weight: 500;">Pipeline: daily at 4:30 PM ET</span>
    </div>
    """, unsafe_allow_html=True)

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

# ── Hero Header ───────────────────────────────────────────────
if not prices.empty:
    latest = prices["close"].iloc[-1]
    prev   = prices["close"].iloc[-2] if len(prices) > 1 else latest
    if pd.isna(latest) or pd.isna(prev):
        st.markdown(f"""
        <div style="margin-top: -1rem; margin-bottom: 1.5rem;">
            <h1 style="font-size: 2.8rem; font-weight: 800; margin: 0; letter-spacing: -0.03em;">{ticker}</h1>
            <p style="color: #888888; font-size: 0.9rem; margin-top: 6px; margin-bottom: 0;">
                No valid price records available.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        change_pct = ((latest - prev) / prev) * 100
        change_color = "#00d084" if change_pct >= 0 else "#ff4d4d"
        change_sign = "+" if change_pct >= 0 else ""
        
        st.markdown(f"""
        <div style="margin-top: -1rem; margin-bottom: 1.5rem;">
            <div style="display: flex; align-items: baseline; gap: 16px; flex-wrap: wrap;">
                <h1 style="font-size: 2.8rem; font-weight: 800; margin: 0; letter-spacing: -0.03em;">{ticker}</h1>
                <span style="font-size: 2rem; font-weight: 700; font-family: monospace; color: #787878;">${latest:.2f}</span>
                <span style="font-size: 1.05rem; font-weight: 700; color: {change_color}; background-color: {change_color}15; padding: 4px 10px; border-radius: 8px; border: 1px solid {change_color}33; display: inline-flex; align-items: center; gap: 4px;">
                    {change_sign}{change_pct:.2f}%
                </span>
            </div>
            <p style="color: #888888; font-size: 0.9rem; margin-top: 6px; margin-bottom: 0;">
                Institutional-Grade Analytics &bull; Showing last {lookback} trading days
            </p>
        </div>
        """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div style="margin-top: -1rem; margin-bottom: 1.5rem;">
        <h1 style="font-size: 2.8rem; font-weight: 800; margin: 0; letter-spacing: -0.03em;">{ticker}</h1>
        <p style="color: #888888; font-size: 0.9rem; margin-top: 6px; margin-bottom: 0;">
            No price records available. Please run pipeline first.
        </p>
    </div>
    """, unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    if not prices.empty:
        latest = prices["close"].iloc[-1]
        prev   = prices["close"].iloc[-2] if len(prices) > 1 else latest
        if pd.isna(latest) or pd.isna(prev):
            st.metric("Latest Close", "N/A", "N/A")
        else:
            st.metric("Latest Close", f"${latest:.2f}", f"{((latest-prev)/prev)*100:.2f}%")

with col2:
    if not features.empty:
        rsi = features["rsi_14"].iloc[-1]
        if pd.isna(rsi):
            st.metric("RSI (14)", "N/A", "Neutral")
        else:
            st.metric("RSI (14)", f"{rsi:.1f}", "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral")

with col3:
    preds = load_predictions()
    if not preds.empty:
        row = preds[preds["Ticker"] == ticker]
        if not row.empty:
            direction = row["Direction"].iloc[0]
            conf      = row["Confidence"].iloc[0]
            if pd.isna(conf):
                st.metric("Prediction", direction or "N/A", "N/A confidence")
            else:
                st.metric("Prediction", direction, f"{conf:.1%} confidence")

with col4:
    if not features.empty:
        vix = features["vix"].iloc[-1]
        if pd.isna(vix):
            st.metric("VIX", "N/A")
        else:
            st.metric("VIX", f"{vix:.2f}")

st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)

# ── Panel 1: Candlestick + Technicals ─────────────────────────
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.subheader(f"Price & Technicals")

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
            line=dict(color=PALETTE["blue"], width=1.5, dash="dot"),
            name="BB Upper",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=features.index, y=features["bb_lower"],
            line=dict(color=PALETTE["blue"], width=1.5, dash="dot"),
            fill="tonexty",
            fillcolor="rgba(77, 166, 255, 0.04)",
            name="BB Lower",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=features.index, y=features["bb_mid"],
            line=dict(color=PALETTE["muted"], width=1.2),
            name="BB Mid",
        ), row=1, col=1)

    # RSI
    if not features.empty and "rsi_14" in features.columns:
        fig.add_trace(go.Scatter(
            x=features.index, y=features["rsi_14"],
            line=dict(color=PALETTE["gold"], width=2),
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
            line=dict(color=PALETTE["blue"], width=2),
            name="MACD",
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=features.index, y=features["macd_signal"],
            line=dict(color=PALETTE["gold"], width=2),
            name="Signal",
        ), row=3, col=1)

    fig.update_layout(
        height=780,
        paper_bgcolor=PALETTE["bg"],
        plot_bgcolor=PALETTE["bg"],
        font_color=PALETTE["text"],
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="right",
            x=1,
            bgcolor="rgba(28, 30, 38, 0.7)",
            bordercolor="#2a2d3a",
            borderwidth=1,
            font=dict(color="#787878")
        ),
        margin=dict(l=50, r=40, t=110, b=40),
    )
    fig.update_xaxes(gridcolor="#1F232D", showgrid=True)
    fig.update_yaxes(gridcolor="#1F232D", showgrid=True)

    st.plotly_chart(fig, width='stretch')

st.markdown("<div style='margin-top: 3.5rem;'></div>", unsafe_allow_html=True)

# ── Panel 2: Sentiment Timeline ───────────────────────────────
st.subheader(f"News Sentiment vs. Price")

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
            line=dict(color=PALETTE["blue"], width=2.5),
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
            height=450,
            paper_bgcolor=PALETTE["bg"],
            plot_bgcolor=PALETTE["bg"],
            font_color=PALETTE["text"],
            margin=dict(l=50, r=40, t=60, b=40),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.05,
                xanchor="right",
                x=1,
                bgcolor="rgba(28, 30, 38, 0.7)",
                bordercolor="#2a2d3a",
                borderwidth=1,
                font=dict(color="#787878")
            ),
        )
        fig2.update_xaxes(gridcolor="#1F232D", showgrid=True)
        fig2.update_yaxes(
            title_text="Price ($)",
            gridcolor="#1F232D",
            secondary_y=False,
        )
        fig2.update_yaxes(
            title_text="Sentiment Score",
            range=[-1, 1],
            gridcolor="#1F232D",
            secondary_y=True,
        )

        st.plotly_chart(fig2, width='stretch')

        # Summary stats below chart
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Sentiment", f"{sent.mean():+.3f}")
        with col2:
            st.metric("Latest Sentiment", f"{sent.iloc[-1]:+.3f}")
        with col3:
            positive_days = (sent > 0).sum()
            st.metric("Positive Days", f"{positive_days}/{len(sent)}")

st.markdown("<div style='margin-top: 3.5rem;'></div>", unsafe_allow_html=True)

# ── Panel 3: Anomaly Alerts ───────────────────────────────────
st.subheader("Anomaly Alerts")

anomalies_all = load_anomalies(lookback)
anomalies = anomalies_all[anomalies_all["Ticker"] == ticker] if not anomalies_all.empty else anomalies_all

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
        if pd.isna(score):
            severity = "LOW"
            sev_color = "#00d084"
            score_str = "N/A"
        else:
            severity = "HIGH" if score < -0.10 else "MED" if score < -0.05 else "LOW"
            sev_color = "#ff4d4d" if severity == "HIGH" else "#ffd700" if severity == "MED" else "#00d084"
            score_str = f"{score:.4f}"

        st.markdown(f"""<div class="anomaly-alert-card" style="border-left: 4px solid {color} !important;">
<div>
<span style="color:{color};font-weight:700;font-size:1.05rem">{row['Ticker']}</span>
<span style="color:#888;margin-left:12px;font-size:0.85rem">{row['Date']}</span>
<span style="color:#888;margin-left:12px;font-size:0.82rem">Top feature: <code style="background-color:#2a2d3a;color:#e0e0e0;padding:2px 6px;border-radius:4px;font-family:monospace;font-size:0.8rem;">{row['Top Feature']}</code></span>
</div>
<div style="text-align:right">
<span style="background:{sev_color}15;color:{sev_color};border:1px solid {sev_color}33;border-radius:6px;padding:4px 10px;font-size:0.75rem;font-weight:700;display:inline-flex;align-items:center;gap:6px;letter-spacing:0.05em;">
<span style="width:6px;height:6px;background-color:{sev_color};border-radius:50%;"></span>
{severity}
</span>
<span style="color:#888;font-size:0.85rem;font-weight:600;font-family:monospace;margin-left:14px;">Score: {score_str}</span>
</div>
</div>""", unsafe_allow_html=True)

st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)

# ── Panel 4: Next-Day Predictions ─────────────────────────────
st.subheader(f"Next-Day Prediction — {date.today().isoformat()}")

preds = load_predictions()

if preds.empty:
    st.info("No predictions yet. Run the pipeline.")
else:
    preds_filtered = preds[preds["Ticker"] == ticker]
    if preds_filtered.empty:
        st.info(f"No prediction for {ticker} today.")
    else:
        row = preds_filtered.iloc[0]
        direction  = row["Direction"] or "N/A"
        confidence = row["Confidence"]
        prob_up    = row["P(Up)"]
        prob_down  = row["P(Down)"]
        color      = PALETTE["green"] if direction == "UP" else PALETTE["red"]
        arrow      = "⬆" if direction == "UP" else "⬇"

        # Bar width for prob visual
        up_pct   = int(prob_up * 100) if (prob_up is not None and not pd.isna(prob_up)) else 0
        down_pct = int(prob_down * 100) if (prob_down is not None and not pd.isna(prob_down)) else 0

        conf_str = f"{confidence:.1%}" if (confidence is not None and not pd.isna(confidence)) else "N/A"
        prob_up_str = f"{prob_up:.1%}" if (prob_up is not None and not pd.isna(prob_up)) else "N/A"
        prob_down_str = f"{prob_down:.1%}" if (prob_down is not None and not pd.isna(prob_down)) else "N/A"

        left, center, right = st.columns([1, 1.2, 1])
        center.markdown(f"""<div style="background:{PALETTE['card']};border:1px solid {color}66;border-radius:14px;padding:24px 28px;text-align:center;">
<div style="font-size:1rem;font-weight:800;color:#888;letter-spacing:2px;margin-bottom:12px">NEXT DAY &bull; {ticker}</div>
<div style="font-size:3.5rem;color:{color};line-height:1">{arrow}</div>
<div style="font-size:1.6rem;font-weight:800;color:{color};margin:6px 0">{direction}</div>
<div style="color:{PALETTE['muted']};font-size:0.85rem;margin-bottom:18px">Model confidence: <b style="color:{PALETTE['text']};font-size:1rem">{conf_str}</b></div>
<div style="background:#2a2d3a;border-radius:6px;overflow:hidden;height:8px;margin-bottom:6px">
<div style="width:{up_pct}%;height:100%;background:linear-gradient(90deg,{PALETTE['green']},{PALETTE['green']}99)"></div>
</div>
<div style="display:flex;justify-content:space-between;font-size:0.78rem">
<span style="color:{PALETTE['green']}">▲ UP {prob_up_str}</span>
<span style="color:{PALETTE['red']}">▼ DOWN {prob_down_str}</span>
</div>
</div>""", unsafe_allow_html=True)

st.markdown("<div style='margin-top: 3rem;'></div>", unsafe_allow_html=True)
st.caption("⚠️ For educational and portfolio purposes only. Not financial advice.")