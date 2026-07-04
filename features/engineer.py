import os
from datetime import date, timedelta
import pandas as pd
from loguru import logger
from sqlalchemy import select

from db.models import Features, RawNews, RawMacro
from db.session import get_session
from ingestion.market import get_latest_prices
from features.indicators import compute_all

TICKERS = os.getenv("TICKERS", "AAPL,TSLA,GOOGL,MSFT,NVDA").split(",")

FEATURE_COLS = [
    "close", "volume", "returns_1d", "returns_5d",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_lower", "bb_mid", "bb_pct", "atr_14",
    "sentiment_compound_mean", "sentiment_compound_std",
    "sentiment_pos_mean", "sentiment_neg_mean", "news_count",
    "cpi", "fed_funds_rate", "vix",
]


def _get_sentiment(tickers: list, lookback_days: int) -> pd.DataFrame:
    start = date.today() - timedelta(days=lookback_days)
    with get_session() as session:
        rows = session.execute(
            select(RawNews).where(
                RawNews.ticker.in_(tickers),
                RawNews.date >= start,
            )
        ).scalars().all()

        records = [{
            "ticker": r.ticker,
            "date": r.date,
            "compound": r.vader_compound,
            "pos": r.vader_pos,
            "neg": r.vader_neg,
        } for r in rows]

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    agg = df.groupby(["ticker", "date"]).agg(
        sentiment_compound_mean=("compound", "mean"),
        sentiment_compound_std=("compound", "std"),
        sentiment_pos_mean=("pos", "mean"),
        sentiment_neg_mean=("neg", "mean"),
        news_count=("compound", "count"),
    )
    return agg


def _get_macro(lookback_days: int) -> pd.DataFrame:
    start = date.today() - timedelta(days=lookback_days)
    with get_session() as session:
        rows = session.execute(
            select(RawMacro).where(RawMacro.date >= start)
        ).scalars().all()

        records = [{"series_id": r.series_id, "date": r.date, "value": r.value} for r in rows]

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])

    wide = df.pivot(index="date", columns="series_id", values="value")
    wide = wide.rename(columns={
        "CPIAUCSL": "cpi",
        "FEDFUNDS": "fed_funds_rate",
        "VIXCLS":   "vix",
    })

    wide = wide.reindex(columns=["cpi", "fed_funds_rate", "vix"])

    daily_idx = pd.date_range(wide.index.min(), date.today(), freq="D")
    wide = wide.reindex(daily_idx).ffill()
    return wide


def build_features(
    tickers: list = TICKERS,
    lookback_days: int = 400,
) -> dict:
    price_data = get_latest_prices(tickers, lookback_days=lookback_days)
    sentiment  = _get_sentiment(tickers, lookback_days=lookback_days)
    macro      = _get_macro(lookback_days=lookback_days)
    results    = {}

    for ticker in tickers:
        try:
            if ticker not in price_data.index.get_level_values("ticker"):
                logger.warning(f"{ticker}: no price data, skipping")
                results[ticker] = 0
                continue

            prices = price_data.loc[ticker].copy()
            prices = compute_all(prices)

            # Merge sentiment
            if not sentiment.empty and ticker in sentiment.index.get_level_values("ticker"):
                sent_t = sentiment.loc[ticker]
                prices = prices.join(sent_t, how="left")
            else:
                for col in ["sentiment_compound_mean", "sentiment_compound_std",
                            "sentiment_pos_mean", "sentiment_neg_mean", "news_count"]:
                    prices[col] = None

            # Merge macro (forward-filled)
            if not macro.empty:
                prices = prices.join(macro, how="left")
                prices[["cpi", "fed_funds_rate", "vix"]] = (
                    prices[["cpi", "fed_funds_rate", "vix"]].ffill()
                )
            else:
                prices[["cpi", "fed_funds_rate", "vix"]] = None

            prices = prices.reset_index()
            prices["date"] = pd.to_datetime(prices["date"]).dt.date

            rows = []
            for _, row in prices.iterrows():
                def g(col, is_int=False):
                    v = row.get(col)
                    if v is None or (isinstance(v, float) and pd.isna(v)):
                        return None
                    return int(v) if is_int else float(v)

                rows.append(Features(
                    ticker=ticker,
                    date=row["date"],
                    close=g("close"),
                    volume=g("volume"),
                    returns_1d=g("returns_1d"),
                    returns_5d=g("returns_5d"),
                    rsi_14=g("rsi_14"),
                    macd=g("macd"),
                    macd_signal=g("macd_signal"),
                    macd_hist=g("macd_hist"),
                    bb_upper=g("bb_upper"),
                    bb_mid=g("bb_mid"),
                    bb_lower=g("bb_lower"),
                    bb_pct=g("bb_pct"),
                    atr_14=g("atr_14"),
                    sentiment_compound_mean=g("sentiment_compound_mean"),
                    sentiment_compound_std=g("sentiment_compound_std"),
                    sentiment_pos_mean=g("sentiment_pos_mean"),
                    sentiment_neg_mean=g("sentiment_neg_mean"),
                    news_count=g("news_count", is_int=True),
                    cpi=g("cpi"),
                    fed_funds_rate=g("fed_funds_rate"),
                    vix=g("vix"),
                    next_day_up=bool(row["next_day_up"]) if row.get("next_day_up") is not None and pd.notna(row.get("next_day_up")) else None,
                ))

            with get_session() as session:
                existing = {
                    r.date for r in session.query(Features.date)
                    .filter(Features.ticker == ticker)
                    .all()
                }
                new_rows = [r for r in rows if r.date not in existing]
                session.bulk_save_objects(new_rows)

            results[ticker] = len(new_rows)
            logger.info(f"{ticker}: inserted {len(new_rows)} feature rows")

        except Exception as e:
            logger.error(f"{ticker}: feature engineering failed — {e}")
            results[ticker] = -1

    return results


def load_features_df(tickers: list = TICKERS, lookback_days: int = 252) -> pd.DataFrame:
    start = date.today() - timedelta(days=lookback_days)
    with get_session() as session:
        rows = session.execute(
            select(Features).where(
                Features.ticker.in_(tickers),
                Features.date >= start,
            ).order_by(Features.ticker, Features.date)
        ).scalars().all()

        records = [{
            "ticker": r.ticker,
            "date": r.date,
            **{col: getattr(r, col) for col in FEATURE_COLS},
            "next_day_up": r.next_day_up,
        } for r in rows]

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df
