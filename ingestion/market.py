import os
from datetime import date, timedelta
from typing import Optional
import yfinance as yf
import pandas as pd
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from db.models import RawPrice
from db.session import get_session

TICKERS = os.getenv("TICKERS", "AAPL,TSLA,GOOGL,MSFT,NVDA").split(",")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _fetch_ticker(ticker: str, start: date, end: date) -> pd.DataFrame:
    import yfinance as yf
    t = yf.Ticker(ticker, session=None)
    df = t.history(
        start=start.isoformat(),
        end=end.isoformat(),
        auto_adjust=True,
        proxy=None,
    )
    return df


def ingest_market_data(
    tickers: list = TICKERS,
    start: Optional[date] = None,
    end: Optional[date] = None,
) -> dict:
    end = end or date.today()
    start = start or (end - timedelta(days=1))
    results = {}

    for ticker in tickers:
        try:
            df = _fetch_ticker(ticker, start, end)
            if df.empty:
                logger.warning(f"{ticker}: no data returned")
                results[ticker] = 0
                continue

            df = df.reset_index()
            # Normalize column names
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]

            rows = []
            for _, row in df.iterrows():
                row_date = pd.Timestamp(row["date"]).normalize().date()
                rows.append(RawPrice(
                    ticker=ticker,
                    date=row_date,
                    open=float(row["open"]) if pd.notna(row["open"]) else None,
                    high=float(row["high"]) if pd.notna(row["high"]) else None,
                    low=float(row["low"]) if pd.notna(row["low"]) else None,
                    close=float(row["close"]) if pd.notna(row["close"]) else None,
                    volume=float(row["volume"]) if pd.notna(row["volume"]) else None,
                    adj_close=float(row["close"]) if pd.notna(row["close"]) else None,
                ))

            with get_session() as session:
                existing_dates = {
                    r.date for r in session.query(RawPrice.date)
                    .filter(RawPrice.ticker == ticker,
                            RawPrice.date >= start,
                            RawPrice.date <= end)
                    .all()
                }
                new_rows = [r for r in rows if r.date not in existing_dates]
                session.bulk_save_objects(new_rows)

            results[ticker] = len(new_rows)
            logger.info(f"{ticker}: inserted {len(new_rows)} rows")

        except Exception as e:
            logger.error(f"{ticker}: failed — {e}")
            results[ticker] = -1

    return results

def get_latest_prices(tickers: list = TICKERS, lookback_days: int = 252) -> pd.DataFrame:
    from datetime import date, timedelta
    from sqlalchemy import select

    start = date.today() - timedelta(days=lookback_days)

    with get_session() as session:
        rows = session.execute(
            select(RawPrice).where(
                RawPrice.ticker.in_(tickers),
                RawPrice.date >= start,
            ).order_by(RawPrice.ticker, RawPrice.date)
        ).scalars().all()

        records = [{
            "ticker": r.ticker, "date": r.date,
            "open": r.open, "high": r.high,
            "low": r.low, "close": r.close,
            "volume": r.volume,
        } for r in rows]

    df = pd.DataFrame(records)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index(["ticker", "date"]).sort_index()