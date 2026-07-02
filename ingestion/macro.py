import os
from datetime import date, timedelta
from typing import Optional
import pandas as pd
from fredapi import Fred
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from db.models import RawMacro
from db.session import get_session

FRED_SERIES = {
    "CPIAUCSL": "CPI (Urban Consumers, All Items)",
    "FEDFUNDS": "Federal Funds Effective Rate",
    "VIXCLS":   "CBOE Volatility Index (VIX)",
}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _fetch_series(fred: Fred, series_id: str, start: date, end: date) -> pd.Series:
    return fred.get_series(series_id, observation_start=start, observation_end=end)


def ingest_macro_data(
    series: dict = FRED_SERIES,
    start: Optional[date] = None,
    end: Optional[date] = None,
) -> dict:
    api_key = os.environ["FRED_API_KEY"]
    fred = Fred(api_key=api_key)

    end = end or date.today()
    start = start or (end - timedelta(days=30))
    results = {}

    for series_id in series:
        try:
            s = _fetch_series(fred, series_id, start, end)
            if s.empty:
                logger.warning(f"{series_id}: no data returned")
                results[series_id] = 0
                continue

            rows = [
                RawMacro(
                    series_id=series_id,
                    date=idx.date() if hasattr(idx, "date") else idx,
                    value=float(val) if pd.notna(val) else None,
                )
                for idx, val in s.items()
            ]

            with get_session() as session:
                existing_dates = {
                    r.date for r in session.query(RawMacro.date)
                    .filter(RawMacro.series_id == series_id,
                            RawMacro.date >= start,
                            RawMacro.date <= end)
                    .all()
                }
                new_rows = [r for r in rows if r.date not in existing_dates]
                session.bulk_save_objects(new_rows)

            results[series_id] = len(new_rows)
            logger.info(f"{series_id}: inserted {len(new_rows)} rows")

        except Exception as e:
            logger.error(f"{series_id}: macro ingestion failed — {e}")
            results[series_id] = -1

    return results


def get_macro_snapshot(lookback_days: int = 400) -> pd.DataFrame:
    """
    Pull macro data from DB and return a daily forward-filled DataFrame.
    Index: date. Columns: cpi, fed_funds_rate, vix.
    """
    from sqlalchemy import select
    from datetime import date, timedelta

    start = date.today() - timedelta(days=lookback_days)

    with get_session() as session:
        rows = session.execute(
            select(RawMacro).where(RawMacro.date >= start)
        ).scalars().all()

    if not rows:
        return pd.DataFrame()

    records = [{"series_id": r.series_id, "date": r.date, "value": r.value} for r in rows]
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])

    wide = df.pivot(index="date", columns="series_id", values="value")
    wide = wide.rename(columns={
        "CPIAUCSL": "cpi",
        "FEDFUNDS": "fed_funds_rate",
        "VIXCLS":   "vix",
    })

    daily_idx = pd.date_range(wide.index.min(), date.today(), freq="D")
    wide = wide.reindex(daily_idx).ffill()

    return wide