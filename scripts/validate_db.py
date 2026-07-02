import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import date, timedelta
from dotenv import load_dotenv
load_dotenv()

from db.session import init_db, get_session
from db.models import RawPrice, RawNews, RawMacro

init_db()

TICKERS = ["AAPL", "TSLA", "GOOGL", "MSFT", "NVDA"]
MACRO_SERIES = ["CPIAUCSL", "FEDFUNDS", "VIXCLS"]

print("=" * 55)
print("FINSENTINEL — Week 1 DB Validation")
print("=" * 55)

with get_session() as session:

    # ── raw_prices ────────────────────────────────────────────
    print("\n[ raw_prices ]")
    total_prices = session.query(RawPrice).count()
    print(f"  Total rows   : {total_prices}")
    for ticker in TICKERS:
        count = session.query(RawPrice).filter(RawPrice.ticker == ticker).count()
        latest = (
            session.query(RawPrice)
            .filter(RawPrice.ticker == ticker)
            .order_by(RawPrice.date.desc())
            .first()
        )
        oldest = (
            session.query(RawPrice)
            .filter(RawPrice.ticker == ticker)
            .order_by(RawPrice.date.asc())
            .first()
        )
        if oldest and latest:
            print(f"  {ticker:5s}  rows={count:4d}  range={oldest.date} -> {latest.date}  latest_close=${latest.close:.2f}")
        else:
            print(f"  {ticker:5s}  rows={count:4d}  range=N/A -> N/A  latest_close=N/A")

    # ── raw_news ──────────────────────────────────────────────
    print("\n[ raw_news ]")
    total_news = session.query(RawNews).count()
    print(f"  Total rows   : {total_news}")
    for ticker in TICKERS:
        count = session.query(RawNews).filter(RawNews.ticker == ticker).count()
        avg_compound = (
            session.query(RawNews.vader_compound)
            .filter(RawNews.ticker == ticker)
            .all()
        )
        avg = sum(r[0] for r in avg_compound if r[0] is not None) / max(len(avg_compound), 1)
        print(f"  {ticker:5s}  rows={count:4d}  avg_sentiment={avg:+.3f}")

    # ── raw_macro ─────────────────────────────────────────────
    print("\n[ raw_macro ]")
    total_macro = session.query(RawMacro).count()
    print(f"  Total rows   : {total_macro}")
    for series in MACRO_SERIES:
        count = session.query(RawMacro).filter(RawMacro.series_id == series).count()
        latest = (
            session.query(RawMacro)
            .filter(RawMacro.series_id == series)
            .order_by(RawMacro.date.desc())
            .first()
        )
        if latest:
            print(f"  {series:10s}  rows={count:4d}  latest={latest.date}  value={latest.value}")
        else:
            print(f"  {series:10s}  rows={count:4d}  latest=N/A  value=N/A")

    # ── Assertions ────────────────────────────────────────────
    print("\n[ Assertions ]")
    checks = [
        (total_prices >= 1000,       f"raw_prices has >= 1000 rows (got {total_prices})"),
        (total_news >= 400,          f"raw_news has >= 400 rows (got {total_news})"),
        (total_macro >= 50,          f"raw_macro has >= 50 rows (got {total_macro})"),
        (all(
            session.query(RawPrice).filter(RawPrice.ticker == t).count() >= 200
            for t in TICKERS
        ),                           "All 5 tickers have >= 200 price rows"),
        (all(
            session.query(RawNews).filter(RawNews.ticker == t).count() >= 50
            for t in TICKERS
        ),                           "All 5 tickers have >= 50 news rows"),
        (all(
            session.query(RawMacro).filter(RawMacro.series_id == s).count() >= 10
            for s in MACRO_SERIES
        ),                           "All 3 macro series have >= 10 rows"),
    ]

    all_passed = True
    for passed, label in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {label}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 55)
    print("  ALL CHECKS PASSED " if all_passed else "  SOME CHECKS FAILED ")
    print("=" * 55)