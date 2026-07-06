import os
from datetime import date, timedelta
from loguru import logger
from dotenv import load_dotenv
load_dotenv()


def run_pipeline(backfill_days: int = 2):
    from db.session import init_db
    from ingestion.market import ingest_market_data
    from ingestion.news import ingest_news_sentiment
    from ingestion.macro import ingest_macro_data
    from features.engineer import build_features, load_features_df
    from models.anomaly import train_isolation_forest, run_anomaly_detection
    from models.predictor import train_xgboost, run_predictions

    logger.info(f"=== Pipeline run started (backfill_days={backfill_days}) ===")

    init_db()
    end   = date.today()
    start = end - timedelta(days=backfill_days)

    logger.info("Step 1: Market data...")
    ingest_market_data(start=start, end=end)

    logger.info("Step 2: News sentiment...")
    ingest_news_sentiment(start=start, end=end)

    logger.info("Step 3: Macro data...")
    ingest_macro_data(start=start, end=end)

    logger.info("Step 4: Feature engineering...")
    build_features(lookback_days=400)

    logger.info("Step 5: Training models...")
    df = load_features_df(lookback_days=400)
    if df.empty:
        logger.error("No feature data — aborting")
        return

    train_isolation_forest(df)
    train_xgboost(df)

    logger.info("Step 6: Anomaly detection...")
    run_anomaly_detection(df)

    logger.info("Step 7: Predictions...")
    run_predictions(df)

    logger.info("=== Pipeline complete ===")


def start_scheduler():
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger

    hour   = int(os.getenv("SCHEDULE_HOUR",   "16"))
    minute = int(os.getenv("SCHEDULE_MINUTE", "30"))

    scheduler = BlockingScheduler(timezone="America/New_York")
    scheduler.add_job(
        run_pipeline,
        trigger=CronTrigger(
            hour=hour,
            minute=minute,
            day_of_week="mon-fri",
        ),
        id="daily_pipeline",
        replace_existing=True,
    )
    logger.info(f"Scheduler started — runs at {hour:02d}:{minute:02d} ET Mon–Fri")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-now",  action="store_true")
    parser.add_argument("--backfill", action="store_true")
    parser.add_argument("--days",     type=int, default=365)
    args = parser.parse_args()

    if args.run_now or args.backfill:
        days = args.days if args.backfill else 2
        run_pipeline(backfill_days=days)
    else:
        start_scheduler()