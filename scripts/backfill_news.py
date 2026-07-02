import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import date, timedelta
from dotenv import load_dotenv
load_dotenv()

from db.session import init_db
from ingestion.news import ingest_news_sentiment

init_db()

# NewsAPI free tier: max 30 days back
start = date.today() - timedelta(days=29)
end = date.today()

print(f"Backfilling news {start} -> {end}")
results = ingest_news_sentiment(start=start, end=end)
print(results)