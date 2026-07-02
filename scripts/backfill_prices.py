import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import date, timedelta
from dotenv import load_dotenv
load_dotenv()

from db.session import init_db
from ingestion.market import ingest_market_data

init_db()

start = date.today() - timedelta(days=365)
end = date.today()

print(f"Backfilling {start} -> {end}")
results = ingest_market_data(start=start, end=end)
print(results)