import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import date, timedelta
from dotenv import load_dotenv
load_dotenv()

from db.session import init_db
from ingestion.macro import ingest_macro_data

init_db()

# Go back 2 years to get enough CPI + FEDFUNDS history
start = date.today() - timedelta(days=730)
end = date.today()

print(f"Backfilling macro {start} → {end}")
results = ingest_macro_data(start=start, end=end)
print(results)