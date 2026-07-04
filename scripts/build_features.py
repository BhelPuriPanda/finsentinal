import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from db.session import init_db
from features.engineer import build_features, load_features_df

init_db()

print("Building features...")
results = build_features(lookback_days=400)
print("Inserted:", results)

print("\nVerifying...")
df = load_features_df(lookback_days=400)
print(f"Total rows: {len(df)}")
print(f"Tickers: {df['ticker'].unique().tolist()}")
print(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")
print(f"Columns: {df.columns.tolist()}")
print("\nSample row (AAPL latest):")
aapl = df[df['ticker'] == 'AAPL'].tail(1)
print(aapl[['ticker','date','close','rsi_14','macd','bb_pct','sentiment_compound_mean','vix','next_day_up']].to_string())