import os
from datetime import date, timedelta
from typing import Optional
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from db.models import RawNews
from db.session import get_session

TICKERS = os.getenv("TICKERS", "AAPL,TSLA,GOOGL,MSFT,NVDA").split(",")

TICKER_QUERIES = {
    "AAPL":  "Apple stock",
    "TSLA":  "Tesla stock",
    "GOOGL": "Google Alphabet stock",
    "MSFT":  "Microsoft stock",
    "NVDA":  "NVIDIA stock",
}

_analyzer = SentimentIntensityAnalyzer()


def _score(text: str) -> dict:
    return _analyzer.polarity_scores(text)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _fetch_articles(client: NewsApiClient, query: str, from_date: date, to_date: date) -> list:
    resp = client.get_everything(
        q=query,
        from_param=from_date.isoformat(),
        to=to_date.isoformat(),
        language="en",
        sort_by="publishedAt",
        page_size=100,
    )
    return resp.get("articles", [])


def ingest_news_sentiment(
    tickers: list = TICKERS,
    start: Optional[date] = None,
    end: Optional[date] = None,
) -> dict:
    api_key = os.environ["NEWS_API_KEY"]
    client = NewsApiClient(api_key=api_key)

    end = end or date.today()
    start = start or (end - timedelta(days=1))
    results = {}

    for ticker in tickers:
        query = TICKER_QUERIES.get(ticker, ticker)
        try:
            articles = _fetch_articles(client, query, start, end)
            if not articles:
                logger.warning(f"{ticker}: no articles found")
                results[ticker] = 0
                continue

            rows = []
            for art in articles:
                title = art.get("title") or ""
                desc  = art.get("description") or ""
                text  = f"{title} {desc}".strip()
                if not text:
                    continue

                scores = _score(text)

                pub = art.get("publishedAt", "")[:10]
                try:
                    article_date = date.fromisoformat(pub)
                except ValueError:
                    article_date = end

                rows.append(RawNews(
                    ticker=ticker,
                    date=article_date,
                    headline=title[:500],
                    source=art.get("source", {}).get("name", "")[:100],
                    url=(art.get("url") or "")[:1000],
                    vader_compound=scores["compound"],
                    vader_pos=scores["pos"],
                    vader_neg=scores["neg"],
                    vader_neu=scores["neu"],
                ))

            with get_session() as session:
                session.bulk_save_objects(rows)

            results[ticker] = len(rows)
            logger.info(f"{ticker}: inserted {len(rows)} news rows")

        except Exception as e:
            logger.error(f"{ticker}: news ingestion failed — {e}")
            results[ticker] = -1

    return results