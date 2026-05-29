import os
import time
import requests
from datetime import datetime, timedelta
import config  # loads .env from project root

FINNHUB_BASE = "https://finnhub.io/api/v1"


def _get_key(api_key: str | None) -> str:
    key = api_key or os.getenv("FINNHUB_API_KEY")
    if not key:
        raise ValueError("FINNHUB_API_KEY not set in environment")
    return key


def get_news(
    ticker: str,
    days_back: int = 1,
    api_key: str | None = None,
) -> list[dict]:
    key = _get_key(api_key)
    to_date = datetime.utcnow()
    from_date = to_date - timedelta(days=days_back)

    params = {
        "symbol": ticker,
        "from": from_date.strftime("%Y-%m-%d"),
        "to": to_date.strftime("%Y-%m-%d"),
        "token": key,
    }

    for attempt in range(3):
        try:
            resp = requests.get(f"{FINNHUB_BASE}/company-news", params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except (requests.exceptions.Timeout, requests.exceptions.HTTPError) as e:
            # Mask API key from error message before re-raising
            err_str = str(e)
            key = params.get("token", "")
            if key:
                err_str = err_str.replace(key, "***")
            if attempt < 2:
                time.sleep(3 * (attempt + 1))
            else:
                raise requests.exceptions.HTTPError(err_str) from None
    return []


def get_daily_sentiment_inputs(
    ticker: str,
    days_back: int = 1,
    api_key: str | None = None,
    max_articles: int = 20,
) -> list[dict]:
    """Returns list of {title, summary} dicts ready for FinBERT."""
    articles = get_news(ticker, days_back=days_back, api_key=api_key)
    result = []
    for article in articles[:max_articles]:
        title = article.get("headline", "")
        summary = article.get("summary", "")
        if title or summary:
            result.append({"title": title, "summary": summary})
    return result
