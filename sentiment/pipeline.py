import time
import pandas as pd
from datetime import datetime, date

from sentiment.finbert import score_article, aggregate_scores
from sentiment.news import get_daily_sentiment_inputs


def compute_ticker_sentiment(
    ticker: str,
    days_back: int = 1,
    api_key: str | None = None,
) -> float:
    """Returns a single sentiment score [-1, +1] for a ticker."""
    articles = get_daily_sentiment_inputs(ticker, days_back=days_back, api_key=api_key)
    if not articles:
        return 0.0
    scores = [score_article(a["title"], a["summary"]) for a in articles]
    return aggregate_scores(scores)


def compute_sentiment_batch(
    tickers: list[str],
    days_back: int = 1,
    api_key: str | None = None,
    delay: float = 1.0,  # respect Finnhub rate limit
) -> dict[str, float]:
    """Returns {ticker: sentiment_score} for a list of tickers."""
    results = {}
    for ticker in tickers:
        try:
            score = compute_ticker_sentiment(ticker, days_back=days_back, api_key=api_key)
            results[ticker] = score
            print(f"  {ticker}: {score:+.3f}")
        except Exception as e:
            print(f"  {ticker}: error — {e}")
            results[ticker] = 0.0
        time.sleep(delay)
    return results


def attach_sentiment_to_df(
    df: pd.DataFrame,
    sentiment_map: dict[str, float],
) -> pd.DataFrame:
    """
    Adds sentiment_score column to a pooled multi-ticker DataFrame.
    sentiment_map: {ticker: score}
    """
    df = df.copy()
    df["sentiment_score"] = df["ticker"].map(sentiment_map).fillna(0.0)
    return df
