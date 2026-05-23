"""
One-time backfill of historical sentiment scores.
Fetches 6 months of news per ticker (1 API call each = 15 calls total),
groups by trading day, scores with FinBERT, appends to sentiment_history.csv.

Run once:
    python sentiment/backfill.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import csv
import time
import os
import requests
from pathlib import Path
from datetime import date, timedelta
from collections import defaultdict

import config
from data.universe import TICKERS
from sentiment.finbert import score_article, aggregate_scores
from sentiment.logger import LOG_PATH, FIELDNAMES, already_logged, _ensure_csv

FINNHUB_BASE = "https://finnhub.io/api/v1"


def _fetch_range(ticker: str, from_date: date, to_date: date) -> list[dict]:
    key = os.getenv("FINNHUB_API_KEY")
    if not key:
        raise ValueError("FINNHUB_API_KEY not set")
    params = {
        "symbol": ticker,
        "from": from_date.strftime("%Y-%m-%d"),
        "to": to_date.strftime("%Y-%m-%d"),
        "token": key,
    }
    resp = requests.get(f"{FINNHUB_BASE}/company-news", params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def _group_by_date(articles: list[dict]) -> dict[str, list[dict]]:
    """Group articles by calendar date string."""
    by_date: dict[str, list[dict]] = defaultdict(list)
    for a in articles:
        ts = a.get("datetime", 0)
        if ts:
            from datetime import datetime
            d = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
            by_date[d].append(a)
    return by_date


def _score_articles(articles: list[dict], max_articles: int = 20) -> float:
    inputs = []
    for a in articles[:max_articles]:
        title = a.get("headline", "")
        summary = a.get("summary", "")
        if title or summary:
            inputs.append({"title": title, "summary": summary})
    if not inputs:
        return 0.0
    scores = [score_article(x["title"], x["summary"]) for x in inputs]
    return aggregate_scores(scores)


def _daily_ranges(months_back: int) -> list[tuple[date, date]]:
    """One API call per day — guarantees full coverage regardless of article volume."""
    today = date.today()
    start = today - timedelta(days=months_back * 30)
    ranges = []
    current = start
    while current <= today:
        if current.weekday() < 5:  # weekdays only — no market on weekends
            ranges.append((current, current))
        current += timedelta(days=1)
    return ranges


def backfill(months_back: int = 6, delay: float = 1.5):
    """
    Fetch and score news for the past `months_back` months for all tickers.
    Fetches month by month to avoid Finnhub's 250-article-per-request cap.
    Skips dates already in the CSV.
    """
    _ensure_csv()
    weeks = _daily_ranges(months_back)
    n_calls = len(TICKERS) * len(weeks)
    print(f"Backfilling {months_back} months for {len(TICKERS)} tickers")
    print(f"API calls: {n_calls} (~{n_calls * delay / 60:.1f} min at {delay}s delay)\n")

    total_written = 0

    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)

        for ticker in TICKERS:
            ticker_written = 0
            print(f"{ticker}: ", end="", flush=True)

            for from_date, to_date in weeks:
                try:
                    articles = _fetch_range(ticker, from_date, to_date)
                except Exception as e:
                    print(f"[ERR {e}] ", end="", flush=True)
                    time.sleep(delay)
                    continue

                by_date = _group_by_date(articles)
                current = from_date
                while current <= to_date:
                    date_str = current.strftime("%Y-%m-%d")
                    if not already_logged(ticker, current) and date_str in by_date:
                        try:
                            score = _score_articles(by_date[date_str])
                            writer.writerow({
                                "date": date_str,
                                "ticker": ticker,
                                "sentiment_score": round(score, 6),
                                "timestamp": f"{date_str}T00:00:00",
                            })
                            f.flush()
                            ticker_written += 1
                            total_written += 1
                        except Exception as e:
                            print(f"\n  {ticker}/{date_str}: score error — {e}")
                    current += timedelta(days=1)

                print(".", end="", flush=True)
                time.sleep(delay)

            print(f" {ticker_written} days")

    print(f"\nDone. {total_written} new rows added to {LOG_PATH}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", type=int, default=18, help="How many months back to fetch")
    parser.add_argument("--delay", type=float, default=1.1, help="Seconds between API calls")
    args = parser.parse_args()
    backfill(months_back=args.months, delay=args.delay)
