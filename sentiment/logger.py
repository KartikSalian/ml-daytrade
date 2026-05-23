"""
Daily sentiment logger.
Fetches news for all 15 tickers, scores with FinBERT, appends to CSV.
Run once per day at market close (9PM BST / 4PM EST).
"""
import csv
import time
from pathlib import Path
from datetime import datetime, date

import config
from data.universe import TICKERS
from sentiment.pipeline import compute_ticker_sentiment

LOG_PATH = Path(__file__).parent.parent / "data" / "sentiment_history.csv"
FIELDNAMES = ["date", "ticker", "sentiment_score", "timestamp"]


def _ensure_csv():
    if not LOG_PATH.exists():
        with open(LOG_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
        print(f"Created sentiment log: {LOG_PATH}")


def already_logged(ticker: str, log_date: date) -> bool:
    if not LOG_PATH.exists():
        return False
    with open(LOG_PATH, "r") as f:
        for row in csv.DictReader(f):
            if row["ticker"] == ticker and row["date"] == str(log_date):
                return True
    return False


def log_today(tickers: list[str] = TICKERS, days_back: int = 1, delay: float = 1.0):
    _ensure_csv()
    today = date.today()
    print(f"Logging sentiment for {today}...")

    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        for ticker in tickers:
            if already_logged(ticker, today):
                print(f"  {ticker}: already logged today, skipping")
                continue
            try:
                score = compute_ticker_sentiment(ticker, days_back=days_back)
                writer.writerow({
                    "date": str(today),
                    "ticker": ticker,
                    "sentiment_score": round(score, 6),
                    "timestamp": datetime.utcnow().isoformat(),
                })
                f.flush()
                print(f"  {ticker}: {score:+.4f}")
            except Exception as e:
                print(f"  {ticker}: error — {e}")
            time.sleep(delay)

    print(f"Done. Log: {LOG_PATH}")


def load_history() -> dict[str, dict[str, float]]:
    """
    Returns {ticker: {date_str: score}} for merging with price data.
    """
    history: dict[str, dict[str, float]] = {}
    if not LOG_PATH.exists():
        return history
    with open(LOG_PATH, "r") as f:
        for row in csv.DictReader(f):
            ticker = row["ticker"]
            if ticker not in history:
                history[ticker] = {}
            history[ticker][row["date"]] = float(row["sentiment_score"])
    return history


def get_score(ticker: str, log_date: date) -> float:
    """Get the logged sentiment score for a ticker on a given date. Returns 0.0 if missing."""
    history = load_history()
    return history.get(ticker, {}).get(str(log_date), 0.0)


if __name__ == "__main__":
    log_today()
