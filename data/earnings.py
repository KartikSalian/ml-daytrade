"""
Fetches earnings calendar from Finnhub and computes days-to-earnings feature.
Uses a local cache (data/earnings_cache.pkl) to avoid re-fetching.
"""
import os
import pickle
import requests
from datetime import date, datetime, timedelta
from pathlib import Path

import config

CACHE_PATH = Path(__file__).parent / "earnings_cache.pkl"
FINNHUB_BASE = "https://finnhub.io/api/v1"


def _fetch_earnings_dates(ticker: str, from_date: str, to_date: str) -> list[str]:
    key = os.getenv("FINNHUB_API_KEY")
    if not key:
        return []
    params = {"symbol": ticker, "from": from_date, "to": to_date, "token": key}
    try:
        resp = requests.get(f"{FINNHUB_BASE}/calendar/earnings", params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("earningsCalendar", [])
        return sorted({e["date"] for e in data if e.get("date")})
    except Exception:
        return []


def load_earnings_cache() -> dict[str, list[str]]:
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)
    return {}


def save_earnings_cache(cache: dict[str, list[str]]) -> None:
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)


def get_earnings_dates(
    tickers: list[str],
    years_back: int = 2,
    refresh: bool = False,
) -> dict[str, list[str]]:
    """
    Returns {ticker: [date_str, ...]} sorted ascending.
    Fetches from Finnhub if not cached or refresh=True.
    """
    cache = load_earnings_cache()
    today = date.today()
    from_str = (today - timedelta(days=years_back * 365)).strftime("%Y-%m-%d")
    to_str = (today + timedelta(days=90)).strftime("%Y-%m-%d")  # include upcoming

    for ticker in tickers:
        if ticker not in cache or refresh:
            dates = _fetch_earnings_dates(ticker, from_str, to_str)
            cache[ticker] = dates
            print(f"  {ticker}: {len(dates)} earnings dates fetched")

    save_earnings_cache(cache)
    return cache


def days_until_next_earnings(ref_date: date, earnings_dates: list[str]) -> int:
    """Days from ref_date to the next upcoming earnings. Returns 90 if none found."""
    for d_str in earnings_dates:
        d = datetime.strptime(d_str, "%Y-%m-%d").date()
        if d >= ref_date:
            return (d - ref_date).days
    return 90
