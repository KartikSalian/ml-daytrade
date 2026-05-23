import os
import pickle
from pathlib import Path

import pandas as pd
import yfinance as yf
from newsapi import NewsApiClient

import config  # loads .env from project root
from data.universe import TICKERS, MACRO_TICKERS, INTERVAL, PERIOD

CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.pkl"


def _load_cache(key: str) -> pd.DataFrame | None:
    path = _cache_path(key)
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def _save_cache(key: str, df: pd.DataFrame) -> None:
    with open(_cache_path(key), "wb") as f:
        pickle.dump(df, f)


def get_ohlcv(
    ticker: str,
    period: str = PERIOD,
    interval: str = INTERVAL,
    use_cache: bool = True,
) -> pd.DataFrame:
    cache_key = f"{ticker}_{interval}_{period}"
    if use_cache:
        cached = _load_cache(cache_key)
        if cached is not None:
            return cached

    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    df["ticker"] = ticker

    if use_cache:
        _save_cache(cache_key, df)
    return df


def get_macro_data(
    period: str = PERIOD,
    interval: str = INTERVAL,
    use_cache: bool = True,
) -> dict[str, pd.DataFrame]:
    result = {}
    for name, symbol in MACRO_TICKERS.items():
        cache_key = f"macro_{name}_{interval}_{period}"
        if use_cache:
            cached = _load_cache(cache_key)
            if cached is not None:
                result[name] = cached
                continue

        df = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.dropna(inplace=True)

        if use_cache:
            _save_cache(cache_key, df)
        result[name] = df
    return result


def merge_macro_features(ohlcv: pd.DataFrame, macro: dict[str, pd.DataFrame]) -> pd.DataFrame:
    df = ohlcv.copy()

    vix = macro.get("VIX")
    if vix is not None:
        df["VIX_close"] = vix["Close"].reindex(df.index, method="ffill")

    sp500 = macro.get("SP500")
    if sp500 is not None:
        df["SP500_ret"] = sp500["Close"].pct_change().reindex(df.index, method="ffill")

    tech = macro.get("TECH_ETF")
    if tech is not None:
        df["XLK_ret"] = tech["Close"].pct_change().reindex(df.index, method="ffill")

    for key, col in [("ENERGY_ETF", "XLE_ret"), ("FINANCE_ETF", "XLF_ret"), ("HEALTH_ETF", "XLV_ret")]:
        etf = macro.get(key)
        if etf is not None:
            df[col] = etf["Close"].pct_change().reindex(df.index, method="ffill")

    df.dropna(inplace=True)
    return df


def build_dataset(
    tickers: list[str] = TICKERS,
    period: str = PERIOD,
    interval: str = INTERVAL,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Download and merge all tickers into one pooled DataFrame."""
    macro = get_macro_data(period=period, interval=interval, use_cache=use_cache)
    frames = []
    for ticker in tickers:
        print(f"  Fetching {ticker}...")
        df = get_ohlcv(ticker, period=period, interval=interval, use_cache=use_cache)
        df = merge_macro_features(df, macro)
        frames.append(df)

    combined = pd.concat(frames)
    combined.sort_index(inplace=True)
    return combined


def get_news(ticker: str, api_key: str | None = None) -> list[dict]:
    key = api_key or os.getenv("NEWSAPI_KEY")
    if not key:
        raise ValueError("NEWSAPI_KEY not set in environment or passed as argument")
    client = NewsApiClient(api_key=key)
    response = client.get_everything(q=ticker, language="en", sort_by="publishedAt", page_size=20)
    return response.get("articles", [])
