import pandas as pd
import numpy as np


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # Trend
    df["SMA_20"] = close.rolling(20).mean()
    df["SMA_50"] = close.rolling(50).mean()
    df["EMA_12"] = close.ewm(span=12, adjust=False).mean()
    df["EMA_26"] = close.ewm(span=26, adjust=False).mean()
    df["SMA_cross"] = (df["SMA_20"] - df["SMA_50"]) / df["SMA_50"]

    # MACD
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["BB_upper"] = bb_mid + 2 * bb_std
    df["BB_lower"] = bb_mid - 2 * bb_std
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / bb_mid
    df["BB_position"] = (close - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"])

    # ATR
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()
    df["ATR_pct"] = df["ATR"] / close

    # Volume
    vol_ma = volume.rolling(20).mean()
    df["Volume_ratio"] = volume / vol_ma

    # Momentum
    df["ROC_5"] = close.pct_change(5)
    df["ROC_10"] = close.pct_change(10)
    df["ROC_20"] = close.pct_change(20)

    # Stochastic %K/%D
    low_14 = low.rolling(14).min()
    high_14 = high.rolling(14).max()
    df["Stoch_K"] = 100 * (close - low_14) / (high_14 - low_14)
    df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()

    # 52-week high/low proximity (hourly: 52w ≈ 1638 bars)
    w52 = 52 * 5 * 63 // 10  # ~1638 hourly bars
    high_52w = high.rolling(w52, min_periods=100).max()
    low_52w = low.rolling(w52, min_periods=100).min()
    df["pct_from_52w_high"] = (close - high_52w) / high_52w
    df["pct_from_52w_low"] = (close - low_52w) / low_52w

    df.dropna(inplace=True)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Time-of-day features based on US Eastern market hours."""
    # Convert index to Eastern time for market session logic
    idx = df.index
    if hasattr(idx, "tz") and idx.tz is not None:
        eastern = idx.tz_convert("US/Eastern")
    else:
        eastern = idx

    hour = eastern.hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)   # cyclical encoding
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["is_open"]      = ((hour == 9) | (hour == 10)).astype(int)   # 9:30-10:30 volatile open
    df["is_close"]     = ((hour == 15) | (hour == 16)).astype(int)  # 3-4PM institutional close
    df["is_dead_zone"] = ((hour >= 12) & (hour <= 13)).astype(int)  # 12-2PM low volume
    df["day_of_week"]  = pd.to_datetime(df.index.date).map(
        lambda d: d.weekday()
    ).values  # 0=Mon, 4=Fri
    return df


def add_target(
    df: pd.DataFrame,
    horizon: int = 4,       # 4 hours forward by default for hourly bars
    threshold: float = 0.003,  # tighter threshold for intraday (0.3%)
) -> pd.DataFrame:
    future_return = df["Close"].shift(-horizon) / df["Close"] - 1
    df["target"] = np.where(future_return > threshold, 1,
                   np.where(future_return < -threshold, -1, 0))
    df["future_return"] = future_return
    df.dropna(inplace=True)
    return df


def build_features_for_ticker(
    df: pd.DataFrame,
    sentiment_history: dict[str, dict[str, float]] | None = None,
    ticker: str | None = None,
    earnings_dates: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Run the full feature pipeline on a single ticker's DataFrame."""
    from data.earnings import days_until_next_earnings
    df = df.copy()
    df = add_technical_features(df)
    df = add_time_features(df)

    if sentiment_history and ticker and ticker in sentiment_history:
        scores = sentiment_history[ticker]
        df["sentiment_score"] = [
            scores.get(d.strftime("%Y-%m-%d"), 0.0) for d in df.index.date
        ]
    elif "sentiment_score" not in df.columns:
        df["sentiment_score"] = 0.0

    if earnings_dates and ticker and ticker in earnings_dates:
        edates = earnings_dates[ticker]
        df["days_to_earnings"] = [
            days_until_next_earnings(d, edates) for d in df.index.date
        ]
    else:
        df["days_to_earnings"] = 90

    df = add_target(df)
    return df


def build_full_dataset(
    pooled: pd.DataFrame,
    sentiment_history: dict[str, dict[str, float]] | None = None,
    earnings_dates: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Apply feature engineering per ticker then re-pool."""
    frames = []
    for ticker, group in pooled.groupby("ticker"):
        try:
            featured = build_features_for_ticker(group, sentiment_history, ticker, earnings_dates)
            frames.append(featured)
        except Exception as e:
            print(f"  Skipping {ticker}: {e}")
    return pd.concat(frames).sort_index()


FEATURE_COLS = [
    "SMA_cross", "MACD", "MACD_hist", "RSI",
    "BB_width", "BB_position", "ATR_pct",
    "Volume_ratio", "ROC_5", "ROC_10", "ROC_20",
    "Stoch_K", "Stoch_D",
    "VIX_close", "SP500_ret", "XLK_ret",
    "sentiment_score",
    "hour_sin", "hour_cos", "is_open", "is_close", "is_dead_zone", "day_of_week",
]

# Extended feature set for next retrain (after alignment fix)
FEATURE_COLS_V2 = FEATURE_COLS + [
    "pct_from_52w_high", "pct_from_52w_low",
    "XLE_ret", "XLF_ret", "XLV_ret",
    "days_to_earnings",
]
