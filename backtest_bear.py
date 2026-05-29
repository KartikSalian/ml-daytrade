"""
Bear market stress test for US model.
Tests Feb-Apr 2025 (tariff shock) and Feb-Mar 2026 bear periods.

    python backtest_bear.py
"""
import sys
sys.path.insert(0, "D:/ml-daytrade")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
import config

from data.fetcher import build_dataset
from features.engineering import (
    build_full_dataset, add_bear_features, FEATURE_COLS, BEAR_FEATURE_COLS
)
from models import lgbm_model, cnn_lstm
from models.ensemble import load_ensemble, build_meta_features, _get_cnn_output_indices

import vectorbt as vbt

MIN_CONFIDENCE = 0.35  # bear model uses lower threshold
VIX_HALT       = 30.0

BEAR_WINDOWS = {
    "Feb-Apr 2025 (Tariff Shock)": ("2025-02-01", "2025-04-30"),
    "Feb-Mar 2026":                ("2026-02-01", "2026-03-31"),
    "Full Bear Combined":          [("2025-02-01", "2025-04-30"), ("2026-02-01", "2026-03-31")],
}


def apply_filters(signals, probs, ticker_df, seq_idx):
    filtered   = signals.copy()
    vix_series = ticker_df["VIX_close"]
    sp500_ret  = (1 + ticker_df["SP500_ret"]).rolling(120, min_periods=60).apply(
        lambda x: x.prod(), raw=True
    ) - 1

    aligned_vix    = vix_series.iloc[seq_idx[:len(signals)]]
    aligned_sp_ret = sp500_ret.iloc[seq_idx[:len(signals)]]

    for i in range(len(filtered)):
        if float(probs[i].max()) < MIN_CONFIDENCE:
            filtered[i] = 0
            continue

        vix = float(aligned_vix.iloc[i])
        ret = float(aligned_sp_ret.iloc[i])

        if vix >= VIX_HALT:
            filtered[i] = 0
        elif vix > 20 and ret < -0.05:   # BEAR — block BUY
            if filtered[i] == 1:
                filtered[i] = 0
        elif vix > 20 and ret < 0:       # CHOPPY — allow both, keep as-is
            pass
        else:                            # BULL — block SELL
            if filtered[i] == -1:
                filtered[i] = 0

    return filtered


def run_on_window(df, lgbm, cnn, meta, start, end, label):
    mask     = (df.index >= start) & (df.index <= end)
    test_df  = df[mask]
    if len(test_df) < cnn_lstm.SEQ_LEN + 10:
        print(f"   Not enough data for {label}")
        return None

    results = []
    for ticker in test_df["ticker"].unique():
        tdf = test_df[test_df["ticker"] == ticker].copy()
        if len(tdf) < cnn_lstm.SEQ_LEN + 5:
            continue
        try:
            X_meta, _ = build_meta_features(tdf, lgbm, cnn, feature_cols=BEAR_FEATURE_COLS)
            probs      = meta.predict_proba(X_meta)
            signals    = (meta.predict(X_meta) - 1).astype(int)
            seq_idx    = _get_cnn_output_indices(tdf)
            signals    = apply_filters(signals, probs, tdf, seq_idx)
            price      = tdf["Close"].iloc[seq_idx[:len(signals)]]

            sig = pd.Series(signals, index=price.index)
            pf  = vbt.Portfolio.from_signals(
                price, sig == 1, sig == -1,
                init_cash=100_000, fees=0.001, slippage=0.001, freq="1h",
            )
            stats = pf.stats()
            results.append({
                "ticker":       ticker,
                "total_return": stats.get("Total Return [%]", 0),
                "sharpe":       stats.get("Sharpe Ratio", 0),
                "max_drawdown": stats.get("Max Drawdown [%]", 0),
                "win_rate":     stats.get("Win Rate [%]", 0),
                "total_trades": stats.get("Total Trades", 0),
                "bh_return":    (price.iloc[-1] / price.iloc[0] - 1) * 100,
            })
        except Exception as e:
            pass

    if not results:
        return None

    rdf = pd.DataFrame(results)
    return {
        "label":         label,
        "mean_return":   rdf["total_return"].mean(),
        "mean_sharpe":   rdf["sharpe"].mean(),
        "mean_drawdown": rdf["max_drawdown"].mean(),
        "win_rate":      rdf["win_rate"].mean(),
        "total_trades":  rdf["total_trades"].sum(),
        "bh_return":     rdf["bh_return"].mean(),
        "alpha":         rdf["total_return"].mean() - rdf["bh_return"].mean(),
    }


def main():
    print("=" * 60)
    print("US Model — Bear Market Stress Test")
    print("=" * 60)

    print("\n1. Loading data...")
    pooled = build_dataset(use_cache=True)
    df     = build_full_dataset(pooled)
    frames = []
    for ticker, group in df.groupby("ticker"):
        frames.append(add_bear_features(group.copy()))
    df = pd.concat(frames).sort_index().dropna()
    print(f"   Total rows: {len(df):,}")
    print(f"   Date range: {df.index.min().date()} to {df.index.max().date()}")

    print("\n2. Loading models (BEAR specialist)...")
    SAVE_DIR   = Path("D:/ml-daytrade/models/saved")
    n_features = len(BEAR_FEATURE_COLS)
    lgbm = lgbm_model.load(SAVE_DIR / "lgbm_bear.pkl")
    cnn  = cnn_lstm.load(n_features, SAVE_DIR / "cnn_lstm_bear.pt")
    meta = load_ensemble(SAVE_DIR / "ensemble_bear.pkl")

    print("\n3. Running bear window tests...")
    all_results = []

    for label, window in BEAR_WINDOWS.items():
        print(f"\n--- {label} ---")
        if isinstance(window, list):
            # Combined windows
            combined = []
            for s, e in window:
                r = run_on_window(df, lgbm, cnn, meta, s, e, label)
                if r:
                    combined.append(r)
            if combined:
                result = {
                    "label":         label,
                    "mean_return":   np.mean([r["mean_return"] for r in combined]),
                    "mean_sharpe":   np.mean([r["mean_sharpe"] for r in combined]),
                    "mean_drawdown": np.mean([r["mean_drawdown"] for r in combined]),
                    "win_rate":      np.mean([r["win_rate"] for r in combined]),
                    "total_trades":  sum(r["total_trades"] for r in combined),
                    "bh_return":     np.mean([r["bh_return"] for r in combined]),
                    "alpha":         np.mean([r["alpha"] for r in combined]),
                }
                all_results.append(result)
        else:
            s, e = window
            result = run_on_window(df, lgbm, cnn, meta, s, e, label)
            if result:
                all_results.append(result)

    print("\n=== BEAR MARKET RESULTS ===")
    print(f"{'Period':30s} {'Return':>8} {'Sharpe':>8} {'Drawdown':>10} {'WinRate':>9} {'Trades':>7} {'B&H':>8} {'Alpha':>8}")
    print("-" * 95)
    for r in all_results:
        print(f"{r['label']:30s} {r['mean_return']:>7.2f}% {r['mean_sharpe']:>8.2f} "
              f"{r['mean_drawdown']:>9.2f}% {r['win_rate']:>8.2f}% {r['total_trades']:>7.0f} "
              f"{r['bh_return']:>7.2f}% {r['alpha']:>7.2f}%")


if __name__ == "__main__":
    main()
