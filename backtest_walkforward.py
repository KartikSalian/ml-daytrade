"""
Walk-forward validation to test overfitting.

Splits 2 years of data into 6 windows:
  - Train on window N
  - Test on window N+1 (unseen)
  - Roll forward

If results are consistent across all windows = generalises well.
If results are good in some windows, bad in others = overfitting.

    python backtest_walkforward.py
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
from features.engineering import build_full_dataset, FEATURE_COLS
from models import lgbm_model, cnn_lstm
from models.ensemble import train as train_ensemble, build_meta_features, _get_cnn_output_indices
from data.earnings import get_earnings_dates
from data.universe import TICKERS
from sentiment.logger import load_history

import vectorbt as vbt

SAVE_DIR     = Path("D:/ml-daytrade/models/saved")
N_WINDOWS    = 6
MIN_CONFIDENCE = 0.40

# Bull/bear class weights
BULL_LGBM_W = {0: 0.5, 1: 1.0, 2: 2.0}
BEAR_LGBM_W = {0: 2.0, 1: 1.0, 2: 0.5}
BULL_CNN_W  = [0.5, 1.0, 2.0]
BEAR_CNN_W  = [2.0, 1.0, 0.5]


def label_regime(vix: float, sp500_ret_20d: float) -> str:
    if vix >= 30 or vix > 28:
        return "BEAR"
    elif vix > 20 and sp500_ret_20d < -0.05:
        return "BEAR"
    elif vix > 20 and sp500_ret_20d < 0:
        return "BEAR"
    elif sp500_ret_20d > 0:
        return "BULL"
    return "BEAR"


def run_window(train_df, test_df, window_num, model_type):
    """Train fresh model on train_df, evaluate on test_df."""
    lgbm_w = BULL_LGBM_W if model_type == "BULL" else BEAR_LGBM_W
    cnn_w  = BULL_CNN_W  if model_type == "BULL" else BEAR_CNN_W

    cnn_path      = SAVE_DIR / f"wf_{model_type.lower()}_w{window_num}.pt"
    ensemble_path = SAVE_DIR / f"wf_{model_type.lower()}_ens_w{window_num}.pkl"

    lgbm = lgbm_model.train(train_df, class_weight=lgbm_w)
    cnn  = cnn_lstm.train(train_df, save_path=cnn_path, class_weights=cnn_w)
    meta, lgbm, cnn = train_ensemble(train_df, save_path=ensemble_path)

    # Evaluate on test
    results = []
    for ticker in test_df["ticker"].unique():
        tdf = test_df[test_df["ticker"] == ticker].copy()
        if len(tdf) < cnn_lstm.SEQ_LEN + 5:
            continue
        try:
            X_meta, _ = build_meta_features(tdf, lgbm, cnn)
            probs      = meta.predict_proba(X_meta)
            signals    = (meta.predict(X_meta) - 1).astype(int)
            seq_idx    = _get_cnn_output_indices(tdf)

            # Confidence filter
            for i in range(len(signals)):
                if float(probs[i].max()) < MIN_CONFIDENCE:
                    signals[i] = 0
                # Bear model: block BUY; Bull model: block SELL
                if model_type == "BULL" and signals[i] == -1:
                    signals[i] = 0
                elif model_type == "BEAR" and signals[i] == 1:
                    signals[i] = 0

            price = tdf["Close"].iloc[seq_idx[:len(signals)]]
            sig   = pd.Series(signals, index=price.index)

            pf = vbt.Portfolio.from_signals(
                price, sig == 1, sig == -1,
                init_cash=100_000, fees=0.001, slippage=0.001, freq="1h",
            )
            stats = pf.stats()
            results.append({
                "total_return": stats.get("Total Return [%]", 0),
                "sharpe":       stats.get("Sharpe Ratio", 0),
                "max_drawdown": stats.get("Max Drawdown [%]", 0),
                "win_rate":     stats.get("Win Rate [%]", 0),
                "trades":       stats.get("Total Trades", 0),
                "bh_return":    (price.iloc[-1] / price.iloc[0] - 1) * 100,
            })
        except Exception:
            pass

    if not results:
        return None

    rdf = pd.DataFrame(results)
    return {
        "return":    rdf["total_return"].mean(),
        "sharpe":    rdf["sharpe"].mean(),
        "drawdown":  rdf["max_drawdown"].mean(),
        "win_rate":  rdf["win_rate"].mean(),
        "trades":    rdf["trades"].sum(),
        "bh_return": rdf["bh_return"].mean(),
        "alpha":     rdf["total_return"].mean() - rdf["bh_return"].mean(),
    }


def main():
    print("=" * 60)
    print("Walk-Forward Validation — Overfitting Test")
    print("=" * 60)

    print("\nLoading data...")
    sentiment_history = load_history()
    earnings_dates    = get_earnings_dates(TICKERS)
    pooled = build_dataset(use_cache=True)
    df     = build_full_dataset(pooled, sentiment_history=sentiment_history,
                                earnings_dates=earnings_dates)
    print(f"  Total: {len(df):,} rows | {df.index.min().date()} to {df.index.max().date()}")

    # Split into N_WINDOWS equal chunks
    tickers   = df["ticker"].unique()
    # Use single-ticker timeline for splitting
    timeline  = df[df["ticker"] == tickers[0]].index
    window_sz = len(timeline) // (N_WINDOWS + 1)

    print(f"\n  {N_WINDOWS} windows of ~{window_sz} rows each per ticker")
    print(f"  Train: {window_sz * N_WINDOWS // (N_WINDOWS)} rows | Test: {window_sz} rows per window\n")

    bull_results = []
    bear_results = []

    for w in range(N_WINDOWS):
        train_end = timeline[window_sz * (w + 1)]
        test_end  = timeline[min(window_sz * (w + 2) - 1, len(timeline) - 1)]
        test_start = train_end

        train_df = df[df.index <= train_end]
        test_df  = df[(df.index > test_start) & (df.index <= test_end)]

        if len(train_df) < 5000 or len(test_df) < 500:
            continue

        date_range = f"{test_start.date()} to {test_end.date()}"
        print(f"Window {w+1}/{N_WINDOWS} — Test: {date_range}")

        print(f"  Training BULL model...")
        bull_r = run_window(train_df, test_df, w+1, "BULL")
        if bull_r:
            bull_r["window"] = w + 1
            bull_r["period"] = date_range
            bull_results.append(bull_r)
            print(f"  BULL: return={bull_r['return']:.2f}% sharpe={bull_r['sharpe']:.2f} trades={bull_r['trades']:.0f} alpha={bull_r['alpha']:.2f}%")

        print(f"  Training BEAR model...")
        bear_r = run_window(train_df, test_df, w+1, "BEAR")
        if bear_r:
            bear_r["window"] = w + 1
            bear_r["period"] = date_range
            bear_results.append(bear_r)
            print(f"  BEAR: return={bear_r['return']:.2f}% sharpe={bear_r['sharpe']:.2f} trades={bear_r['trades']:.0f} alpha={bear_r['alpha']:.2f}%")

        print()

    # Summary
    for label, results in [("BULL", bull_results), ("BEAR", bear_results)]:
        if not results:
            continue
        rdf = pd.DataFrame(results)
        print(f"\n=== {label} MODEL WALK-FORWARD SUMMARY ===")
        print(rdf[["period", "return", "sharpe", "win_rate", "trades", "alpha"]].round(2).to_string(index=False))
        print(f"\n  Mean return : {rdf['return'].mean():.2f}%")
        print(f"  Std  return : {rdf['return'].std():.2f}%  ← low std = consistent = not overfitting")
        print(f"  Mean sharpe : {rdf['sharpe'].mean():.2f}")
        print(f"  Mean alpha  : {rdf['alpha'].mean():.2f}%")
        positive = (rdf['return'] > 0).sum()
        print(f"  Positive windows: {positive}/{len(rdf)}")

    # Cleanup temp model files
    for f in SAVE_DIR.glob("wf_*.pt"):
        f.unlink()
    for f in SAVE_DIR.glob("wf_*.pkl"):
        f.unlink()


if __name__ == "__main__":
    main()
