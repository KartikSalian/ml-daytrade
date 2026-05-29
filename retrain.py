"""
Retrain — Bull and Bear models trained on SAME data with different class weights.

Bull model: favours BUY signals  → class_weight {SELL:0.5, HOLD:1.0, BUY:2.0}
Bear model: favours SELL signals → class_weight {SELL:2.0, HOLD:1.0, BUY:0.5}

Both models see the full 2-year dataset so they understand all market conditions.
The regime filter in runner.py picks which model to use at inference time.

Steps:
  1. python sentiment/backfill.py   (if not done)
  2. python retrain.py
"""
import sys
sys.path.insert(0, "D:/ml-daytrade")

from pathlib import Path
from sklearn.metrics import accuracy_score

import config
from data.fetcher import build_dataset
from data.earnings import get_earnings_dates
from data.universe import TICKERS
from features.engineering import build_full_dataset, FEATURE_COLS
from sentiment.logger import load_history
from models import lgbm_model, cnn_lstm
from models.ensemble import train as train_ensemble, build_meta_features

SAVE_DIR = Path("D:/ml-daytrade/models/saved")

# Bull mindset: reward BUY signals, penalise false SELLs
BULL_LGBM_WEIGHTS = {0: 0.5, 1: 1.0, 2: 2.0}   # {SELL, HOLD, BUY}
BULL_CNN_WEIGHTS  = [0.5, 1.0, 2.0]

# Bear mindset: reward SELL signals, penalise false BUYs
BEAR_LGBM_WEIGHTS = {0: 2.0, 1: 1.0, 2: 0.5}
BEAR_CNN_WEIGHTS  = [2.0, 1.0, 0.5]


def train_set(train_df, label, lgbm_w, cnn_w):
    print(f"\n{'='*50}")
    print(f"Training {label} model set ({len(train_df):,} rows, same data different mindset)")

    lgbm_path     = SAVE_DIR / f"lgbm_{label}.pkl"
    cnn_path      = SAVE_DIR / f"cnn_lstm_{label}.pt"
    ensemble_path = SAVE_DIR / f"ensemble_{label}.pkl"

    print(f"\n[{label}] LightGBM (class_weight={lgbm_w})...")
    lgbm = lgbm_model.train(train_df, class_weight=lgbm_w)
    lgbm_model.save(lgbm, path=lgbm_path)

    print(f"\n[{label}] CNN-LSTM (class_weights={cnn_w})...")
    cnn = cnn_lstm.train(train_df, save_path=cnn_path, class_weights=cnn_w)

    print(f"\n[{label}] Ensemble...")
    meta, lgbm, cnn = train_ensemble(train_df, save_path=ensemble_path)

    return lgbm, cnn, meta


def main():
    print("=" * 60)
    print("Retrain — Bull + Bear Models (same data, different mindset)")
    print("=" * 60)

    print("\nLoading sentiment history...")
    sentiment_history = load_history()
    n_rows = sum(len(v) for v in sentiment_history.values())
    print(f"  {n_rows} daily scores loaded")

    print("\nFetching earnings calendar...")
    earnings_dates = get_earnings_dates(TICKERS)

    print("\nBuilding dataset...")
    pooled = build_dataset()
    df = build_full_dataset(
        pooled,
        sentiment_history=sentiment_history,
        earnings_dates=earnings_dates,
    )
    print(f"  Dataset: {len(df):,} rows")
    print(f"  Features: {len([c for c in FEATURE_COLS if c in df.columns])}")

    nonzero = (df["sentiment_score"] != 0.0).mean()
    print(f"  Sentiment coverage: {nonzero:.1%}")

    # Chronological split — same for both models
    split = int(len(df) * 0.8)
    train_df = df.iloc[:split]
    test_df  = df.iloc[split:]
    print(f"\n  Train: {len(train_df):,} | Test: {len(test_df):,}")

    # Train BULL model (BUY-biased)
    bull_lgbm, bull_cnn, bull_meta = train_set(
        train_df, "bull", BULL_LGBM_WEIGHTS, BULL_CNN_WEIGHTS
    )

    # Train BEAR model (SELL-biased)
    bear_lgbm, bear_cnn, bear_meta = train_set(
        train_df, "bear", BEAR_LGBM_WEIGHTS, BEAR_CNN_WEIGHTS
    )

    # OOS evaluation
    print("\n" + "=" * 50)
    print("OOS Accuracy on same test set:")
    for label, lgbm, cnn, meta in [
        ("BULL model", bull_lgbm, bull_cnn, bull_meta),
        ("BEAR model", bear_lgbm, bear_cnn, bear_meta),
    ]:
        try:
            X_meta, y_test = build_meta_features(test_df, lgbm, cnn)
            preds = meta.predict(X_meta)
            acc = accuracy_score(y_test, preds)
            print(f"  {label}: {acc:.1%}")
        except Exception as e:
            print(f"  {label} eval failed: {e}")

    print("\nRetrain complete.")
    print("  models/saved/lgbm_bull.pkl + cnn_lstm_bull.pt + ensemble_bull.pkl")
    print("  models/saved/lgbm_bear.pkl + cnn_lstm_bear.pt + ensemble_bear.pkl")


if __name__ == "__main__":
    main()
