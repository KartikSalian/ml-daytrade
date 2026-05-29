"""
Retrain BEAR model only — bull model untouched.

Adds 6 macro bear features: VIX momentum, VIX z-score, SP500 realized vol,
HYG (credit stress), TLT (flight to safety), DXY (risk-off dollar).

    python retrain_bear.py
"""
import sys
sys.path.insert(0, "D:/ml-daytrade")

from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report

import config
from data.fetcher import build_dataset
from data.earnings import get_earnings_dates
from data.universe import TICKERS
from features.engineering import (
    build_full_dataset, add_bear_features, BEAR_FEATURE_COLS
)
from sentiment.logger import load_history
from models import lgbm_model, cnn_lstm
from models.ensemble import train as train_ensemble, build_meta_features

SAVE_DIR = Path("D:/ml-daytrade/models/saved")

# Bear mindset: reward SELL, penalise false BUYs
BEAR_LGBM_W = {0: 2.0, 1: 1.0, 2: 0.5}
BEAR_CNN_W  = [2.0, 1.0, 0.5]


def prepare_bear_df(df):
    """Add the 6 bear features on top of the standard feature set."""
    import pandas as pd
    frames = []
    for ticker, group in df.groupby("ticker"):
        g = group.copy()
        g = add_bear_features(g)
        frames.append(g)
    result = pd.concat(frames).sort_index()
    result.dropna(inplace=True)
    return result


def main():
    print("=" * 60)
    print("Bear Model Retrain — with 6 macro bear features")
    print("=" * 60)

    n_bear_features = len(BEAR_FEATURE_COLS)
    print(f"\nBear features: {n_bear_features}")
    print(f"New features: VIX_momentum, VIX_zscore, SP500_realized_vol, HYG_ret, TLT_ret, DXY_ret")

    print("\nLoading sentiment history...")
    sentiment_history = load_history()
    n_rows = sum(len(v) for v in sentiment_history.values())
    print(f"  {n_rows} daily scores loaded")

    print("\nFetching earnings calendar...")
    earnings_dates = get_earnings_dates(TICKERS)

    print("\nBuilding dataset (fetching HYG, TLT, DXY too)...")
    pooled = build_dataset(use_cache=False)  # force refresh to get new macro tickers
    df = build_full_dataset(
        pooled,
        sentiment_history=sentiment_history,
        earnings_dates=earnings_dates,
    )
    print(f"  Base dataset: {len(df):,} rows")

    print("\nAdding bear macro features...")
    df = prepare_bear_df(df)
    print(f"  After bear features: {len(df):,} rows")

    available = [c for c in BEAR_FEATURE_COLS if c in df.columns]
    print(f"  Available bear features: {len(available)}/{n_bear_features}")
    missing = [c for c in BEAR_FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"  Missing: {missing}")

    split = int(len(df) * 0.8)
    train_df = df.iloc[:split]
    test_df  = df.iloc[split:]
    print(f"\n  Train: {len(train_df):,} | Test: {len(test_df):,}")

    lgbm_path     = SAVE_DIR / "lgbm_bear.pkl"
    cnn_path      = SAVE_DIR / "cnn_lstm_bear.pt"
    ensemble_path = SAVE_DIR / "ensemble_bear.pkl"

    print("\n[1/3] Training Bear LightGBM (29 features)...")
    lgbm = lgbm_model.train(train_df, class_weight=BEAR_LGBM_W,
                             feature_cols=BEAR_FEATURE_COLS)
    lgbm_model.save(lgbm, path=lgbm_path)

    print("\n[2/3] Training Bear CNN-LSTM (29 features)...")
    cnn = cnn_lstm.train(train_df, save_path=cnn_path,
                         class_weights=BEAR_CNN_W,
                         feature_cols=BEAR_FEATURE_COLS)

    print("\n[3/3] Training Bear Ensemble (29 features)...")
    meta, lgbm, cnn = train_ensemble(
        train_df,
        save_path=ensemble_path,
        feature_cols=BEAR_FEATURE_COLS,
        lgbm_model_obj=lgbm,
        cnn_model_obj=cnn,
    )

    print("\n[OOS] Evaluating bear model...")
    X_meta, y_test = build_meta_features(test_df, lgbm, cnn,
                                          feature_cols=BEAR_FEATURE_COLS)
    preds = meta.predict(X_meta)
    acc = accuracy_score(y_test, preds)
    print(f"  OOS accuracy: {acc:.1%}")
    print(classification_report(
        y_test, preds,
        target_names=["SELL", "HOLD", "BUY"],
        zero_division=0,
    ))

    print("\nBear model retrain complete.")
    print(f"  Saved: lgbm_bear.pkl, cnn_lstm_bear.pt, ensemble_bear.pkl")
    print(f"  Features: {len(available)}")
    print("Bull model unchanged.")


if __name__ == "__main__":
    main()
