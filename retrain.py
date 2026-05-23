"""
Retrain all three models with real FinBERT sentiment data.

Steps:
  1. Run backfill first:  python sentiment/backfill.py
  2. Then retrain:        python retrain.py

The script merges sentiment_history.csv into the training data so
sentiment_score reflects real FinBERT scores instead of 0.0.
"""
import sys
sys.path.insert(0, "D:/ml-daytrade")

import config
from data.fetcher import build_dataset
from data.earnings import get_earnings_dates
from data.universe import TICKERS
from features.engineering import build_full_dataset, FEATURE_COLS
from sentiment.logger import load_history
from models import lgbm_model, cnn_lstm
from models.ensemble import train as train_ensemble, build_meta_features


def main():
    print("Loading sentiment history...")
    sentiment_history = load_history()
    n_tickers = len(sentiment_history)
    n_rows = sum(len(v) for v in sentiment_history.values())
    print(f"  {n_tickers} tickers, {n_rows} daily scores loaded")

    if n_rows == 0:
        print("No sentiment history found. Run sentiment/backfill.py first.")
        return

    print("\nFetching earnings calendar...")
    earnings_dates = get_earnings_dates(TICKERS)

    print("\nBuilding dataset...")
    pooled = build_dataset()

    print("Applying feature engineering + sentiment + earnings merge...")
    df = build_full_dataset(pooled, sentiment_history=sentiment_history, earnings_dates=earnings_dates)
    print(f"  Dataset: {len(df):,} rows")

    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    n_features = len(feature_cols)
    print(f"  Features: {n_features} ({feature_cols})")

    # Check how much sentiment coverage we have
    nonzero = (df["sentiment_score"] != 0.0).mean()
    print(f"  Sentiment coverage: {nonzero:.1%} of rows have real scores")

    # Train/test split (chronological)
    split = int(len(df) * 0.8)
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]
    print(f"\nTrain: {len(train_df):,} rows | Test: {len(test_df):,} rows")

    # --- LightGBM ---
    print("\n[1/3] Training LightGBM...")
    lgbm = lgbm_model.train(train_df)
    lgbm_model.save(lgbm)
    print("  Saved.")

    # --- CNN-LSTM ---
    print("\n[2/3] Training CNN-LSTM...")
    cnn = cnn_lstm.train(train_df)
    print("  Saved.")

    # --- Ensemble ---
    print("\n[3/3] Training ensemble meta-learner...")
    meta, lgbm, cnn = train_ensemble(train_df)
    print("  Saved.")

    # --- OOS accuracy ---
    print("\nEvaluating OOS accuracy on held-out test set...")
    from sklearn.metrics import accuracy_score
    X_meta, y_test = build_meta_features(test_df, lgbm, cnn)
    preds = meta.predict(X_meta)
    acc = accuracy_score(y_test, preds)
    print(f"  Ensemble OOS accuracy: {acc:.1%}")

    print("\nRetrain complete. All models saved to models/saved/")


if __name__ == "__main__":
    main()
