import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

from features.engineering import FEATURE_COLS

SAVE_DIR = Path(__file__).parent / "saved"
SAVE_DIR.mkdir(exist_ok=True)
MODEL_PATH = SAVE_DIR / "lgbm_model.pkl"


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].copy()
    y = df["target"].copy()
    # LightGBM needs 0,1,2 for multiclass
    y = y + 1  # -1→0, 0→1, 1→2
    return X, y


def train(df: pd.DataFrame, n_splits: int = 5) -> lgb.LGBMClassifier:
    X, y = prepare_xy(df)

    params = dict(
        objective="multiclass",
        num_class=3,
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=63,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    tscv = TimeSeriesSplit(n_splits=n_splits)
    val_accs = []

    model = lgb.LGBMClassifier(**params)

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(False)],
        )

        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        val_accs.append(acc)
        print(f"  Fold {fold+1}: accuracy={acc:.4f}")

    print(f"\nMean CV accuracy: {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}")
    print("\nFinal fold classification report:")
    print(classification_report(y_val, preds, target_names=["SELL", "HOLD", "BUY"]))

    return model


def get_feature_importance(model: lgb.LGBMClassifier, feature_names: list[str]) -> pd.DataFrame:
    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    return importance


def save(model: lgb.LGBMClassifier) -> None:
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {MODEL_PATH}")


def load() -> lgb.LGBMClassifier:
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict_proba(model: lgb.LGBMClassifier, X: pd.DataFrame) -> np.ndarray:
    """Returns probabilities for [SELL, HOLD, BUY] classes."""
    available = [c for c in FEATURE_COLS if c in X.columns]
    return model.predict_proba(X[available])


def predict_signal(model: lgb.LGBMClassifier, X: pd.DataFrame) -> np.ndarray:
    """Returns -1 (sell), 0 (hold), 1 (buy)."""
    available = [c for c in FEATURE_COLS if c in X.columns]
    return model.predict(X[available]) - 1
