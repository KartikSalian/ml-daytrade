import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report

from models import lgbm_model, cnn_lstm
from features.engineering import FEATURE_COLS

SAVE_DIR = Path(__file__).parent / "saved"
ENSEMBLE_PATH = SAVE_DIR / "ensemble.pkl"


class WeightedEnsemble:
    """Combines LightGBM and CNN-LSTM via weighted probability average."""

    def __init__(self, lgbm_weight: float = 0.5, cnn_weight: float = 0.5):
        self.lgbm_weight = lgbm_weight
        self.cnn_weight = cnn_weight

    def predict_proba(self, X_meta: np.ndarray) -> np.ndarray:
        # X_meta = [lgbm_p0, lgbm_p1, lgbm_p2, cnn_p0, cnn_p1, cnn_p2]
        lgbm_probs = X_meta[:, :3]
        cnn_probs = X_meta[:, 3:]
        return self.lgbm_weight * lgbm_probs + self.cnn_weight * cnn_probs

    def predict(self, X_meta: np.ndarray) -> np.ndarray:
        return self.predict_proba(X_meta).argmax(axis=1)


def _get_cnn_output_indices(df: pd.DataFrame) -> np.ndarray:
    """
    Returns integer positions in df that correspond to CNN-LSTM output rows.
    Mirrors build_sequences() groupby logic exactly so LightGBM predictions
    can be correctly aligned with CNN-LSTM predictions.
    """
    df_pos = df.copy()
    df_pos["__pos"] = np.arange(len(df))
    indices = []
    for ticker, group in df_pos.groupby("ticker"):
        positions = group.sort_index()["__pos"].values
        for i in range(cnn_lstm.SEQ_LEN, len(positions)):
            indices.append(positions[i])
    return np.array(indices)


def build_meta_features(
    df: pd.DataFrame,
    lgbm: object,
    cnn: cnn_lstm.CNNLSTM,
) -> tuple[np.ndarray, np.ndarray]:
    """Get probability outputs from both models — properly aligned."""
    # LightGBM probabilities for all rows
    lgbm_probs = lgbm_model.predict_proba(lgbm, df)

    # CNN-LSTM sequences and probabilities
    sequences, labels = cnn_lstm.build_sequences(df, seq_len=cnn_lstm.SEQ_LEN)
    cnn_probs = cnn_lstm.predict_proba(cnn, sequences)

    # Exact alignment: select only the lgbm rows that CNN-LSTM outputs correspond to
    seq_indices = _get_cnn_output_indices(df)
    lgbm_probs_aligned = lgbm_probs[seq_indices]

    # Safety trim in case of any off-by-one
    min_len = min(len(lgbm_probs_aligned), len(cnn_probs), len(labels))
    lgbm_probs_aligned = lgbm_probs_aligned[:min_len]
    cnn_probs = cnn_probs[:min_len]
    labels = labels[:min_len]

    meta_X = np.hstack([lgbm_probs_aligned, cnn_probs])
    return meta_X, labels


def train(df: pd.DataFrame) -> tuple[WeightedEnsemble, object, cnn_lstm.CNNLSTM]:
    print("Loading base models...")
    lgbm = lgbm_model.load()
    n_features = len([c for c in FEATURE_COLS if c in df.columns])
    cnn = cnn_lstm.load(input_size=n_features)

    print("Finding optimal weights on validation set...")
    split = int(len(df) * 0.8)
    val_df = df.iloc[split:]
    X_meta_val, y_val = build_meta_features(val_df, lgbm, cnn)

    # Grid search over weights
    best_acc, best_w = 0.0, 0.5
    for w in np.arange(0.3, 0.8, 0.05):
        ens = WeightedEnsemble(lgbm_weight=w, cnn_weight=1 - w)
        preds = ens.predict(X_meta_val)
        acc = accuracy_score(y_val, preds)
        if acc > best_acc:
            best_acc, best_w = acc, w

    meta = WeightedEnsemble(lgbm_weight=best_w, cnn_weight=1 - best_w)
    preds = meta.predict(X_meta_val)
    print(f"Best weights: LightGBM={best_w:.2f} CNN={1-best_w:.2f}  val_acc={best_acc:.4f}")
    print(classification_report(y_val, preds, target_names=["SELL", "HOLD", "BUY"]))

    with open(ENSEMBLE_PATH, "wb") as f:
        pickle.dump(meta, f)
    print(f"Ensemble saved to {ENSEMBLE_PATH}")

    return meta, lgbm, cnn


def load_ensemble() -> WeightedEnsemble:
    with open(ENSEMBLE_PATH, "rb") as f:
        return pickle.load(f)


def predict(
    meta: WeightedEnsemble,
    lgbm: object,
    cnn: cnn_lstm.CNNLSTM,
    df: pd.DataFrame,
) -> np.ndarray:
    X_meta, _ = build_meta_features(df, lgbm, cnn)
    return meta.predict(X_meta) - 1


def predict_proba(
    meta: WeightedEnsemble,
    lgbm: object,
    cnn: cnn_lstm.CNNLSTM,
    df: pd.DataFrame,
) -> np.ndarray:
    X_meta, _ = build_meta_features(df, lgbm, cnn)
    return meta.predict_proba(X_meta)
