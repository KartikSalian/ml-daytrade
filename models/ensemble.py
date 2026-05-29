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
    feature_cols: list | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Get probability outputs from both models — properly aligned."""
    # LightGBM probabilities for all rows
    lgbm_probs = lgbm_model.predict_proba(lgbm, df, feature_cols=feature_cols)

    # CNN-LSTM sequences and probabilities
    sequences, labels = cnn_lstm.build_sequences(df, seq_len=cnn_lstm.SEQ_LEN, feature_cols=feature_cols)
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


def train(
    df: pd.DataFrame,
    save_path: Path | None = None,
    feature_cols: list | None = None,
    lgbm_model_obj=None,
    cnn_model_obj=None,
    lgbm_path: Path | None = None,
    cnn_path: Path | None = None,
) -> tuple[WeightedEnsemble, object, cnn_lstm.CNNLSTM]:
    print("Loading base models...")
    cols = feature_cols if feature_cols is not None else FEATURE_COLS
    n_features = len([c for c in cols if c in df.columns])

    if lgbm_model_obj is not None:
        lgbm = lgbm_model_obj
    else:
        lgbm = lgbm_model.load(lgbm_path)

    if cnn_model_obj is not None:
        cnn = cnn_model_obj
    else:
        cnn = cnn_lstm.load(input_size=n_features, path=cnn_path)

    print("Finding optimal weights on validation set...")
    split = int(len(df) * 0.8)
    val_df = df.iloc[split:]
    X_meta_val, y_val = build_meta_features(val_df, lgbm, cnn, feature_cols=feature_cols)

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

    p = Path(save_path) if save_path else ENSEMBLE_PATH
    with open(p, "wb") as f:
        pickle.dump(meta, f)
    print(f"Ensemble saved to {p}")

    return meta, lgbm, cnn


def load_ensemble(path: Path | None = None) -> WeightedEnsemble:
    p = Path(path) if path else ENSEMBLE_PATH
    with open(p, "rb") as f:
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
