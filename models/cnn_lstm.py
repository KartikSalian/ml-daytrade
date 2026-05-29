import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.metrics import accuracy_score

from features.engineering import FEATURE_COLS

SAVE_DIR = Path(__file__).parent / "saved"
SAVE_DIR.mkdir(exist_ok=True)
MODEL_PATH = SAVE_DIR / "cnn_lstm.pt"

SEQ_LEN = 24        # 24 hours lookback window
BATCH_SIZE = 512
EPOCHS = 30
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CNNLSTM(nn.Module):
    def __init__(self, input_size: int, num_classes: int = 3):
        super().__init__()

        # CNN: extract local patterns across time
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # LSTM: capture temporal dependencies
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )

        # Classifier head
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        x = x.permute(0, 2, 1)          # → (batch, features, seq_len) for Conv1d
        x = self.conv(x)                 # → (batch, 128, seq_len)
        x = x.permute(0, 2, 1)          # → (batch, seq_len, 128) for LSTM
        out, _ = self.lstm(x)            # → (batch, seq_len, 128)
        out = out[:, -1, :]              # last timestep → (batch, 128)
        return self.fc(out)              # → (batch, 3)


def build_sequences(df: pd.DataFrame, seq_len: int = SEQ_LEN, feature_cols: list | None = None):
    cols = feature_cols if feature_cols is not None else FEATURE_COLS
    available = [c for c in cols if c in df.columns]
    has_target = "target" in df.columns
    sequences, labels = [], []

    for ticker, group in df.groupby("ticker"):
        group = group.sort_index()
        X = group[available].values
        y = (group["target"] + 1).values if has_target else np.zeros(len(X), dtype=np.int64)

        for i in range(seq_len, len(X)):
            sequences.append(X[i - seq_len:i])
            labels.append(y[i])

    return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.int64)


def train(
    df: pd.DataFrame,
    val_ratio: float = 0.2,
    save_path: Path | None = None,
    class_weights: list[float] | None = None,
    feature_cols: list | None = None,
) -> CNNLSTM:
    """
    class_weights: [sell_w, hold_w, buy_w]
      None               — auto-balanced from class counts
      [0.5, 1.0, 2.0]   — bull mindset: favour BUY
      [2.0, 1.0, 0.5]   — bear mindset: favour SELL
    """
    print(f"Building sequences (seq_len={SEQ_LEN})...")
    X, y = build_sequences(df, feature_cols=feature_cols)
    print(f"Sequences: {X.shape}, Labels: {y.shape}")

    split = int(len(X) * (1 - val_ratio))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    train_loader = DataLoader(SequenceDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(SequenceDataset(X_val, y_val), batch_size=BATCH_SIZE)

    model = CNNLSTM(input_size=X.shape[2]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    if class_weights is not None:
        weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    else:
        counts = np.bincount(y_train)
        weights = torch.tensor(1.0 / counts, dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb.to(DEVICE)).argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(yb.numpy())

        val_acc = accuracy_score(all_labels, all_preds)
        scheduler.step(1 - val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path if save_path else MODEL_PATH)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS}  val_acc={val_acc:.4f}  best={best_val_acc:.4f}")

    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    p = save_path if save_path else MODEL_PATH
    model.load_state_dict(torch.load(p, weights_only=True))
    return model


def load(input_size: int, path: Path | None = None) -> CNNLSTM:
    p = Path(path) if path else MODEL_PATH
    model = CNNLSTM(input_size=input_size).to(DEVICE)
    model.load_state_dict(torch.load(p, weights_only=True, map_location=DEVICE))
    model.eval()
    return model


def predict_proba(model: CNNLSTM, sequences: np.ndarray) -> np.ndarray:
    """Returns softmax probabilities (N, 3) for [SELL, HOLD, BUY]."""
    device = next(model.parameters()).device  # use model's actual device (GPU or CPU)
    model.eval()
    all_probs = []
    loader = DataLoader(
        SequenceDataset(sequences, np.zeros(len(sequences), dtype=np.int64)),
        batch_size=BATCH_SIZE,
    )
    with torch.no_grad():
        for xb, _ in loader:
            logits = model(xb.to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
    return np.vstack(all_probs)
