import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "yiyanghkust/finbert-tone"
_tokenizer = None
_model = None


def _load_model():
    global _tokenizer, _model
    if _model is None:
        print("Loading FinBERT model (first run downloads ~500MB)...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = _model.to(device)
        _model.eval()
        print(f"FinBERT loaded on {device}")
    return _tokenizer, _model


def score_text(text: str) -> float:
    """
    Returns a sentiment score in [-1, +1].
    +1 = fully bullish, -1 = fully bearish, 0 = neutral.
    """
    if not text or not text.strip():
        return 0.0

    tokenizer, model = _load_model()
    device = next(model.parameters()).device

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

    # finbert-tone labels: 0=Neutral, 1=Positive, 2=Negative
    neutral, positive, negative = probs[0], probs[1], probs[2]
    return float(positive - negative)


def score_article(title: str, summary: str) -> float:
    text = f"{title}. {summary}".strip()
    return score_text(text)


def aggregate_scores(scores: list[float]) -> float:
    """Confidence-weighted mean — ignores near-zero scores."""
    if not scores:
        return 0.0
    weights = [abs(s) for s in scores]
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0
    return sum(s * w for s, w in zip(scores, weights)) / total_weight
