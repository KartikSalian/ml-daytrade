import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import config  # loads .env
from data.fetcher import get_ohlcv, get_macro_data, merge_macro_features
from data.universe import TICKERS, INTERVAL
from features.engineering import add_technical_features, add_time_features, FEATURE_COLS
from models import lgbm_model, cnn_lstm
from models.ensemble import load_ensemble, build_meta_features
from risk.manager import RiskManager
from execution.runner import get_vix_level

app = FastAPI(title="ML Day Trading API", version="1.0.0")

# Load models once on startup
_lgbm = None
_cnn = None
_meta = None
_risk = None


@app.on_event("startup")
def load_models():
    global _lgbm, _cnn, _meta, _risk
    sample = get_ohlcv("AAPL", period="1mo", interval=INTERVAL, use_cache=False)
    macro = get_macro_data(period="1mo", interval=INTERVAL, use_cache=False)
    sample = merge_macro_features(sample, macro)
    sample = add_technical_features(sample)
    sample = add_time_features(sample)
    sample["sentiment_score"] = 0.0
    n_features = len([c for c in FEATURE_COLS if c in sample.columns])

    _lgbm = lgbm_model.load()
    _cnn = cnn_lstm.load(input_size=n_features)
    _meta = load_ensemble()
    _risk = RiskManager(capital=10_000, risk_per_trade=0.02, max_drawdown=0.10)
    print("Models loaded.")


class SignalRequest(BaseModel):
    ticker: str
    sentiment_score: float = 0.0


class SignalResponse(BaseModel):
    ticker: str
    signal: str
    confidence: float
    price: float | None
    approved: bool
    qty: int
    stop_loss: float | None
    vix: float
    sentiment_score: float
    reason: str


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": _meta is not None}


@app.get("/account")
def account():
    from execution.alpaca import get_account
    try:
        return get_account()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/positions")
def positions():
    from execution.alpaca import get_positions
    try:
        return get_positions()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/signal", response_model=SignalResponse)
def get_signal(req: SignalRequest):
    ticker = req.ticker.upper()
    if ticker not in TICKERS:
        raise HTTPException(status_code=400, detail=f"{ticker} not in universe")

    vix = get_vix_level()
    if vix >= 30.0:
        return SignalResponse(
            ticker=ticker, signal="HOLD", confidence=0.0,
            price=None, approved=False, qty=0, stop_loss=None,
            vix=vix, sentiment_score=0.0, reason="vix_regime_halt",
        )

    try:
        macro = get_macro_data(period="3mo", interval=INTERVAL, use_cache=False)
        df = get_ohlcv(ticker, period="3mo", interval=INTERVAL, use_cache=False)
        df = merge_macro_features(df, macro)
        df = add_technical_features(df)
        df = add_time_features(df)
        # Auto-fetch live sentiment if caller didn't supply one
        if req.sentiment_score == 0.0:
            try:
                from sentiment.pipeline import compute_ticker_sentiment
                req.sentiment_score = compute_ticker_sentiment(ticker, days_back=1)
            except Exception:
                pass
        df["sentiment_score"] = req.sentiment_score
        df["ticker"] = ticker
        df.dropna(inplace=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data error: {e}")

    try:
        X_meta, _ = build_meta_features(df, _lgbm, _cnn)
        probs = _meta.predict_proba(X_meta)
        raw_signal = int(_meta.predict(X_meta)[-1]) - 1
        confidence = float(probs[-1].max())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    signal_map = {1: "BUY", -1: "SELL", 0: "HOLD"}

    from execution.alpaca import get_latest_price
    try:
        price = get_latest_price(ticker)
    except Exception:
        price = float(df["Close"].iloc[-1])

    atr = float(df["ATR"].iloc[-1])
    from execution.alpaca import get_account
    try:
        capital = float(get_account()["portfolio_value"])
    except Exception:
        capital = 10_000.0

    risk_result = _risk.evaluate_signal(
        ticker=ticker,
        signal=raw_signal,
        entry_price=price,
        atr=atr,
        current_capital=capital,
        confidence=confidence,
    )

    return SignalResponse(
        ticker=ticker,
        signal=signal_map[raw_signal],
        confidence=round(confidence, 4),
        price=price,
        approved=risk_result["approved"],
        qty=risk_result.get("qty", 0),
        stop_loss=risk_result.get("stop_loss"),
        vix=round(vix, 2),
        sentiment_score=round(req.sentiment_score, 4),
        reason=risk_result.get("reason", ""),
    )


@app.post("/execute/{ticker}")
def execute(ticker: str, req: SignalRequest):
    signal_resp = get_signal(req)
    if not signal_resp.approved:
        return {"executed": False, "reason": signal_resp.reason}

    from execution.alpaca import execute_signal
    signal_int = {"BUY": 1, "SELL": -1, "HOLD": 0}[signal_resp.signal]
    result = execute_signal(ticker, signal_int, signal_resp.qty, approved=True)
    return {"executed": result is not None, "order": result}
