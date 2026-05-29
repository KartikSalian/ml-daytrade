import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
import json

REGIME_STATE_PATH = Path(__file__).parent.parent / "data" / "regime_state.json"
REGIME_CONFIRM_PERIODS = 3  # require 3 consecutive cycles before switching model


def _load_regime_state() -> dict:
    if REGIME_STATE_PATH.exists():
        with open(REGIME_STATE_PATH) as f:
            return json.load(f)
    return {"confirmed_regime": "BULL", "pending_regime": "BULL", "pending_count": 0}


def _save_regime_state(state: dict) -> None:
    REGIME_STATE_PATH.parent.mkdir(exist_ok=True)
    with open(REGIME_STATE_PATH, "w") as f:
        json.dump(state, f)


def get_stable_model_regime(current_regime: str) -> str:
    """
    Returns the model regime to USE — only switches after 3 consecutive
    cycles of the same new regime. Prevents whipsawing between bull/bear models
    on short-lived VIX spikes or single-day signals.
    """
    bull_regimes = {"BULL", "NEUTRAL"}
    bear_regimes = {"BEAR", "CHOPPY", "HIGH_FEAR"}

    # Simplify to bull/bear for model selection
    current_model = "BULL" if current_regime in bull_regimes else "BEAR"

    state = _load_regime_state()
    confirmed = state["confirmed_regime"]

    if current_model == confirmed:
        # Same as confirmed — reset pending
        state["pending_regime"] = current_model
        state["pending_count"] = 0
    elif current_model == state["pending_regime"]:
        # Same pending signal — increment counter
        state["pending_count"] += 1
        if state["pending_count"] >= REGIME_CONFIRM_PERIODS:
            print(f"Regime switch confirmed: {confirmed} -> {current_model} ({REGIME_CONFIRM_PERIODS} consecutive signals)")
            state["confirmed_regime"] = current_model
            state["pending_count"] = 0
    else:
        # New different signal — start counting
        state["pending_regime"] = current_model
        state["pending_count"] = 1

    _save_regime_state(state)

    if state["confirmed_regime"] != current_model:
        print(f"Regime signal: {current_model} (pending {state['pending_count']}/{REGIME_CONFIRM_PERIODS} — holding {state['confirmed_regime']} model)")

    return state["confirmed_regime"]


def get_market_regime() -> dict:
    """
    Detects current market regime using VIX + SP500 trend.
    Returns regime info and trading rules to apply.
    """
    # Fetch recent VIX and SP500
    vix = yf.download("^VIX", period="1mo", interval="1d", progress=False, auto_adjust=True)
    sp500 = yf.download("^GSPC", period="3mo", interval="1d", progress=False, auto_adjust=True)

    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    if isinstance(sp500.columns, pd.MultiIndex):
        sp500.columns = sp500.columns.get_level_values(0)

    current_vix = float(vix["Close"].iloc[-1])
    vix_5d_avg = float(vix["Close"].tail(5).mean())

    sp500_ret_20d = float(sp500["Close"].pct_change(20).iloc[-1])
    sp500_ret_5d = float(sp500["Close"].pct_change(5).iloc[-1])
    sp500_above_sma50 = float(sp500["Close"].iloc[-1]) > float(sp500["Close"].rolling(50).mean().iloc[-1])

    # Regime classification
    if current_vix > 30 or vix_5d_avg > 28:
        regime = "HIGH_FEAR"
    elif current_vix > 20 and sp500_ret_20d < -0.05:
        regime = "BEAR"
    elif current_vix > 20 and sp500_ret_20d < 0:
        regime = "CHOPPY"
    elif sp500_above_sma50 and sp500_ret_20d > 0:
        regime = "BULL"
    else:
        regime = "NEUTRAL"

    rules = _get_rules(regime)

    return {
        "regime": regime,
        "vix": round(current_vix, 2),
        "vix_5d_avg": round(vix_5d_avg, 2),
        "sp500_ret_20d": round(sp500_ret_20d * 100, 2),
        "sp500_ret_5d": round(sp500_ret_5d * 100, 2),
        "sp500_above_sma50": sp500_above_sma50,
        **rules,
    }


def _get_rules(regime: str) -> dict:
    """
    Returns trading rules for each regime.
    allowed_signals: which signals to act on
    size_multiplier: scale position size up/down
    trade: whether to trade at all
    """
    return {
        "HIGH_FEAR": {
            "trade": False,
            "allowed_signals": [],
            "size_multiplier": 0.0,
            "reason": "VIX too high — sitting out",
        },
        "BEAR": {
            "trade": True,
            "allowed_signals": [-1],       # SELL only
            "size_multiplier": 0.75,       # smaller size in bear
            "reason": "Bear regime — SELL signals only",
        },
        "CHOPPY": {
            "trade": True,
            "allowed_signals": [-1, 1],    # both but small
            "size_multiplier": 0.5,        # half size in choppy
            "reason": "Choppy regime — half position size",
        },
        "BULL": {
            "trade": True,
            "allowed_signals": [1],        # BUY only
            "size_multiplier": 1.0,
            "reason": "Bull regime — BUY signals only",
        },
        "NEUTRAL": {
            "trade": True,
            "allowed_signals": [-1, 1],
            "size_multiplier": 0.75,
            "reason": "Neutral regime — both signals, reduced size",
        },
    }[regime]


def filter_signal_by_regime(signal: int, regime_info: dict) -> tuple[int, float]:
    """
    Filters a signal based on current regime.
    Returns (filtered_signal, size_multiplier).
    """
    if not regime_info["trade"]:
        return 0, 0.0
    if signal not in regime_info["allowed_signals"]:
        return 0, 0.0
    return signal, regime_info["size_multiplier"]
