import time
import pandas as pd
import numpy as np
from datetime import datetime, date

from data.fetcher import get_ohlcv, get_macro_data, merge_macro_features
from data.universe import TICKERS, INTERVAL
from data.earnings import get_earnings_dates, days_until_next_earnings
from features.engineering import (
    add_technical_features, add_time_features,
    add_bear_features, FEATURE_COLS, BEAR_FEATURE_COLS
)
from pathlib import Path
from models import lgbm_model, cnn_lstm
from models.ensemble import load_ensemble, build_meta_features
from risk.manager import RiskManager
from risk.regime import get_market_regime, filter_signal_by_regime, get_stable_model_regime
from sentiment.pipeline import compute_ticker_sentiment
from execution.alpaca import get_account, get_latest_price, execute_signal, get_positions

VIX_HALT_THRESHOLD = 30.0
SAVE_DIR = Path("D:/ml-daytrade/models/saved")
BULL_REGIMES = {"BULL", "NEUTRAL"}
BEAR_REGIMES = {"BEAR", "CHOPPY", "HIGH_FEAR"}


def load_models(n_features: int):
    """Load both bull and bear model sets."""
    from features.engineering import BEAR_FEATURE_COLS
    n_bear_features = len(BEAR_FEATURE_COLS)

    bull_lgbm = lgbm_model.load(SAVE_DIR / "lgbm_bull.pkl")
    bull_cnn  = cnn_lstm.load(n_features, SAVE_DIR / "cnn_lstm_bull.pt")
    bull_meta = load_ensemble(SAVE_DIR / "ensemble_bull.pkl")

    bear_lgbm = lgbm_model.load(SAVE_DIR / "lgbm_bear.pkl")
    bear_cnn  = cnn_lstm.load(n_bear_features, SAVE_DIR / "cnn_lstm_bear.pt")
    bear_meta = load_ensemble(SAVE_DIR / "ensemble_bear.pkl")

    return (bull_lgbm, bull_cnn, bull_meta), (bear_lgbm, bear_cnn, bear_meta)


def select_models(regime_info: dict, bull_models: tuple, bear_models: tuple) -> tuple:
    """Pick bull or bear model set based on current regime."""
    if regime_info["regime"] in BULL_REGIMES:
        return bull_models
    return bear_models


def get_vix_level() -> float:
    import yfinance as yf
    vix = yf.download("^VIX", period="1d", interval="1h", progress=False, auto_adjust=True)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    return float(vix["Close"].iloc[-1])


def build_ticker_df(ticker: str, macro: dict, include_bear_features: bool = False) -> pd.DataFrame | None:
    try:
        df = get_ohlcv(ticker, period="3mo", interval=INTERVAL, use_cache=False)
        df = merge_macro_features(df, macro)
        df = add_technical_features(df)
        df = add_time_features(df)
        if include_bear_features:
            df = add_bear_features(df)
        df["sentiment_score"] = 0.0
        df["ticker"] = ticker
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"  {ticker}: data error — {e}")
        return None


def run_once(
    risk_manager: RiskManager,
    bull_models: tuple,
    bear_models: tuple,
    use_live_sentiment: bool = True,
) -> list[dict]:
    """
    Run one full signal cycle across all tickers.
    Returns list of executed trades.
    """
    print(f"\n{'='*50}")
    print(f"Signal cycle: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Regime filter
    vix = get_vix_level()
    print(f"VIX: {vix:.1f}", end="")
    if vix >= VIX_HALT_THRESHOLD:
        print(f" — HALTED (VIX >= {VIX_HALT_THRESHOLD})")
        return []
    print(" — OK")

    regime_info = get_market_regime()
    print(f"Regime: {regime_info['regime']} — {regime_info['reason']}")
    if not regime_info["trade"]:
        print("HALTED — regime filter")
        return []

    # Select bull or bear model — requires 3 consecutive regime signals to switch
    stable_regime    = get_stable_model_regime(regime_info["regime"])
    using_bear       = stable_regime == "BEAR"
    lgbm, cnn, meta  = bear_models if using_bear else bull_models
    min_conf         = 0.35 if using_bear else 0.50  # bull raised to 0.50 for selectivity
    max_trades       = 5
    print(f"Model set: {stable_regime} (signal: {regime_info['regime']}, min_conf: {min_conf})")

    # Account state
    account = get_account()
    capital = float(account["portfolio_value"])
    cash    = float(account["cash"])
    print(f"Portfolio: ${capital:,.2f} | Cash: ${cash:,.2f}")
    risk_manager.update_peak(capital)

    if risk_manager.is_halted(capital):
        print("HALTED — max drawdown breached")
        return []

    # Safety: never deploy more than available cash
    if cash <= 0:
        print("HALTED — no cash available")
        return []

    # Macro data (shared across tickers)
    macro = get_macro_data(period="3mo", interval=INTERVAL, use_cache=False)

    # Earnings calendar (cached, refreshed weekly)
    earnings_dates = get_earnings_dates(TICKERS)

    # Live sentiment (optional — requires Finnhub key)
    sentiment_map: dict[str, float] = {}
    if use_live_sentiment:
        from sentiment.pipeline import compute_sentiment_batch
        print("Fetching sentiment...")
        sentiment_map = compute_sentiment_batch(TICKERS, days_back=1, delay=0.5)

    # Sync open positions into risk manager so max_positions limit works
    try:
        live_positions = get_positions()
        risk_manager.open_positions = {p["ticker"]: p for p in live_positions}
    except Exception as e:
        print(f"Warning: could not sync positions — {e}")

    # Score every ticker
    signals: list[dict] = []
    for ticker in TICKERS:
        # Never add to a position we already hold
        if ticker in risk_manager.open_positions:
            continue

        df = build_ticker_df(ticker, macro, include_bear_features=using_bear)
        if df is None or len(df) < cnn_lstm.SEQ_LEN + 1:
            continue

        # Inject live sentiment if available
        if ticker in sentiment_map:
            df["sentiment_score"] = sentiment_map[ticker]

        # Ensemble prediction
        try:
            feat_cols = BEAR_FEATURE_COLS if using_bear else None
            X_meta, _ = build_meta_features(df, lgbm, cnn, feature_cols=feat_cols)
            probs = meta.predict_proba(X_meta)
            signal = int(meta.predict(X_meta)[-1]) - 1
            confidence = float(probs[-1].max())
        except Exception as e:
            print(f"  {ticker}: prediction error — {e}")
            continue

        if signal == 0:
            continue

        # Regime filter
        signal, size_mult = filter_signal_by_regime(signal, regime_info)
        if signal == 0:
            continue

        # Risk check
        latest_atr = float(df["ATR"].iloc[-1])
        try:
            latest_price = get_latest_price(ticker, signal)
        except Exception as e:
            print(f"  {ticker}: price fetch failed — {e}")
            continue
        dte = days_until_next_earnings(date.today(), earnings_dates.get(ticker, []))
        risk_result = risk_manager.evaluate_signal(
            ticker=ticker,
            signal=signal,
            entry_price=latest_price,
            atr=latest_atr,
            current_capital=capital,
            confidence=confidence,
            days_to_earnings=dte,
            min_confidence=min_conf,
        )

        # Scale qty by regime size multiplier
        scaled_qty = int(risk_result.get("qty", 0) * size_mult)

        signals.append({
            "ticker": ticker,
            "signal": signal,
            "confidence": round(confidence, 3),
            "price": latest_price,
            "approved": risk_result["approved"] and scaled_qty > 0,
            "qty": scaled_qty,
            "stop_loss": risk_result.get("stop_loss"),
            "reason": risk_result.get("reason", ""),
            "regime": regime_info["regime"],
        })

    # Sort by confidence, take top signals
    signals.sort(key=lambda x: -x["confidence"])
    print(f"\n{'Ticker':6} {'Signal':6} {'Conf':6} {'Price':8} {'Qty':5} {'Regime':8} {'Approved'}")
    print("-" * 60)
    for s in signals:
        label = "BUY " if s["signal"] == 1 else "SELL"
        approved = "APPROVED" if s["approved"] else f"REJECTED {s['reason']}"
        print(f"{s['ticker']:6} {label:6} {s['confidence']:.3f}  ${s['price']:7.2f} {s['qty']:5}  {s['regime']:8} {approved}")

    # Execute top N approved signals — hard cap at max_trades and available cash
    executed = []
    deployed = 0.0
    for s in signals:
        if len(executed) >= max_trades:
            break
        if not s["approved"] or s["qty"] <= 0:
            continue
        trade_cost = s["qty"] * s["price"]
        if deployed + trade_cost > cash:
            print(f"  {s['ticker']}: skipped — would exceed available cash")
            continue
        result = execute_signal(s["ticker"], s["signal"], s["qty"])
        if result:
            executed.append({**s, "order": result})
            deployed += trade_cost

    print(f"\nExecuted {len(executed)} trades.")
    return executed


def run_loop(interval_minutes: int = 60, use_live_sentiment: bool = True):
    """Main trading loop — runs every interval_minutes."""
    print("Loading models...")
    # build a sample df to get n_features
    sample = get_ohlcv("AAPL", period="1mo", interval=INTERVAL, use_cache=False)
    macro = get_macro_data(period="1mo", interval=INTERVAL, use_cache=False)
    sample = merge_macro_features(sample, macro)
    sample = add_technical_features(sample)
    sample = add_time_features(sample)
    sample["sentiment_score"] = 0.0
    n_features = len([c for c in FEATURE_COLS if c in sample.columns])

    lgbm, cnn, meta = load_models(n_features)
    risk_manager = RiskManager(
        capital=10_000,
        risk_per_trade=0.02,
        max_drawdown=0.10,
        max_positions=5,
    )

    print(f"Starting trading loop (every {interval_minutes} min)...")
    while True:
        try:
            run_once(risk_manager, lgbm, cnn, meta, use_live_sentiment)
        except KeyboardInterrupt:
            print("\nStopped by user.")
            break
        except Exception as e:
            print(f"Cycle error: {e}")

        time.sleep(interval_minutes * 60)
