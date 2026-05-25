"""
Master automation script.
- Runs trading loop every hour during US market hours (9:30 AM - 4 PM EST)
- Runs sentiment logger at 4:15 PM EST daily
- Logs all activity to data/autorun.log and data/trade_journal.csv
- Set this up in Windows Task Scheduler to run on weekdays at 2 PM IST/GMT (Ireland)

Run manually:
    python autorun.py
"""
import sys
sys.path.insert(0, "D:/ml-daytrade")

import time
import csv
import logging
from datetime import datetime, date
from pathlib import Path

import pytz
import config
from data.universe import TICKERS, INTERVAL
from data.fetcher import get_ohlcv, get_macro_data, merge_macro_features
from features.engineering import add_technical_features, add_time_features, FEATURE_COLS
from models import lgbm_model, cnn_lstm
from models.ensemble import load_ensemble, build_meta_features
from risk.manager import RiskManager
from execution.runner import get_vix_level, build_ticker_df, run_once
from sentiment.logger import log_today

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
LOG_PATH = ROOT / "data" / "autorun.log"
JOURNAL_PATH = ROOT / "data" / "trade_journal.csv"
JOURNAL_FIELDS = [
    "date", "time", "ticker", "signal", "confidence",
    "price", "qty", "stop_loss", "regime", "reason", "approved",
]

EST = pytz.timezone("US/Eastern")

# ── Logging setup ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def now_est() -> datetime:
    return datetime.now(EST)


def is_market_hours() -> bool:
    n = now_est()
    if n.weekday() >= 5:
        return False
    after_open = n.hour > 9 or (n.hour == 9 and n.minute >= 30)
    before_close = n.hour < 16
    return after_open and before_close


def is_sentiment_window() -> bool:
    """4:15-4:25 PM EST — run once after market close."""
    n = now_est()
    return n.weekday() < 5 and n.hour == 16 and 15 <= n.minute <= 25


def _ensure_journal():
    if not JOURNAL_PATH.exists():
        with open(JOURNAL_PATH, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=JOURNAL_FIELDS).writeheader()


def log_trades(signals: list[dict]):
    """Append trade signals to the journal CSV."""
    _ensure_journal()
    n = now_est()
    with open(JOURNAL_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=JOURNAL_FIELDS)
        for s in signals:
            writer.writerow({
                "date": n.strftime("%Y-%m-%d"),
                "time": n.strftime("%H:%M"),
                "ticker": s["ticker"],
                "signal": "BUY" if s["signal"] == 1 else "SELL",
                "confidence": s["confidence"],
                "price": s["price"],
                "qty": s["qty"],
                "stop_loss": s.get("stop_loss", ""),
                "regime": s.get("regime", ""),
                "reason": s.get("reason", ""),
                "approved": s["approved"],
            })


def load_models():
    log.info("Loading models...")
    sample = get_ohlcv("AAPL", period="1mo", interval=INTERVAL, use_cache=False)
    macro = get_macro_data(period="1mo", interval=INTERVAL, use_cache=False)
    sample = merge_macro_features(sample, macro)
    sample = add_technical_features(sample)
    sample = add_time_features(sample)
    sample["sentiment_score"] = 0.0
    n_features = len([c for c in FEATURE_COLS if c in sample.columns])

    lgbm = lgbm_model.load()
    cnn = cnn_lstm.load(input_size=n_features)
    meta = load_ensemble()
    risk = RiskManager(
        capital=100_000,
        risk_per_trade=0.02,
        max_drawdown=0.10,
        max_positions=5,
    )
    log.info(f"Models loaded. Features: {n_features}")
    return lgbm, cnn, meta, risk


def main():
    log.info("=" * 60)
    log.info("Autorun started")
    _ensure_journal()

    lgbm, cnn, meta, risk = load_models()

    last_trade_hour = -1
    sentiment_logged_today = False
    today = date.today()

    log.info("Waiting for market hours (9:30 AM - 4:00 PM EST)...")

    while True:
        n = now_est()

        # Reset daily flags at midnight
        if n.date() != today:
            today = n.date()
            sentiment_logged_today = False
            last_trade_hour = -1
            log.info(f"New trading day: {today}")

        # Skip weekends
        if n.weekday() >= 5:
            log.info("Weekend — sleeping 1 hour")
            time.sleep(3600)
            continue

        # ── Trading cycle (once per hour during market hours) ──────────────
        if is_market_hours() and n.hour != last_trade_hour:
            last_trade_hour = n.hour
            log.info(f"--- Signal cycle {n.strftime('%H:%M EST')} ---")
            try:
                signals = run_once(risk, lgbm, cnn, meta, use_live_sentiment=True)
                log_trades(signals)
                approved = [s for s in signals if s["approved"]]
                log.info(f"Cycle done: {len(signals)} signals, {len(approved)} executed")
            except Exception as e:
                log.error(f"Cycle error: {e}")

        # ── Sentiment logger (once per day at 4:15 PM EST) ─────────────────
        elif is_sentiment_window() and not sentiment_logged_today:
            sentiment_logged_today = True
            log.info("Running daily sentiment logger...")
            try:
                log_today()
                log.info("Sentiment logged.")
            except Exception as e:
                log.error(f"Sentiment error: {e}")

        time.sleep(60)  # check every minute


if __name__ == "__main__":
    main()
