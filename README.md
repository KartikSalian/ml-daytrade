# ML Day Trading System

Production-grade algorithmic trading system using machine learning ensemble for US equities. Running on Alpaca paper trading since May 2026.

## Live Performance (Paper Trading)
| Day | Trades | P&L |
|-----|--------|-----|
| May 26, 2026 | 3 | +$106 |
| May 27, 2026 | 5 | +$760 |
| May 28, 2026 | 15* | +$245 |
| **Total (3 days)** | **23** | **+$1,111** |

*Day 3 had a bug (position cap bypass) — fixed same day.

## Backtest Performance (OOS)
- **Sharpe Ratio**: 2.49 (bull model)
- **Sortino Ratio**: 4.37
- **Total Return**: +7.76% (OOS test period)
- **Max Drawdown**: 11.72%
- **Ensemble OOS Accuracy**: 40.8%
- **Bear Market Alpha**: +7.67% vs buy-and-hold (stress test)

## Architecture

### Dual Specialist Models
Two models trained on the same 2-year dataset with different class weights:

| Model | Class Weights | Used When | Features |
|-------|-------------|-----------|----------|
| **Bull** | SELL=0.5, HOLD=1.0, BUY=2.0 | BULL / NEUTRAL regime | 23 |
| **Bear** | SELL=2.0, HOLD=1.0, BUY=0.5 | BEAR / CHOPPY / HIGH_FEAR | 29 |

Each model = LightGBM + CNN-LSTM (24h lookback) + Weighted Ensemble

Saved separately: `models/saved/lgbm_bull.pkl`, `lgbm_bear.pkl`, `cnn_lstm_bull.pt`, `cnn_lstm_bear.pt`, `ensemble_bull.pkl`, `ensemble_bear.pkl`

### Signal Pipeline
```
Market Data → Feature Engineering → Regime Detection → Regime Persistence Check
→ Model Selection (Bull/Bear) → Ensemble Prediction → Risk Manager → Alpaca Execution
```

### Features
**Bull model (23):**
- Technical: MACD, RSI, Bollinger Bands, ATR, Stochastic, SMA cross, Volume ratio, momentum
- Macro: VIX, S&P500 return, XLK sector ETF
- Sentiment: FinBERT (yiyanghkust/finbert-tone) via Finnhub news
- Time-of-day: hour encoding, market open/close/dead-zone flags

**Bear model adds 6 macro bear indicators (29 total):**
- VIX momentum (5-period VIX change — panic spike detection)
- VIX z-score (VIX vs 30-day mean — fear level normalised)
- SP500 realised volatility (rolling turbulence measure)
- HYG return (high yield bond ETF — credit stress)
- TLT return (long treasury ETF — flight to safety)
- DXY return (US dollar ETF — risk-off signal)

### Regime Detection
Uses India/US VIX + 20-day index return to classify: BULL / BEAR / CHOPPY / HIGH_FEAR / NEUTRAL

**Regime persistence:** requires 3 consecutive cycles before switching models — prevents whipsawing on short-lived VIX spikes. State persisted to `data/regime_state.json` across sessions.

### Risk Management
- **Trailing stop**: 1.5% — moves up with price, locks in gains automatically
- **ATR-based position sizing**: 2% risk per trade
- **Dynamic position cap**: `(capital / max_positions) × 0.8` per position — spreads capital evenly; ATR naturally caps volatile stocks, this kicks in for stable ones
- **Cash guard**: hard stop if cash ≤ 0; each trade deducted from available cash before next order
- **Max 5 positions** per cycle (top 5 by confidence)
- **50% minimum confidence** threshold (bull) / 35% (bear)
- **10% max portfolio drawdown halt**
- **Earnings proximity guard**: halves size within 5 days of earnings
- **EOD close**: all positions closed at 3:50 PM EST daily
- **Process lock**: `data/autorun.lock` prevents duplicate autorun instances

## Tech Stack
- **Data**: yfinance (15 tickers, hourly, 2 years)
- **Sentiment**: FinBERT on CUDA (RTX 3050) + Finnhub free tier
- **Backtesting**: vectorbt + walk-forward validation
- **Execution**: Alpaca paper trading (alpaca-py)
- **Automation**: Windows Task Scheduler, runs Mon-Fri 2 PM IST (Ireland)

## Universe
AAPL, MSFT, GOOGL, AMZN, NVDA, META, AMD, JPM, V, BAC, TSLA, NFLX, XOM, JNJ, CRM

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env  # add your API keys
python sentiment/backfill.py  # backfill 18 months of sentiment
python retrain.py             # train bull + bear models
python autorun.py             # start trading loop
```

## Retraining
```bash
python retrain.py       # retrain both bull and bear models (same data, different class weights)
python retrain_bear.py  # retrain bear model only (faster)
```

## Backtesting
```bash
python backtest_bear.py        # bear market stress test
python backtest_walkforward.py # walk-forward overfitting validation
```

## Status
Live paper trading since May 26, 2026. Target: 200+ trades over 3 months before evaluating real capital allocation.

## Disclaimer
This is a personal research project. Not financial advice. Past performance does not guarantee future results.
