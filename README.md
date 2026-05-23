# ML Day Trading System

Production-grade algorithmic trading system using machine learning ensemble for US equities. Currently running on Alpaca paper trading.

## Performance (OOS Backtest)
- **Sharpe Ratio**: 2.49
- **Sortino Ratio**: 4.37
- **Total Return**: +7.76% (OOS test period)
- **Max Drawdown**: 11.72%
- **Ensemble OOS Accuracy**: 40.8%

## Architecture

### Models
- **LightGBM** — tabular feature model (CV accuracy: 40.3%)
- **CNN-LSTM** — sequential pattern model with 24-hour lookback (val accuracy: 40.3%)
- **Weighted Ensemble** — combines both models via optimised probability averaging

### Signal Pipeline
```
Market Data → Feature Engineering → Ensemble Prediction → Regime Filter → Risk Manager → Alpaca Execution
```

### Features (23)
- Technical indicators: MACD, RSI, Bollinger Bands, ATR, Stochastic, SMA cross
- Macro: VIX, S&P500 return, XLK sector ETF
- Sentiment: FinBERT (yiyanghkust/finbert-tone) scored daily via Finnhub news
- Time-of-day: hour encoding, market open/close/dead-zone flags
- Target: 4-hour forward return with 0.3% threshold → BUY / SELL / HOLD

### Risk Management
- ATR-based stop losses (2× ATR)
- Fixed 2% risk per trade
- 10% max portfolio drawdown halt
- 45% minimum confidence threshold
- Earnings proximity guard (halves size within 5 days of earnings)
- Market regime filter (BULL/BEAR/CHOPPY/HIGH_FEAR/NEUTRAL)

## Tech Stack
- **Data**: yfinance (15 tickers, hourly, 2 years)
- **Sentiment**: FinBERT on CUDA (RTX 3050) + Finnhub free tier
- **Backtesting**: vectorbt
- **Execution**: Alpaca paper trading (alpaca-py)
- **API**: FastAPI + Docker
- **Automation**: Windows Task Scheduler, runs Mon-Fri 2:30-9PM IST

## Universe
AAPL, MSFT, GOOGL, AMZN, NVDA, META, AMD, JPM, V, BAC, TSLA, NFLX, XOM, JNJ, CRM

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env  # add your API keys
python retrain.py     # train models
python autorun.py     # start trading loop
```

## Status
Paper trading since May 2026. Target: 200+ trades over 3 months before evaluating real capital allocation.

## Disclaimer
This is a personal research project. Not financial advice. Past backtest performance does not guarantee future results.
