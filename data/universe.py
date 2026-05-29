TICKERS = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "AMD",
    # Finance
    "JPM", "V", "BAC",
    # Consumer / Growth
    "TSLA", "NFLX",
    # Energy / Healthcare / SaaS
    "XOM", "JNJ", "CRM",
]

MACRO_TICKERS = {
    "VIX": "^VIX",
    "SP500": "^GSPC",
    "TECH_ETF": "XLK",
    "ENERGY_ETF": "XLE",
    "FINANCE_ETF": "XLF",
    "HEALTH_ETF": "XLV",
    # Bear market macro indicators
    "HYG": "HYG",      # High yield corporate bond ETF (credit stress)
    "TLT": "TLT",      # 20yr treasury ETF (flight to safety)
    "DXY": "UUP",      # US dollar ETF (risk-off indicator)
}

INTERVAL = "1h"
PERIOD = "2y"
