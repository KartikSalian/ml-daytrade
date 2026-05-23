import os
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest

import config  # loads .env


def _get_client() -> TradingClient:
    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env")
    return TradingClient(key, secret, paper=True)


def _get_data_client() -> StockHistoricalDataClient:
    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_SECRET_KEY")
    return StockHistoricalDataClient(key, secret)


def get_account() -> dict:
    client = _get_client()
    account = client.get_account()
    return {
        "cash": float(account.cash),
        "portfolio_value": float(account.portfolio_value),
        "buying_power": float(account.buying_power),
        "equity": float(account.equity),
        "status": account.status,
    }


def get_positions() -> list[dict]:
    client = _get_client()
    positions = client.get_all_positions()
    return [
        {
            "ticker": p.symbol,
            "qty": float(p.qty),
            "side": p.side,
            "avg_entry": float(p.avg_entry_price),
            "market_value": float(p.market_value),
            "unrealized_pnl": float(p.unrealized_pl),
            "unrealized_pnl_pct": float(p.unrealized_plpc) * 100,
        }
        for p in positions
    ]


def get_latest_price(ticker: str) -> float:
    client = _get_data_client()
    req = StockLatestQuoteRequest(symbol_or_symbols=ticker)
    quote = client.get_stock_latest_quote(req)
    return float(quote[ticker].ask_price)


def submit_market_order(
    ticker: str,
    qty: int,
    side: str,  # "buy" or "sell"
) -> dict:
    client = _get_client()
    order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL
    req = MarketOrderRequest(
        symbol=ticker,
        qty=qty,
        side=order_side,
        time_in_force=TimeInForce.DAY,
    )
    order = client.submit_order(req)
    return {
        "id": str(order.id),
        "ticker": order.symbol,
        "qty": float(order.qty),
        "side": str(order.side),
        "status": str(order.status),
        "type": str(order.type),
    }


def close_position(ticker: str) -> dict:
    client = _get_client()
    order = client.close_position(ticker)
    return {
        "id": str(order.id),
        "ticker": order.symbol,
        "status": str(order.status),
    }


def close_all_positions() -> None:
    client = _get_client()
    client.close_all_positions(cancel_orders=True)
    print("All positions closed.")


def execute_signal(
    ticker: str,
    signal: int,
    qty: int,
    approved: bool = True,
) -> dict | None:
    """
    Execute a trade signal through Alpaca paper trading.
    signal: 1=BUY, -1=SELL/SHORT, 0=HOLD
    """
    if not approved or signal == 0 or qty <= 0:
        return None

    side = "buy" if signal == 1 else "sell"
    print(f"  Executing {side.upper()} {qty} {ticker}...")
    result = submit_market_order(ticker, qty, side)
    print(f"  Order {result['id']}: {result['status']}")
    return result
