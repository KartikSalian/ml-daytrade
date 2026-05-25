import numpy as np
import pandas as pd


def kelly_position_size(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    capital: float,
    max_fraction: float = 0.25,
) -> float:
    """Kelly criterion capped at max_fraction of capital."""
    if avg_loss == 0:
        return 0.0
    b = avg_win / avg_loss
    kelly = win_rate - ((1 - win_rate) / b)
    kelly = max(0.0, min(kelly, max_fraction))
    return capital * kelly


def position_size_fixed_risk(
    capital: float,
    entry_price: float,
    stop_loss_price: float,
    risk_pct: float = 0.02,
) -> float:
    """Risk a fixed % of capital per trade based on stop distance."""
    risk_per_share = abs(entry_price - stop_loss_price)
    if risk_per_share == 0:
        return 0.0
    dollar_risk = capital * risk_pct
    shares = dollar_risk / risk_per_share
    return shares


def atr_stop_loss(
    entry_price: float,
    atr: float,
    signal: int,
    multiplier: float = 2.0,
) -> float:
    """ATR-based stop loss. signal=1 (long) → stop below, signal=-1 (short) → stop above."""
    if signal == 1:
        return entry_price - multiplier * atr
    elif signal == -1:
        return entry_price + multiplier * atr
    return entry_price


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical VaR at given confidence level."""
    return float(np.percentile(returns.dropna(), (1 - confidence) * 100))


def conditional_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """CVaR (Expected Shortfall) — average loss beyond VaR."""
    var = value_at_risk(returns, confidence)
    tail = returns[returns <= var]
    return float(tail.mean()) if len(tail) > 0 else var


def check_drawdown_limit(
    portfolio_value: float,
    peak_value: float,
    max_drawdown: float = 0.10,
) -> bool:
    """Returns True if drawdown limit breached — stop trading."""
    drawdown = (peak_value - portfolio_value) / peak_value
    return drawdown >= max_drawdown


class RiskManager:
    def __init__(
        self,
        capital: float = 10_000,
        risk_per_trade: float = 0.02,
        max_drawdown: float = 0.10,
        atr_multiplier: float = 2.0,
        max_positions: int = 5,
        max_position_pct: float = 0.15,  # max 15% of capital per single position
    ):
        self.capital = capital
        self.risk_per_trade = risk_per_trade
        self.max_drawdown = max_drawdown
        self.atr_multiplier = atr_multiplier
        self.max_positions = max_positions
        self.max_position_pct = max_position_pct
        self.peak_value = capital
        self.open_positions: dict = {}

    def update_peak(self, current_value: float) -> None:
        if current_value > self.peak_value:
            self.peak_value = current_value

    def is_halted(self, current_value: float) -> bool:
        """Halt trading if max drawdown breached."""
        return check_drawdown_limit(current_value, self.peak_value, self.max_drawdown)

    def get_position_size(
        self,
        ticker: str,
        entry_price: float,
        atr: float,
        signal: int,
    ) -> dict:
        """Returns qty, stop_loss, and dollar_risk for a trade."""
        if len(self.open_positions) >= self.max_positions:
            return {"qty": 0, "reason": "max_positions_reached"}

        stop = atr_stop_loss(entry_price, atr, signal, self.atr_multiplier)
        qty = position_size_fixed_risk(
            self.capital, entry_price, stop, self.risk_per_trade
        )
        qty = max(0, int(qty))

        # Hard cap: no single position can exceed max_position_pct of capital
        max_qty_by_value = int(self.capital * self.max_position_pct / entry_price)
        qty = min(qty, max_qty_by_value)

        dollar_risk = qty * abs(entry_price - stop)

        return {
            "qty": qty,
            "stop_loss": round(stop, 4),
            "dollar_risk": round(dollar_risk, 2),
            "pct_risk": round(dollar_risk / self.capital * 100, 2),
        }

    def evaluate_signal(
        self,
        ticker: str,
        signal: int,
        entry_price: float,
        atr: float,
        current_capital: float,
        min_confidence: float = 0.40,
        confidence: float = 0.50,
        days_to_earnings: int = 90,
    ) -> dict:
        """
        Full pre-trade risk check.
        Returns approved=True/False with position sizing.
        """
        self.update_peak(current_capital)
        self.capital = current_capital  # keep sizing in sync with live portfolio

        if self.is_halted(current_capital):
            return {"approved": False, "reason": "drawdown_limit_breached"}

        if signal == 0:
            return {"approved": False, "reason": "hold_signal"}

        if confidence < min_confidence:
            return {"approved": False, "reason": f"low_confidence_{confidence:.2f}"}

        if days_to_earnings <= 1:
            return {"approved": False, "reason": "earnings_tomorrow"}

        sizing = self.get_position_size(ticker, entry_price, atr, signal)

        # Halve position size if earnings within 5 days
        if days_to_earnings <= 5 and sizing["qty"] > 0:
            sizing["qty"] = max(1, sizing["qty"] // 2)
            sizing["reason"] = "near_earnings_half_size"
        if sizing["qty"] == 0:
            return {"approved": False, "reason": sizing.get("reason", "zero_qty")}

        return {
            "approved": True,
            "qty": sizing["qty"],
            "stop_loss": sizing["stop_loss"],
            "dollar_risk": sizing["dollar_risk"],
            "pct_risk": sizing["pct_risk"],
        }
