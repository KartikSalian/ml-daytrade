import numpy as np
import pandas as pd
import vectorbt as vbt


def run_backtest(
    df: pd.DataFrame,
    signals: np.ndarray,
    init_cash: float = 10_000,
    fees: float = 0.001,
    slippage: float = 0.001,
) -> vbt.Portfolio:
    price = df["Close"]
    sig = pd.Series(signals, index=price.index[-len(signals):])

    entries = sig == 1
    exits = sig == -1

    portfolio = vbt.Portfolio.from_signals(
        price[-len(signals):],
        entries,
        exits,
        init_cash=init_cash,
        fees=fees,
        slippage=slippage,
        freq="1h",
    )
    return portfolio


def run_walkforward(
    df: pd.DataFrame,
    train_fn,
    predict_fn,
    train_size: int = 2000,
    test_size: int = 500,
    init_cash: float = 10_000,
) -> dict:
    """
    Walk-forward backtest: train on window, test on next window, roll forward.
    train_fn(train_df) -> model
    predict_fn(model, test_df) -> signals array
    """
    all_signals = []
    all_prices = []
    windows = []

    start = 0
    window_num = 0
    while start + train_size + test_size <= len(df):
        train_df = df.iloc[start: start + train_size]
        test_df = df.iloc[start + train_size: start + train_size + test_size]

        model = train_fn(train_df)
        signals = predict_fn(model, test_df)

        all_signals.append(signals)
        all_prices.append(test_df["Close"].values)
        windows.append((start, start + train_size, start + train_size + test_size))

        window_num += 1
        start += test_size

    signals_concat = np.concatenate(all_signals)
    prices_series = pd.Series(
        np.concatenate(all_prices),
        index=df.index[train_size: train_size + len(signals_concat)],
    )

    sig = pd.Series(signals_concat, index=prices_series.index)
    entries = sig == 1
    exits = sig == -1

    portfolio = vbt.Portfolio.from_signals(
        prices_series,
        entries,
        exits,
        init_cash=init_cash,
        fees=0.001,
        slippage=0.001,
        freq="1h",
    )

    return {
        "portfolio": portfolio,
        "n_windows": window_num,
        "signals": signals_concat,
    }


def print_stats(portfolio: vbt.Portfolio) -> None:
    stats = portfolio.stats()
    metrics = [
        "Total Return [%]",
        "Annualized Return [%]",
        "Max Drawdown [%]",
        "Sharpe Ratio",
        "Sortino Ratio",
        "Win Rate [%]",
        "Total Trades",
    ]
    print("\n=== BACKTEST RESULTS ===")
    for m in metrics:
        if m in stats.index:
            print(f"  {m:30s}: {stats[m]:.2f}")


def compare_to_buyhold(portfolio: vbt.Portfolio, price: pd.Series) -> None:
    bh_return = (price.iloc[-1] / price.iloc[0] - 1) * 100
    strat_return = portfolio.stats()["Total Return [%]"]
    print(f"\n  Strategy return : {strat_return:.2f}%")
    print(f"  Buy & Hold      : {bh_return:.2f}%")
    print(f"  Alpha           : {strat_return - bh_return:.2f}%")
