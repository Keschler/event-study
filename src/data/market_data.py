from __future__ import annotations

import pandas as pd
import yfinance as yf


def download_daily_prices(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    data = yf.download(
        symbols,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="column",
    )
    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.get_level_values(0):
            prices = data["Adj Close"].copy()
        else:
            prices = data["Close"].copy()
    else:
        prices = data.to_frame(name=symbols[0])

    prices.index = pd.to_datetime(prices.index)
    return prices.sort_index()


def compute_daily_returns(
    prices: pd.DataFrame,
    benchmark_symbol: str | None = None,
) -> pd.DataFrame:
    if prices.empty:
        return prices

    if benchmark_symbol and benchmark_symbol in prices.columns:
        benchmark_index = prices[benchmark_symbol].dropna().index
    else:
        benchmark_index = prices.index

    returns = prices.pct_change()
    returns = returns.reindex(benchmark_index)
    return returns
