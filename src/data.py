"""Adjusted daily price data for the BNP Paribas / Société Générale pair."""
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
import yfinance as yf

PAIR_TICKERS = ("BNP.PA", "GLE.PA")

def download_adjusted_closes(
    start: str = "2021-01-01",
    end: Optional[str] = None,
    tickers: tuple[str, str] = PAIR_TICKERS,
) -> pd.DataFrame:
    """Download daily adjusted closes and align both stocks by trading date."""
    columns = []
    for ticker in tickers:
        history = yf.Ticker(ticker).history(
            start=start, end=end, interval="1d", auto_adjust=True, actions=False
        )
        if history.empty or "Close" not in history:
            raise ValueError(f"No daily close data returned for {ticker}.")
        close = history["Close"].rename(ticker)
        close.index = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
        columns.append(close)

    prices = pd.concat(columns, axis=1, join="inner").dropna()
    prices = prices[~prices.index.duplicated(keep="last")].sort_index()
    if len(prices) < 60:
        raise ValueError(f"Only {len(prices)} overlapping observations; need at least 60.")
    if (prices <= 0).any().any():
        raise ValueError("Prices must be positive to calculate log prices.")
    return prices

def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate aligned daily log returns."""
    return np.log(prices.astype(float)).diff().dropna()
