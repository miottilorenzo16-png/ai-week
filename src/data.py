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

def fetch_pairs_data(
    ticker1: str,
    ticker2: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch adjusted closes for two tickers and return aligned daily log returns.

    The returned DataFrame has exactly two columns, named ticker1 and ticker2.
    Its index contains only dates where both stocks have valid adjusted closes.
    """
    if not ticker1 or not ticker2:
        raise ValueError("Both ticker symbols are required.")
    if ticker1 == ticker2:
        raise ValueError("ticker1 and ticker2 must be different.")
    if not start_date or not end_date or start_date >= end_date:
        raise ValueError("Provide valid dates with start_date earlier than end_date.")

    adjusted_closes = []
    for ticker in (ticker1, ticker2):
        history = yf.Ticker(ticker).history(
            start=start_date,
            end=end_date,
            interval="1d",
            auto_adjust=True,
            actions=False,
        )
        if history.empty or "Close" not in history.columns:
            raise ValueError(f"No adjusted closing prices returned for {ticker}.")

        close = history["Close"].rename(ticker)
        close.index = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
        adjusted_closes.append(close)

    prices = pd.concat(adjusted_closes, axis=1, join="inner")
    prices = prices[~prices.index.duplicated(keep="last")].sort_index()
    prices = prices.replace([np.inf, -np.inf], np.nan).dropna(subset=[ticker1, ticker2])
    prices = prices.loc[:, [ticker1, ticker2]]

    if prices.empty:
        raise ValueError("No overlapping valid trading dates were found for the two tickers.")
    if (prices <= 0).any().any():
        raise ValueError("Adjusted closing prices must be positive to calculate log returns.")

    returns = np.log(prices.astype(float)).diff().dropna(how="any")
    returns = returns.loc[:, [ticker1, ticker2]]
    returns.index.name = "Date"
    return returns
