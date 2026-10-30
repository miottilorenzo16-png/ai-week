"""Download and screen a broad universe of CAC 40 stocks."""
from __future__ import annotations

from itertools import combinations
import itertools

import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.tsa.stattools as ts


# CAC 40 constituents as of 2026-09-30, using Yahoo Finance Paris suffixes.
# Index membership changes over time; this is a current-universe snapshot.
CAC40_TICKERS = [
    "SU.PA", "TTE.PA", "AIR.PA", "AI.PA", "SAF.PA",
    "BNP.PA", "MC.PA", "OR.PA", "SAN.PA", "CS.PA",
    "DG.PA", "EL.PA", "GLE.PA", "ENGI.PA", "RMS.PA",
    "BN.PA", "LR.PA", "SGO.PA", "STMPA.PA", "ORA.PA",
    "MT.PA", "ML.PA", "PUB.PA", "HO.PA", "VIE.PA",
    "KER.PA", "CAP.PA", "ACA.PA", "DSY.PA", "RI.PA",
    "ENX.PA", "CA.PA", "URW.PA", "BVI.PA", "EN.PA",
    "ERF.PA", "STLA.PA", "AC.PA", "FGR.PA", "RNO.PA",
]


def fetch_universe_data(tickers, start_date):
    """Download daily Close prices, screen sparse tickers, then fill missing values.

    Columns with more than 10% missing prices over the requested history are
    excluded. Remaining gaps are forward-filled and then backward-filled.
    """
    if not tickers:
        raise ValueError("tickers must contain at least one Yahoo Finance symbol.")
    if not start_date:
        raise ValueError("start_date is required.")

    requested_tickers = list(dict.fromkeys(tickers))
    downloaded = yf.download(
        tickers=requested_tickers,
        start=start_date,
        auto_adjust=False,
        actions=False,
        progress=False,
    )
    if downloaded.empty:
        raise ValueError("Yahoo Finance returned no price data.")

    # Recent yfinance versions return MultiIndex columns for multiple symbols.
    if isinstance(downloaded.columns, pd.MultiIndex):
        if "Close" not in downloaded.columns.get_level_values(0):
            raise ValueError("Yahoo Finance response does not contain Close prices.")
        close = downloaded["Close"].copy()
    elif "Close" in downloaded.columns:
        close = downloaded[["Close"]].copy()
        close.columns = requested_tickers[:1]
    else:
        raise ValueError("Yahoo Finance response does not contain Close prices.")

    # Keep requested ordering and omit symbols Yahoo did not return.
    close = close.reindex(columns=[ticker for ticker in requested_tickers if ticker in close.columns])
    if close.empty:
        raise ValueError("No requested tickers returned Close price data.")

    missing_fraction = close.isna().mean()
    close = close.loc[:, missing_fraction <= 0.10]
    if close.empty:
        raise ValueError("All tickers had more than 10% missing Close prices.")

    close = close.sort_index().ffill().bfill()
    close = close.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    close.index.name = "Date"
    return close
