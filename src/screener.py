"""Download and screen a broad universe of CAC 40 stocks."""
from __future__ import annotations

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
    "MT.AS", "ML.PA", "PUB.PA", "HO.PA", "VIE.PA",
    "KER.PA", "CAP.PA", "ACA.PA", "DSY.PA", "RI.PA",
    "ENX.PA", "CA.PA", "URW.PA", "BVI.PA", "EN.PA",
    "ERF.PA", "STLAP.PA", "AC.PA", "FGR.PA", "RNO.PA",
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

    if isinstance(downloaded.columns, pd.MultiIndex):
        # Handle either yfinance MultiIndex order: (Price, Ticker) or
        # (Ticker, Price).
        close_level = next(
            (
                level
                for level in range(downloaded.columns.nlevels)
                if "Close" in downloaded.columns.get_level_values(level)
            ),
            None,
        )
        if close_level is None:
            raise ValueError("Yahoo Finance response does not contain Close prices.")
        close = downloaded.xs("Close", axis=1, level=close_level).copy()
    elif "Close" in downloaded.columns:
        close = downloaded[["Close"]].copy()
        close.columns = requested_tickers[:1]
    else:
        raise ValueError("Yahoo Finance response does not contain Close prices.")

    returned_tickers = [ticker for ticker in requested_tickers if ticker in close.columns]
    close = close.reindex(columns=returned_tickers)
    if close.empty or close.shape[1] == 0:
        raise ValueError("No requested tickers returned Close price data.")

    missing_fraction = close.isna().mean()
    close = close.loc[:, missing_fraction <= 0.10]
    if close.empty:
        raise ValueError("All tickers had more than 10% missing Close prices.")

    close = close.sort_index().ffill().bfill()
    close = close.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    close.index.name = "Date"
    return close


def scan_cointegrated_pairs(
    prices, p_value_threshold=0.05, progress_callback=None
):
    """Find cointegrated pairs and optionally report scan progress.

    Test statistics and hedge ratios use natural-log prices. The beta is the
    OLS slope from regressing ticker1 log prices on ticker2 log prices.
    progress_callback, when provided, is called with a fraction from 0 to 1.
    """
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame.")
    if prices.shape[1] < 2:
        raise ValueError("prices must contain at least two ticker columns.")
    if not 0 < p_value_threshold < 1:
        raise ValueError("p_value_threshold must be between 0 and 1.")

    pairs = list(itertools.combinations(prices.columns, 2))
    results = []
    total_pairs = len(pairs)

    for pair_number, (ticker1, ticker2) in enumerate(pairs, start=1):
        try:
            pair_prices = prices[[ticker1, ticker2]].replace(
                [np.inf, -np.inf], np.nan
            ).dropna()
            if pair_prices.empty or (pair_prices <= 0).any().any():
                continue

            log_price1 = np.log(pair_prices[ticker1].astype(float))
            log_price2 = np.log(pair_prices[ticker2].astype(float))
            if log_price1.nunique() < 2 or log_price2.nunique() < 2:
                continue

            try:
                test_statistic, p_value, _ = ts.coint(log_price1, log_price2)
            except (ValueError, np.linalg.LinAlgError):
                continue

            if np.isfinite(p_value) and p_value < p_value_threshold:
                beta = float(np.polyfit(log_price2, log_price1, 1)[0])
                results.append(
                    {
                        "ticker1": ticker1,
                        "ticker2": ticker2,
                        "p_value": float(p_value),
                        "test_statistic": float(test_statistic),
                        "beta": beta,
                    }
                )
        finally:
            if progress_callback is not None and total_pairs:
                progress_callback(pair_number / total_pairs)

    columns = ["ticker1", "ticker2", "p_value", "test_statistic", "beta"]
    return pd.DataFrame(results, columns=columns).sort_values(
        "p_value", ascending=True, ignore_index=True
    )


if __name__ == "__main__":
    universe_prices = fetch_universe_data(CAC40_TICKERS, start_date="2021-01-01")
    cointegrated_pairs = scan_cointegrated_pairs(universe_prices)

    print("\nTop 10 cointegrated CAC 40 pairs")
    if cointegrated_pairs.empty:
        print("No pairs found below the p-value threshold.")
    else:
        print(
            cointegrated_pairs.head(10).to_string(
                index=False,
                formatters={
                    "p_value": "{:.4g}".format,
                    "test_statistic": "{:.4f}".format,
                    "beta": "{:.4f}".format,
                },
            )
        )
