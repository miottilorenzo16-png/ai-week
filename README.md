# BNP Paribas × Société Générale Pairs Trading

A Python research repository for exploring a statistical-arbitrage idea between two Paris-listed bank stocks: BNP Paribas (BNP.PA) and Société Générale (GLE.PA).

## Repository structure

```
.
├── notebooks/
│   ├── full_pairs_pipeline.ipynb
│   └── pairs_research.ipynb
├── src/
│   ├── __init__.py
│   ├── analytics.py
│   ├── backtest.py
│   └── data.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Setup

Requires Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Open `notebooks/full_pairs_pipeline.ipynb` in Jupyter for the complete three-chart pipeline, or open `notebooks/pairs_research.ipynb` for the shorter exploratory workflow. You can also run the analysis from Python:

```python
from src.data import download_adjusted_closes
from src.analytics import fit_hedge_ratio, engle_granger_test, rolling_zscore
from src.backtest import backtest_pair, performance_summary

prices = download_adjusted_closes(start="2021-01-01")
fit = fit_hedge_ratio(prices)
test = engle_granger_test(prices)
zscore = rolling_zscore(fit["spread"], window=60)
result = backtest_pair(prices, zscore, fit["beta"])
summary = performance_summary(result)
print(test)
print(summary)
```

## Research workflow

1. Download daily adjusted closes and align the two price series by trading date.
2. Estimate a fixed hedge ratio by regressing log BNP Paribas prices on log Société Générale prices.
3. Run the Engle–Granger cointegration test on log prices.
4. Calculate a trailing spread z-score.
5. Simulate a simple mean-reversion rule with prior-close signals and an estimated transaction cost.
6. Optionally estimate a time-varying hedge ratio with a Kalman filter.

## Important limitations

This is an educational research scaffold, not a validated trading strategy or investment advice. Cointegration in a selected historical window does not establish future stability or profitability. The basic backtest omits borrow availability and fees, market impact, taxes, and realistic order execution. Avoid choosing parameters based on the same period used to report results; use a chronological out-of-sample period before drawing conclusions.

Market data are retrieved through the community-maintained yfinance package from Yahoo Finance. Availability and terms may change. See the [yfinance download documentation](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html).
