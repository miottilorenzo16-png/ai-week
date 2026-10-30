# PairLab: BNP Paribas × Société Générale

A small quantitative-finance hackathon prototype for exploring a pairs-trading idea with two Paris-listed banks.

## What it does

- Downloads real daily adjusted closing prices for BNP Paribas (BNP.PA) and Société Générale (GLE.PA).
- Estimates a hedge ratio from log prices and runs an Engle–Granger cointegration test.
- Calculates a rolling spread z-score and visualizes simple entry/exit rules.
- Runs a small paper backtest with a user-set transaction-cost assumption.
- Optionally uses OpenAI to explain the displayed calculations in plain language.

The model only explains the computed results. The price data, hedge ratio, test, signals and backtest are calculated by code.

## Run it on macOS

1. Install Python 3.10 or newer.
2. Download the repository from GitHub with **Code → Download ZIP**, then open Terminal in the unzipped `ai-week` folder.
3. Create an environment and install the packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

4. Start the app:

```bash
streamlit run app.py
```

The app opens in your browser. Price data works without an OpenAI key.

## Optional OpenAI explanation

To enable the **Explain this result with OpenAI** button:

1. Copy `.env.example` to `.env`.
2. Put your API key after `OPENAI_API_KEY=`.
3. Keep `.env` private. It is excluded by `.gitignore`; never paste the key into GitHub or chat.
4. Restart the app.

The optional summary uses the Responses API and GPT-6 Luna. API access and billing depend on the account attached to the key.

## Method and caveats

Prices are fetched using the community-maintained yfinance package from Yahoo Finance. The app uses adjusted closes, aligns both series to common trading dates, regresses log BNP price on log Société Générale price, and applies the Engle–Granger test. A rolling z-score then drives an illustrative mean-reversion rule. Signals use the previous close; the backtest includes a simple estimated transaction cost.

This is an educational prototype, not a validated strategy or investment advice. A low cointegration p-value in one historical window does not guarantee the relationship will persist or that the backtest will work out of sample. The simple backtest omits borrow fees, market impact, taxes, and several execution details. Yahoo/yfinance availability may vary.

## Sources

- [BNP Paribas investor information](https://invest.bnpparibas/en/bnp-paribas-share)
- [Société Générale investor information](https://investors.societegenerale.com/en/financial-non-financial-information/share-performance/share-capital)
- [yfinance download documentation](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html)
- [OpenAI API quickstart](https://developers.openai.com/api/docs/quickstart)
