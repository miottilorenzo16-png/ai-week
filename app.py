import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import statsmodels.api as sm
import yfinance as yf
from dotenv import load_dotenv
from statsmodels.tsa.stattools import coint

load_dotenv()

st.set_page_config(page_title="PairLab", page_icon="↔", layout="wide")

st.markdown("""
<style>
.stApp { background: #0b1020; color: #eef3ff; }
[data-testid="stMetric"] { background: #131d30; border: 1px solid #26354d; padding: 16px; border-radius: 12px; }
[data-testid="stMetricLabel"] { color: #a4b2c8; }
h1, h2, h3 { letter-spacing: -0.03em; }
.block-container { max-width: 1250px; padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

st.title("PairLab")
st.subheader("Do BNP Paribas and Société Générale move together?")
st.write("Explore a simple pairs-trading research workflow using daily adjusted prices. The analysis estimates a hedge ratio, tests for cointegration, and illustrates z-score entry and exit rules.")

st.info("Research prototype only. A statistical relationship in past prices does not guarantee that it will continue or that a strategy will make money.")

with st.sidebar:
    st.header("Settings")
    period_label = st.selectbox("Price history", ["1 year", "2 years", "5 years"], index=1)
    periods = {"1 year": "1y", "2 years": "2y", "5 years": "5y"}
    period = periods[period_label]
    lookback = st.slider("Rolling window (trading days)", 20, 120, 60, 5)
    entry_z = st.slider("Entry threshold |z|", 1.0, 3.0, 2.0, 0.1)
    exit_z = st.slider("Exit threshold |z|", 0.0, 1.5, 0.5, 0.1)
    cost_bps = st.slider("Estimated cost per turnover (bps)", 0, 30, 5, 1)
    st.caption("The backtest uses close-to-close returns and the previous day's signal to avoid using today's closing price to trade today's return.")

@st.cache_data(ttl=900, show_spinner=False)
def load_prices(selected_period):
    symbols = {"BNP Paribas": "BNP.PA", "Société Générale": "GLE.PA"}
    series = []
    for label, ticker in symbols.items():
        frame = yf.Ticker(ticker).history(period=selected_period, interval="1d", auto_adjust=True, actions=False)
        if frame.empty or "Close" not in frame:
            raise ValueError("No price history was returned for " + ticker)
        close = frame["Close"].rename(label)
        close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
        series.append(close)
    prices = pd.concat(series, axis=1, join="inner").dropna()
    prices = prices[~prices.index.duplicated(keep="last")].sort_index()
    if len(prices) < 100:
        raise ValueError("Not enough overlapping daily observations for a meaningful demonstration.")
    return prices

try:
    with st.spinner("Loading adjusted daily prices…"):
        prices = load_prices(period)
except Exception as exc:
    st.error("Could not load market data. Please retry in a moment. Data provider message: " + str(exc))
    st.stop()

log_prices = np.log(prices)
x = sm.add_constant(log_prices["Société Générale"])
fit = sm.OLS(log_prices["BNP Paribas"], x).fit()
alpha = float(fit.params["const"])
beta = float(fit.params["Société Générale"])
spread = log_prices["BNP Paribas"] - alpha - beta * log_prices["Société Générale"]

eg_stat, eg_pvalue, eg_critical = coint(log_prices["BNP Paribas"], log_prices["Société Générale"], trend="c", autolag="aic")
rolling_mean = spread.rolling(lookback, min_periods=lookback).mean()
rolling_std = spread.rolling(lookback, min_periods=lookback).std()
zscore = (spread - rolling_mean) / rolling_std.replace(0, np.nan)

returns = prices.pct_change().fillna(0)
positions = pd.Series(0.0, index=prices.index)
position = 0.0
for i in range(1, len(prices)):
    previous_z = zscore.iloc[i - 1]
    if np.isfinite(previous_z):
        if position == 0 and previous_z >= entry_z:
            position = -1.0
        elif position == 0 and previous_z <= -entry_z:
            position = 1.0
        elif position == 1.0 and previous_z >= entry_z:
            position = -1.0
        elif position == -1.0 and previous_z <= -entry_z:
            position = 1.0
        elif position != 0 and abs(previous_z) <= exit_z:
            position = 0.0
    positions.iloc[i] = position

pair_return = (returns["BNP Paribas"] - beta * returns["Société Générale"]) / (1 + abs(beta))
turnover = positions.diff().abs().fillna(positions.abs())
strategy_returns = (positions * pair_return - turnover * (cost_bps / 10000)).dropna()
equity = (1 + strategy_returns).cumprod()
drawdown = equity / equity.cummax() - 1
sharpe = (strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)) if strategy_returns.std() > 0 else 0.0
cumulative_return = (equity.iloc[-1] - 1) * 100
max_drawdown = drawdown.min() * 100
current_z = float(zscore.dropna().iloc[-1]) if zscore.notna().any() else float("nan")
last_date = prices.index[-1].date().isoformat()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Engle–Granger p-value", f"{eg_pvalue:.4f}", help="Residual-based cointegration test for this selected historical window. Lower values indicate stronger evidence against no cointegration.")
col2.metric("Estimated hedge ratio", f"{beta:.2f}", help="OLS coefficient from log(BNP) regressed on log(Société Générale).")
col3.metric("Latest spread z-score", f"{current_z:.2f}" if np.isfinite(current_z) else "n/a", help="Distance from the rolling spread mean, in rolling standard deviations.")
col4.metric("Illustrative strategy return", f"{cumulative_return:.1f}%", help="Historical paper backtest with a simple z-score rule and the chosen estimated transaction cost.")

c1, c2, c3 = st.columns(3)
c1.metric("Annualized Sharpe (illustrative)", f"{sharpe:.2f}")
c2.metric("Maximum drawdown", f"{max_drawdown:.1f}%")
c3.metric("Observations", f"{len(prices)} daily closes", help="Last data date: " + last_date)

if eg_pvalue < 0.05:
    st.success("The selected sample shows evidence of cointegration at the 5% level. This is an in-sample statistical result, not proof that the relationship will persist.")
else:
    st.warning("The selected sample does not show strong evidence of cointegration at the 5% level. Treat the spread signal as exploratory, not as a validated pairs trade.")

st.caption("Market data: Yahoo Finance daily adjusted closes via yfinance · BNP.PA and GLE.PA · last close in dataset: " + last_date)

st.subheader("How the pair behaves")
normalized = prices / prices.iloc[0] * 100
fig_prices = px.line(normalized, x=normalized.index, y=normalized.columns, labels={"value": "Indexed price (start = 100)", "index": "Date", "variable": "Share"})
fig_prices.update_layout(template="plotly_dark", height=350, margin=dict(l=10, r=10, t=25, b=10), legend_title="")
st.plotly_chart(fig_prices, use_container_width=True)

left, right = st.columns(2)
with left:
    spread_frame = pd.DataFrame({"Spread z-score": zscore})
    fig_z = go.Figure()
    fig_z.add_trace(go.Scatter(x=spread_frame.index, y=spread_frame["Spread z-score"], name="Spread z-score", line=dict(color="#75d9e8", width=2)))
    fig_z.add_hline(y=entry_z, line_dash="dash", line_color="#ff8492", annotation_text="Short spread threshold")
    fig_z.add_hline(y=-entry_z, line_dash="dash", line_color="#77d6aa", annotation_text="Long spread threshold")
    fig_z.add_hline(y=exit_z, line_dash="dot", line_color="#a4b2c8")
    fig_z.add_hline(y=-exit_z, line_dash="dot", line_color="#a4b2c8")
    fig_z.add_hline(y=0, line_color="#53627b")
    fig_z.update_layout(template="plotly_dark", title="Spread z-score and rule levels", height=360, margin=dict(l=10, r=10, t=40, b=10), yaxis_title="Standard deviations")
    st.plotly_chart(fig_z, use_container_width=True)
with right:
    fig_eq = px.line(x=equity.index, y=(equity - 1) * 100, labels={"x": "Date", "y": "Cumulative return (%)"})
    fig_eq.update_traces(line_color="#77d6aa")
    fig_eq.update_layout(template="plotly_dark", title="Paper strategy · cumulative return", height=360, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_eq, use_container_width=True)

with st.expander("How to read the signals"):
    st.markdown("""
    - The app regresses the **log price** of BNP Paribas on the **log price** of Société Générale to estimate a hedge ratio.
    - The spread is the regression residual. The rolling z-score shows how far it is from its recent mean.
    - If z is above the entry threshold, the demo shorts the spread (short BNP, long Société Générale); below the negative threshold, it goes long the spread.
    - The position exits when the z-score returns inside the selected exit band. Signals use the previous day's close; returns are close-to-close.
    - The Engle–Granger test asks whether the historical spread is statistically consistent with mean reversion. It does not test whether the strategy is profitable.
    - The backtest is intentionally simple: it omits taxes, borrow fees, market impact, slippage beyond the selected cost, and changing hedge ratios.
    """)

st.subheader("AI research note")
api_key = os.getenv("OPENAI_API_KEY", "").strip()
if st.button("Explain this result with OpenAI", disabled=not api_key):
    from openai import OpenAI
    prompt = (
        "Explain these quantitative research results in plain English for a beginner. "
        "Be concise and cautious. Distinguish statistical evidence from profitability; "
        "do not give investment advice or claim future performance. "
        f"Pair: BNP Paribas and Société Générale. Period: {period_label}. "
        f"Observations: {len(prices)}. Engle-Granger p-value: {eg_pvalue:.5f}. "
        f"Hedge ratio: {beta:.3f}. Latest z-score: {current_z:.3f}. "
        f"Paper strategy cumulative return after estimated costs: {cumulative_return:.2f}%. "
        f"Annualized Sharpe: {sharpe:.2f}. Maximum drawdown: {max_drawdown:.2f}%. "
        f"Entry threshold: {entry_z:.1f}; exit threshold: {exit_z:.1f}; estimated cost: {cost_bps} bps."
    )
    try:
        with st.spinner("Asking OpenAI to summarize the calculations…"):
            client = OpenAI(api_key=api_key)
            response = client.responses.create(
                model=os.getenv("OPENAI_MODEL", "gpt-6-luna"),
                input=prompt,
                max_output_tokens=350,
            )
        st.write(response.output_text)
    except Exception as exc:
        st.error("OpenAI summary failed: " + str(exc))
elif not api_key:
    st.caption("Optional: add an OpenAI API key to a local .env file to enable the plain-language research note. The quant calculations work without it.")

st.caption("Educational prototype · No trade execution · Not investment advice")
