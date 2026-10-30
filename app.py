"""Streamlit dashboard for the pairs-trading research pipeline."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.analytics import (
    dynamic_spread,
    fit_hedge_ratio,
    kalman_hedge_ratio,
    rolling_zscore,
)
from src.backtest import PairsBacktester
from src.data import download_adjusted_closes
from src.screener import CAC40_TICKERS, fetch_universe_data, scan_cointegrated_pairs


st.set_page_config(page_title="Pairs Trading Lab", page_icon="📈", layout="wide")
st.title("Pairs Trading Lab")
st.caption(
    "Historical research dashboard for BNP Paribas, Société Générale, and other "
    "user-selected pairs. Results are hypothetical and are not investment advice."
)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_pair_prices(ticker_1: str, ticker_2: str, start_date: str, end_date: str):
    """Cache adjusted daily prices for the selected pair."""
    return download_adjusted_closes(
        start=start_date,
        end=end_date,
        tickers=(ticker_1, ticker_2),
    )


@st.cache_data(ttl=3600, show_spinner=False)
def cached_universe_prices(start_date: str):
    """Cache CAC 40 close prices used by the cointegration screener."""
    return fetch_universe_data(CAC40_TICKERS, start_date=start_date)


with st.sidebar:
    st.header("Strategy controls")
    ticker_1 = st.text_input("ticker_1", value="BNP.PA").strip().upper()
    ticker_2 = st.text_input("ticker_2", value="GLE.PA").strip().upper()
    start_date = st.date_input("Start date", value=date(2021, 1, 1))
    entry_z = st.slider(
        "Entry z-score",
        min_value=0.5,
        max_value=4.0,
        value=1.5,
        step=0.1,
    )
    exit_z = st.slider(
        "Exit z-score",
        min_value=0.0,
        max_value=min(1.0, round(entry_z - 0.1, 1)),
        value=0.0,
        step=0.1,
    )
    stop_min = round(entry_z + 0.1, 1)
    stop_default = min(6.0, max(3.0, stop_min))
    stop_loss_z = st.slider(
        "Stop-loss |z|",
        min_value=stop_min,
        max_value=6.0,
        value=stop_default,
        step=0.1,
    )
    hedge_method = st.selectbox(
        "Hedge ratio method",
        options=["OLS", "Kalman Filter"],
        index=1,
    )
    st.caption("Backtest transaction fee: 10 bps per unit of turnover.")


today = date.today()
if start_date >= today:
    st.error("Choose a start date before today.")
    st.stop()
if not ticker_1 or not ticker_2:
    st.error("Enter both Yahoo Finance ticker symbols.")
    st.stop()
if ticker_1 == ticker_2:
    st.error("Choose two different ticker symbols.")
    st.stop()

end_date = today.isoformat()
start_date_str = start_date.isoformat()

tab_backtest, tab_screener = st.tabs(["Backtest & Signals", "CAC 40 Screener"])

with tab_backtest:
    st.subheader(f"{ticker_1} vs. {ticker_2}")
    with st.spinner("Loading adjusted prices and calculating the backtest..."):
        try:
            prices = cached_pair_prices(
                ticker_1, ticker_2, start_date_str, end_date
            )
            if len(prices) < 90:
                st.warning(
                    f"Only {len(prices)} overlapping price observations were returned. "
                    "The rolling z-score and backtest may be unreliable."
                )

            if hedge_method == "OLS":
                fitted = fit_hedge_ratio(
                    prices, dependent=ticker_1, independent=ticker_2
                )
                spread = fitted["spread"]
                beta = pd.Series(
                    fitted["beta"], index=spread.index, name="hedge_ratio"
                )
            else:
                kalman_state = kalman_hedge_ratio(
                    prices, dependent=ticker_1, independent=ticker_2
                )
                spread = dynamic_spread(
                    prices,
                    kalman_state,
                    dependent=ticker_1,
                    independent=ticker_2,
                )
                beta = kalman_state["hedge_ratio"]

            zscore = rolling_zscore(spread, window=60)
            backtester = PairsBacktester(
                spread_zscore=zscore,
                prices=prices[[ticker_1, ticker_2]],
                hedge_ratio=beta,
                entry_threshold=entry_z,
                exit_threshold=exit_z,
                transaction_fee=0.001,
                stop_loss_z=stop_loss_z,
            )
            summary, pnl = backtester.run()
        except Exception as exc:
            st.error(f"Could not run the pair backtest: {exc}")
            st.stop()

    metric_cols = st.columns(5)
    metric_cols[0].metric("Cumulative PnL", f"{summary['cumulative_pnl']:.2%}")
    metric_cols[1].metric("Annualized Sharpe", f"{summary['annualized_sharpe']:.2f}")
    metric_cols[2].metric("Win rate", f"{summary['win_rate']:.1%}")
    metric_cols[3].metric("Max drawdown", f"{summary['max_drawdown']:.2%}")
    metric_cols[4].metric("Stop-loss exits", str(summary["stop_loss_triggers"]))

    normalized_prices = prices.div(prices.iloc[0]).mul(100)
    buy_hold = (normalized_prices.div(100).mean(axis=1) - 1).reindex(pnl.index).ffill()

    positions = pnl["position"]
    previous_positions = positions.shift(1).fillna(0.0)
    long_entries = (positions == 1.0) & (previous_positions == 0.0)
    short_entries = (positions == -1.0) & (previous_positions == 0.0)
    exits = (positions == 0.0) & (previous_positions != 0.0)
    stop_exits = pnl["stop_loss_trigger"].astype(bool)
    normal_exits = exits & ~stop_exits

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=[0.28, 0.38, 0.34],
        subplot_titles=(
            "Normalized adjusted prices (start = 100)",
            "Spread z-score and trade signals",
            "Cumulative net PnL vs. 50/50 buy-and-hold",
        ),
    )
    for ticker, label in (
        (ticker_1, ticker_1),
        (ticker_2, ticker_2),
    ):
        fig.add_trace(
            go.Scatter(
                x=normalized_prices.index,
                y=normalized_prices[ticker],
                mode="lines",
                name=label,
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}<extra>%{fullData.name}</extra>",
            ),
            row=1,
            col=1,
        )

    z_plot = pnl["zscore"]
    fig.add_trace(
        go.Scatter(
            x=z_plot.index,
            y=z_plot,
            mode="lines",
            name="Spread z-score",
            line={"color": "#3977b8"},
        ),
        row=2,
        col=1,
    )
    signal_specs = [
        (long_entries, "Long spread entry", "triangle-up", "#159447"),
        (short_entries, "Short spread entry", "triangle-down", "#d94b45"),
        (normal_exits, "Mean-reversion exit", "x", "#333333"),
        (stop_exits, "Stop-loss exit", "x", "#ff7f0e"),
    ]
    for mask, label, symbol, color in signal_specs:
        if mask.any():
            fig.add_trace(
                go.Scatter(
                    x=z_plot.index[mask],
                    y=z_plot.loc[mask],
                    mode="markers",
                    name=label,
                    marker={"symbol": symbol, "size": 10, "color": color},
                ),
                row=2,
                col=1,
            )

    for level, label, color, dash in (
        (entry_z, f"Short entry +{entry_z:.1f}", "#c43c39", "dash"),
        (-entry_z, f"Long entry −{entry_z:.1f}", "#23834d", "dash"),
        (exit_z, f"Exit +{exit_z:.1f}", "#555555", "dot"),
        (stop_loss_z, f"Short stop +{stop_loss_z:.1f}", "#f28e2b", "dashdot"),
        (-stop_loss_z, f"Long stop −{stop_loss_z:.1f}", "#f28e2b", "dashdot"),
    ):
        fig.add_hline(
            y=level,
            line_color=color,
            line_dash=dash,
            line_width=1,
            annotation_text=label,
            annotation_position="top left",
            row=2,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=pnl.index,
            y=pnl["cumulative_pnl"] * 100,
            mode="lines",
            name="Pairs strategy (net)",
            line={"color": "#7a3db8", "width": 2},
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=buy_hold.index,
            y=buy_hold * 100,
            mode="lines",
            name="50/50 buy-and-hold",
            line={"color": "#9b6b3d", "width": 1.5},
        ),
        row=3,
        col=1,
    )
    fig.update_yaxes(title_text="Index level", row=1, col=1)
    fig.update_yaxes(title_text="Z-score", row=2, col=1)
    fig.update_yaxes(title_text="Return (%)", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_layout(
        height=1000,
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        margin={"t": 100, "b": 30},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "The 50/50 buy-and-hold line is long-only and is not a like-for-like "
        "market-neutral benchmark. Historical results are not forecasts."
    )


with tab_screener:
    st.subheader("CAC 40 cointegration candidates")
    st.write(
        "Download the current CAC 40 universe from the selected start date and "
        "test each available pair with the Engle–Granger method."
    )
    p_value_threshold = st.number_input(
        "Maximum p-value",
        min_value=0.001,
        max_value=0.25,
        value=0.05,
        step=0.005,
        format="%.3f",
    )

    if st.button("Run CAC 40 scan", type="primary"):
        progress = st.progress(0, text="Downloading CAC 40 prices…")
        try:
            universe_prices = cached_universe_prices(start_date_str)
            progress.progress(10, text="Prices loaded. Testing stock pairs…")

            last_percent = [-1]

            def update_scan_progress(fraction: float) -> None:
                percent = min(100, int(fraction * 100))
                if percent > last_percent[0]:
                    last_percent[0] = percent
                    progress.progress(
                        10 + int(fraction * 90),
                        text=f"Testing cointegration pairs… {percent}%",
                    )

            results = scan_cointegrated_pairs(
                universe_prices,
                p_value_threshold=float(p_value_threshold),
                progress_callback=update_scan_progress,
            )
            progress.progress(100, text="Scan complete.")
            if results.empty:
                st.info("No pairs passed this p-value threshold.")
            else:
                st.success(f"Found {len(results)} pairs below the p-value threshold.")
                st.dataframe(results, use_container_width=True, hide_index=True)
        except Exception as exc:
            progress.empty()
            st.error(f"CAC 40 scan failed: {exc}")
