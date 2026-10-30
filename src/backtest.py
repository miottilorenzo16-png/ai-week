"""Close-to-close paper backtest for a spread mean-reversion rule."""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Union
from .data import PAIR_TICKERS

def generate_positions(zscore: pd.Series, entry_z: float = 2.0, exit_z: float = 0.5) -> pd.Series:
    """Return +1 long-spread, -1 short-spread positions using prior-close signals."""
    if entry_z <= 0 or exit_z < 0 or exit_z >= entry_z:
        raise ValueError("Require entry_z > exit_z >= 0.")
    positions = pd.Series(0.0, index=zscore.index, name="position")
    position = 0.0
    for i in range(1, len(zscore)):
        observed = zscore.iloc[i - 1]
        if np.isfinite(observed):
            if position == 0:
                if observed >= entry_z:
                    position = -1.0
                elif observed <= -entry_z:
                    position = 1.0
            elif position == 1:
                if observed >= entry_z:
                    position = -1.0
                elif abs(observed) <= exit_z:
                    position = 0.0
            elif position == -1:
                if observed <= -entry_z:
                    position = 1.0
                elif abs(observed) <= exit_z:
                    position = 0.0
        positions.iloc[i] = position
    return positions

def backtest_pair(
    prices: pd.DataFrame,
    zscore: pd.Series,
    hedge_ratio: float,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    transaction_cost_bps: float = 5.0,
) -> pd.DataFrame:
    """Backtest positions with normalized gross exposure and turnover costs."""
    if transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps cannot be negative.")
    aligned = prices[list(PAIR_TICKERS)].dropna()
    positions = generate_positions(zscore.reindex(aligned.index), entry_z, exit_z)
    returns = aligned.pct_change().fillna(0.0)
    spread_return = (returns[PAIR_TICKERS[0]] - hedge_ratio * returns[PAIR_TICKERS[1]]) / (1 + abs(hedge_ratio))
    turnover = positions.diff().abs().fillna(positions.abs())
    costs = turnover * transaction_cost_bps / 10_000
    net = positions * spread_return - costs
    equity = (1 + net).cumprod()
    drawdown = equity / equity.cummax() - 1
    return pd.DataFrame(
        {"position": positions, "spread_return": spread_return, "turnover": turnover,
         "transaction_cost": costs, "net_return": net, "equity": equity, "drawdown": drawdown},
        index=aligned.index,
    )

def performance_summary(result: pd.DataFrame) -> dict[str, float]:
    """Summarize a backtest equity curve; historical results are not forecasts."""
    returns = result["net_return"].dropna()
    equity = result["equity"].dropna()
    if returns.empty or equity.empty:
        raise ValueError("Backtest has no observations.")
    volatility = returns.std(ddof=1)
    sharpe = float(np.sqrt(252) * returns.mean() / volatility) if pd.notna(volatility) and volatility > 0 else float("nan")
    years = max(len(returns) / 252, 1 / 252)
    annualized = float(equity.iloc[-1] ** (1 / years) - 1) if equity.iloc[-1] > 0 else float("nan")
    entries = ((result["position"] != 0) & (result["position"].shift(1).fillna(0) == 0)).sum()
    return {
        "total_return": float(equity.iloc[-1] - 1),
        "annualized_return": annualized,
        "annualized_volatility": float(volatility * np.sqrt(252)),
        "sharpe_ratio": sharpe,
        "max_drawdown": float(result["drawdown"].min()),
        "time_in_market": float((result["position"] != 0).mean()),
        "entry_count": float(entries),
    }


class PairsBacktester:
    """Backtest a market-neutral mean-reversion strategy for a two-stock pair.

    Parameters
    ----------
    spread_zscore:
        Rolling z-score series indexed by date.
    prices:
        Adjusted closing prices for exactly two stocks (or a DataFrame whose
        first two columns are the pair).
    hedge_ratio:
        Fixed beta or a date-indexed beta Series. A dynamic beta is lagged one
        session before it is used in the simulated return.
    entry_threshold:
        Enter long spread below -entry_threshold and short spread above it.
    exit_threshold:
        Exit a long spread when z reaches/exceeds this level; exit a short
        spread when z reaches/falls below it. Set to 0 to exit at zero.
    transaction_fee:
        Fractional cost per unit of turnover. For example, 0.001 is 10 bps.
        Fees are charged on entry and exit.
    """

    def __init__(
        self,
        spread_zscore: pd.Series,
        prices: pd.DataFrame,
        hedge_ratio: Union[float, pd.Series] = 1.0,
        entry_threshold: float = 1.5,
        exit_threshold: float = 0.0,
        transaction_fee: float = 0.001,
    ) -> None:
        if prices.shape[1] < 2:
            raise ValueError("prices must contain at least two ticker columns.")
        if entry_threshold <= 0:
            raise ValueError("entry_threshold must be positive.")
        if exit_threshold < 0 or exit_threshold >= entry_threshold:
            raise ValueError("Require 0 <= exit_threshold < entry_threshold.")
        if transaction_fee < 0:
            raise ValueError("transaction_fee cannot be negative.")

        self.spread_zscore = pd.Series(spread_zscore, name="zscore")
        self.prices = prices.iloc[:, :2].copy()
        self.tickers = list(self.prices.columns)
        self.hedge_ratio = hedge_ratio
        self.entry_threshold = float(entry_threshold)
        self.exit_threshold = float(exit_threshold)
        self.transaction_fee = float(transaction_fee)

    def _prepare_inputs(self) -> tuple[pd.DataFrame, pd.Series]:
        frame = self.prices.copy()
        frame["zscore"] = self.spread_zscore
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna(
            subset=[*self.tickers, "zscore"]
        )

        if isinstance(self.hedge_ratio, pd.Series):
            beta = self.hedge_ratio.rename("hedge_ratio").reindex(frame.index)
            frame["hedge_ratio"] = beta
            frame = frame.dropna(subset=["hedge_ratio"])
            beta = frame["hedge_ratio"]
        else:
            beta = pd.Series(float(self.hedge_ratio), index=frame.index, name="hedge_ratio")

        if frame.empty:
            raise ValueError("No overlapping valid prices and z-scores were supplied.")
        if (frame[self.tickers] <= 0).any().any():
            raise ValueError("Prices must be positive.")
        if (beta.abs() == 0).all():
            raise ValueError("hedge_ratio cannot be zero.")

        return frame, beta.astype(float)

    def _make_positions(self, zscore: pd.Series) -> pd.Series:
        """Create next-session positions from the previous close's z-score."""
        positions = pd.Series(0.0, index=zscore.index, name="position")
        state = 0.0

        for i in range(1, len(zscore)):
            observed_z = zscore.iloc[i - 1]

            if state == 0.0:
                if observed_z <= -self.entry_threshold:
                    state = 1.0
                elif observed_z >= self.entry_threshold:
                    state = -1.0
            elif state == 1.0 and observed_z >= self.exit_threshold:
                state = 0.0
            elif state == -1.0 and observed_z <= self.exit_threshold:
                state = 0.0

            positions.iloc[i] = state

        return positions

    @staticmethod
    def _trade_results(positions: pd.Series, net_returns: pd.Series) -> list[float]:
        """Calculate net return for each completed (or still-open) trade."""
        trades: list[float] = []
        active = False
        trade_equity = 1.0

        for position, daily_return in zip(positions.to_numpy(), net_returns.to_numpy()):
            if not active and position != 0.0:
                active = True
                trade_equity = 1.0 + daily_return
            elif active:
                trade_equity *= 1.0 + daily_return
                if position == 0.0:
                    trades.append(trade_equity - 1.0)
                    active = False
                    trade_equity = 1.0

        if active:
            trades.append(trade_equity - 1.0)
        return trades

    def run(self) -> tuple[dict[str, float], pd.DataFrame]:
        """Run the backtest and return (summary metrics, daily PnL DataFrame)."""
        frame, beta = self._prepare_inputs()
        prices = frame[self.tickers]
        zscore = frame["zscore"]
        positions = self._make_positions(zscore)

        simple_returns = prices.pct_change().fillna(0.0)
        beta_for_return = beta.shift(1)
        if len(beta_for_return):
            beta_for_return.iloc[0] = beta.iloc[0]

        gross_pair_return = (
            simple_returns[self.tickers[0]]
            - beta_for_return * simple_returns[self.tickers[1]]
        ) / (1.0 + beta_for_return.abs())

        turnover = positions.diff().abs().fillna(positions.abs())
        transaction_cost = turnover * self.transaction_fee
        net_return = positions * gross_pair_return - transaction_cost
        equity = (1.0 + net_return).cumprod()
        peak = pd.concat(
            [pd.Series([1.0], index=[prices.index[0] - pd.Timedelta(days=1)]), equity]
        ).cummax().iloc[1:]
        drawdown = equity / peak - 1.0
        cumulative_pnl = equity - 1.0

        pnl = pd.DataFrame(
            {
                "zscore": zscore,
                "hedge_ratio": beta,
                "position": positions,
                "spread_return": gross_pair_return,
                "turnover": turnover,
                "transaction_cost": transaction_cost,
                "net_return": net_return,
                "cumulative_pnl": cumulative_pnl,
                "drawdown": drawdown,
            },
            index=prices.index,
        )

        daily_volatility = net_return.std(ddof=1)
        annualized_sharpe = (
            float(np.sqrt(252) * net_return.mean() / daily_volatility)
            if pd.notna(daily_volatility) and daily_volatility > 0
            else 0.0
        )
        trade_returns = self._trade_results(positions, net_return)
        win_rate = (
            float(sum(trade_return > 0 for trade_return in trade_returns) / len(trade_returns))
            if trade_returns
            else 0.0
        )
        summary = {
            "cumulative_pnl": float(cumulative_pnl.iloc[-1]),
            "annualized_sharpe": annualized_sharpe,
            "max_drawdown": float(drawdown.min()),
            "win_rate": win_rate,
            "total_trades": int(len(trade_returns)),
        }
        return summary, pnl
