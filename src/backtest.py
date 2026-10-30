"""Close-to-close paper backtest for a spread mean-reversion rule."""
from __future__ import annotations
import numpy as np
import pandas as pd
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
