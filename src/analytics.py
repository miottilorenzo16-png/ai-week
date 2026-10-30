"""Cointegration, hedge-ratio, and spread analytics."""
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pykalman import KalmanFilter
from statsmodels.tsa.stattools import coint
from .data import PAIR_TICKERS

def fit_hedge_ratio(prices: pd.DataFrame, dependent: str = PAIR_TICKERS[0], independent: str = PAIR_TICKERS[1]) -> dict:
    """Fit log(dependent) = alpha + beta * log(independent) + residual."""
    logp = np.log(prices[[dependent, independent]].dropna())
    model = sm.OLS(logp[dependent], sm.add_constant(logp[independent], has_constant="add")).fit()
    alpha, beta = float(model.params["const"]), float(model.params[independent])
    spread = (logp[dependent] - alpha - beta * logp[independent]).rename("spread")
    return {"alpha": alpha, "beta": beta, "spread": spread, "model": model}

def engle_granger_test(prices: pd.DataFrame, first: str = PAIR_TICKERS[0], second: str = PAIR_TICKERS[1]) -> dict:
    """Run the Engle–Granger two-step cointegration test on log prices."""
    logp = np.log(prices[[first, second]].dropna())
    statistic, pvalue, critical = coint(logp[first], logp[second], trend="c", autolag="aic")
    return {
        "statistic": float(statistic),
        "p_value": float(pvalue),
        "critical_values": {"1%": float(critical[0]), "5%": float(critical[1]), "10%": float(critical[2])},
        "observations": int(len(logp)),
    }

def rolling_zscore(spread: pd.Series, window: int = 60) -> pd.Series:
    """Calculate rolling z-scores from trailing observations."""
    if window < 2:
        raise ValueError("window must be at least 2")
    mean = spread.rolling(window, min_periods=window).mean()
    std = spread.rolling(window, min_periods=window).std(ddof=1).replace(0, np.nan)
    return ((spread - mean) / std).rename("zscore")

def kalman_hedge_ratio(
    prices: pd.DataFrame,
    dependent: str = PAIR_TICKERS[0],
    independent: str = PAIR_TICKERS[1],
    observation_covariance: float = 1e-3,
    transition_covariance: float = 1e-5,
) -> pd.DataFrame:
    """Estimate time-varying intercept and hedge ratio with a Kalman filter."""
    logp = np.log(prices[[dependent, independent]].dropna())
    y = logp[dependent].to_numpy(dtype=float)
    x = logp[independent].to_numpy(dtype=float)
    observations = np.column_stack([np.ones(len(x)), x]).reshape(-1, 1, 2)
    model = KalmanFilter(
        transition_matrices=np.eye(2),
        observation_matrices=observations,
        initial_state_mean=np.array([0.0, 1.0]),
        initial_state_covariance=np.eye(2),
        transition_covariance=np.eye(2) * transition_covariance,
        observation_covariance=observation_covariance,
    )
    states, _ = model.filter(y)
    return pd.DataFrame(states, index=logp.index, columns=["intercept", "hedge_ratio"])

def dynamic_spread(prices: pd.DataFrame, state: pd.DataFrame) -> pd.Series:
    """Calculate residuals with the time-varying Kalman parameters."""
    frame = prices.join(state, how="inner")
    result = np.log(frame[PAIR_TICKERS[0]]) - frame["intercept"] - frame["hedge_ratio"] * np.log(frame[PAIR_TICKERS[1]])
    return result.rename("dynamic_spread")
