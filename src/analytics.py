"""Cointegration, hedge-ratio, and spread analytics."""
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pykalman import KalmanFilter
from statsmodels.tsa.stattools import adfuller, coint
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

def dynamic_spread(
    prices: pd.DataFrame,
    state: pd.DataFrame,
    dependent: str = PAIR_TICKERS[0],
    independent: str = PAIR_TICKERS[1],
) -> pd.Series:
    """Calculate residuals with time-varying Kalman parameters for a ticker pair."""
    frame = prices.join(state, how="inner")
    result = (
        np.log(frame[dependent])
        - frame["intercept"]
        - frame["hedge_ratio"] * np.log(frame[independent])
    )
    return result.rename("dynamic_spread")


def _align_pair(s1: pd.Series, s2: pd.Series) -> pd.DataFrame:
    """Align two numeric series by index and remove invalid observations."""
    pair = pd.concat(
        [pd.Series(s1, name="y"), pd.Series(s2, name="x")],
        axis=1,
        join="inner",
    )
    pair = pair.replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 30:
        raise ValueError("At least 30 aligned observations are required.")
    return pair.astype(float)


def test_cointegration(s1: pd.Series, s2: pd.Series) -> dict:
    """Run Engle–Granger and ADF-on-residuals tests for two aligned series.

    The Engle–Granger p-value and its critical values are the appropriate
    residual-based cointegration inference. The separate ADF residual test is
    also reported as a requested diagnostic; its ordinary p-value does not
    account for the generated-regressor step.
    """
    pair = _align_pair(s1, s2)
    eg_stat, eg_pvalue, eg_critical = coint(
        pair["y"], pair["x"], trend="c", autolag="aic"
    )

    regression = sm.OLS(
        pair["y"],
        sm.add_constant(pair["x"], has_constant="add"),
    ).fit()
    residuals = regression.resid
    adf_stat, adf_pvalue, _, _, adf_critical, _ = adfuller(
        residuals, autolag="AIC", regression="c"
    )

    print(f"Engle–Granger statistic: {eg_stat:.4f}")
    print(f"Engle–Granger p-value: {eg_pvalue:.6g}")
    print("Engle–Granger critical values:")
    for level, value in zip(("1%", "5%", "10%"), eg_critical):
        print(f"  {level}: {value:.4f}")
    print(f"ADF residual statistic: {adf_stat:.4f}")
    print(f"ADF residual p-value (diagnostic): {adf_pvalue:.6g}")
    print("ADF residual critical values:")
    for level, value in adf_critical.items():
        print(f"  {level}: {value:.4f}")

    return {
        "engle_granger_statistic": float(eg_stat),
        "engle_granger_pvalue": float(eg_pvalue),
        "engle_granger_critical_values": {
            level: float(value)
            for level, value in zip(("1%", "5%", "10%"), eg_critical)
        },
        "adf_residual_statistic": float(adf_stat),
        "adf_residual_pvalue": float(adf_pvalue),
        "adf_residual_critical_values": {
            level: float(value) for level, value in adf_critical.items()
        },
        "observations": int(len(pair)),
    }


def estimate_kalman_hedge_ratio(
    s1: pd.Series,
    s2: pd.Series,
    rolling_window: int = 60,
    observation_covariance: float = 1e-3,
    transition_covariance: float = 1e-5,
) -> tuple[pd.Series, pd.Series]:
    """Estimate dynamic beta and return it with the spread rolling z-score.

    State vector is [alpha_t, beta_t] for y_t = alpha_t + beta_t*x_t + eps_t.
    The returned tuple is (beta_series, spread_zscore). The dynamic intercept
    is estimated at every time step and used to calculate the spread.
    """
    if rolling_window < 2:
        raise ValueError("rolling_window must be at least 2.")
    pair = _align_pair(s1, s2)
    y = pair["y"].to_numpy(dtype=float)
    x = pair["x"].to_numpy(dtype=float)
    observation_matrices = np.column_stack([np.ones(len(x)), x]).reshape(-1, 1, 2)

    kalman = KalmanFilter(
        transition_matrices=np.eye(2),
        observation_matrices=observation_matrices,
        initial_state_mean=np.array([0.0, 1.0]),
        initial_state_covariance=np.eye(2),
        transition_covariance=np.eye(2) * transition_covariance,
        observation_covariance=observation_covariance,
    )
    states, _ = kalman.filter(y)
    alpha = pd.Series(states[:, 0], index=pair.index, name="intercept")
    beta = pd.Series(states[:, 1], index=pair.index, name="hedge_ratio")
    spread = (pair["y"] - alpha - beta * pair["x"]).rename("spread")

    mean = spread.rolling(rolling_window, min_periods=rolling_window).mean()
    std = spread.rolling(rolling_window, min_periods=rolling_window).std(ddof=1)
    spread_zscore = ((spread - mean) / std.replace(0, np.nan)).rename("spread_zscore")
    return beta, spread_zscore
