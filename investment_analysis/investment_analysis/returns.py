"""
returns.py
----------
Simple and log return calculations at the asset and portfolio level.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily simple (percentage) returns, first row (NaN) dropped."""
    return prices.pct_change().dropna()


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns, first row (NaN) dropped."""
    return np.log(prices / prices.shift(1)).dropna()


def portfolio_returns(asset_returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """Weighted combination of asset returns into a single portfolio return series."""
    return asset_returns.dot(weights)


def annualize_simple_return(daily_return_mean: float, periods: int = TRADING_DAYS_PER_YEAR) -> float:
    """Compound a mean daily simple return up to an annual figure."""
    return (1 + daily_return_mean) ** periods - 1


def annualize_log_return(daily_log_return_mean: float, periods: int = TRADING_DAYS_PER_YEAR) -> float:
    """Log returns annualize by simple multiplication (they're additive)."""
    return daily_log_return_mean * periods


def annualize_volatility(daily_std: float, periods: int = TRADING_DAYS_PER_YEAR) -> float:
    return daily_std * np.sqrt(periods)
