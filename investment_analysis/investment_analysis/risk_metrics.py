"""
risk_metrics.py
----------------
Risk and risk-adjusted performance metrics for a return series:
beta, alpha, Sharpe, Sortino, Calmar, Treynor, max drawdown, VaR, CVaR.

All functions take plain numpy arrays / pandas Series of *daily* returns
plus an *annualized* risk-free rate, so they can be reused outside the
specific notebook context they came from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .returns import TRADING_DAYS_PER_YEAR


def beta(portfolio_returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
    """Cov(portfolio, benchmark) / Var(benchmark)."""
    cov_matrix = np.cov(portfolio_returns, benchmark_returns)
    return cov_matrix[0, 1] / cov_matrix[1, 1]


def alpha(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    portfolio_beta: float,
    annual_risk_free_rate: float,
    periods: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Jensen's alpha (annualized), via CAPM:
        alpha = Rp - [Rf + beta * (Rm - Rf)]
    computed on daily returns then annualized by multiplying by `periods`.
    """
    daily_rf = annual_risk_free_rate / periods
    daily_alpha = np.mean(portfolio_returns) - (
        daily_rf + portfolio_beta * (np.mean(benchmark_returns) - daily_rf)
    )
    return daily_alpha * periods


def sharpe_ratio(annual_return: float, annual_risk_free_rate: float, annual_volatility: float) -> float:
    """(Rp - Rf) / sigma_p"""
    return (annual_return - annual_risk_free_rate) / annual_volatility


def sortino_ratio(
    annual_return: float,
    annual_risk_free_rate: float,
    daily_downside_deviation: float,
) -> float:
    """
    (Rp - Rf) / downside deviation.

    Note: mirrors the source analysis, which divides by the *daily*
    downside deviation (not annualized) -- kept as-is for consistency;
    annualize `daily_downside_deviation` first if you want a standard
    annualized Sortino ratio.
    """
    return (annual_return - annual_risk_free_rate) / daily_downside_deviation


def downside_deviation(daily_returns: pd.Series) -> float:
    """Standard deviation of negative daily returns only."""
    negative_returns = daily_returns[daily_returns < 0]
    return float(np.std(negative_returns))


def max_drawdown(daily_returns: pd.Series) -> float:
    """Largest peak-to-trough decline in cumulative return."""
    cumulative = (1 + daily_returns).cumprod()
    drawdown = (cumulative.cummax() - cumulative) / cumulative.cummax()
    return float(drawdown.max())


def calmar_ratio(annual_return: float, max_dd: float) -> float:
    """Annual return / max drawdown."""
    return annual_return / max_dd


def treynor_ratio(annual_return: float, annual_risk_free_rate: float, portfolio_beta: float) -> float:
    """(Rp - Rf) / beta"""
    return (annual_return - annual_risk_free_rate) / portfolio_beta


def value_at_risk(daily_returns: pd.Series, portfolio_value: float, confidence: float = 0.95) -> float:
    """
    Historical VaR: the loss (in currency) that returns fall below with
    probability (1 - confidence). E.g. confidence=0.95 -> 5th percentile.
    """
    percentile = (1 - confidence) * 100
    return float(np.percentile(daily_returns, percentile) * portfolio_value)


def conditional_value_at_risk(daily_returns: pd.Series, portfolio_value: float, confidence: float = 0.95) -> float:
    """Expected Shortfall: mean loss given losses worse than the VaR threshold."""
    percentile = (1 - confidence) * 100
    threshold = np.percentile(daily_returns, percentile)
    tail_returns = daily_returns[daily_returns <= threshold]
    return float(tail_returns.mean() * portfolio_value)
