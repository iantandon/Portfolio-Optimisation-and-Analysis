"""
optimization.py
----------------
Markowitz mean-variance optimization: minimum-volatility portfolio,
maximum-Sharpe (tangency) portfolio, Monte Carlo simulation of the
feasible region, and the efficient frontier.

Uses annualized expected returns and an annualized covariance matrix
(derived from log returns) as inputs, mirroring standard mean-variance
optimization practice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .returns import TRADING_DAYS_PER_YEAR


def annualized_inputs(log_return_df: pd.DataFrame, periods: int = TRADING_DAYS_PER_YEAR) -> tuple[pd.Series, pd.DataFrame]:
    """Annualized expected returns (mean * periods) and covariance matrix (cov * periods)."""
    expected_returns = log_return_df.mean() * periods
    cov_matrix_annual = log_return_df.cov() * periods
    return expected_returns, cov_matrix_annual


class PortfolioMath:
    """
    Bundles expected_returns/cov_matrix/risk_free_rate so the objective
    functions don't need to be re-defined with closures every time
    (the original notebook relied on globals for this).
    """

    def __init__(self, expected_returns: pd.Series, cov_matrix_annual: pd.DataFrame, risk_free_rate: float):
        self.expected_returns = expected_returns
        self.cov_matrix_annual = cov_matrix_annual
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(expected_returns)

    def portfolio_return(self, weights: np.ndarray) -> float:
        return float(np.dot(weights, self.expected_returns))

    def portfolio_volatility(self, weights: np.ndarray) -> float:
        return float(np.sqrt(weights.T @ self.cov_matrix_annual @ weights))

    def neg_sharpe_ratio(self, weights: np.ndarray) -> float:
        return -(self.portfolio_return(weights) - self.risk_free_rate) / self.portfolio_volatility(weights)

    def default_bounds(self) -> tuple:
        """No short-selling: each weight in [0, 1]."""
        return tuple((0, 1) for _ in range(self.n_assets))

    def equal_weight_guess(self) -> np.ndarray:
        return np.array([1 / self.n_assets] * self.n_assets)


def monte_carlo_portfolios(pm: PortfolioMath, n_portfolios: int = 10000, seed: int | None = None) -> pd.DataFrame:
    """
    Simulate random long-only portfolios (weights summing to 1) to
    visualize the feasible region of the risk/return plane.
    """
    rng = np.random.default_rng(seed)
    sim_returns = np.zeros(n_portfolios)
    sim_volatility = np.zeros(n_portfolios)
    sim_sharpe = np.zeros(n_portfolios)

    for i in range(n_portfolios):
        w = rng.random(pm.n_assets)
        w /= w.sum()
        sim_returns[i] = pm.portfolio_return(w)
        sim_volatility[i] = pm.portfolio_volatility(w)
        sim_sharpe[i] = (sim_returns[i] - pm.risk_free_rate) / sim_volatility[i]

    return pd.DataFrame({"Return": sim_returns, "Volatility": sim_volatility, "Sharpe": sim_sharpe})


@dataclass
class OptimalPortfolio:
    weights: pd.Series
    expected_return: float
    volatility: float


def minimum_volatility_portfolio(pm: PortfolioMath, tickers: list[str]) -> OptimalPortfolio:
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)
    result = minimize(
        pm.portfolio_volatility,
        pm.equal_weight_guess(),
        method="SLSQP",
        bounds=pm.default_bounds(),
        constraints=constraints,
    )
    weights = result.x
    return OptimalPortfolio(
        weights=pd.Series(weights, index=tickers, name="Weight"),
        expected_return=pm.portfolio_return(weights),
        volatility=pm.portfolio_volatility(weights),
    )


def maximum_sharpe_portfolio(pm: PortfolioMath, tickers: list[str]) -> OptimalPortfolio:
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)
    result = minimize(
        pm.neg_sharpe_ratio,
        pm.equal_weight_guess(),
        method="SLSQP",
        bounds=pm.default_bounds(),
        constraints=constraints,
    )
    weights = result.x
    return OptimalPortfolio(
        weights=pd.Series(weights, index=tickers, name="Weight"),
        expected_return=pm.portfolio_return(weights),
        volatility=pm.portfolio_volatility(weights),
    )


def efficient_frontier(pm: PortfolioMath, return_range: np.ndarray) -> pd.DataFrame:
    """
    For each target return in `return_range`, find the minimum-volatility
    portfolio that achieves it. Returns a DataFrame of (Return, Volatility)
    tracing the efficient frontier.
    """
    frontier_volatility = []

    for target in return_range:
        constraints = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, target=target: pm.portfolio_return(w) - target},
        )
        result = minimize(
            pm.portfolio_volatility,
            pm.equal_weight_guess(),
            method="SLSQP",
            bounds=pm.default_bounds(),
            constraints=constraints,
        )
        frontier_volatility.append(result.fun)

    return pd.DataFrame({"Return": return_range, "Volatility": np.array(frontier_volatility)})
