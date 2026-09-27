"""
visualization.py
-----------------
Plotly charts: individual/combined price lines, log return lines, and the
Markowitz efficient frontier plot (simulated cloud + frontier + optimal points).
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

SEQUENTIAL_BLUE = [
    [0.00, "#cde2fb"], [0.17, "#9ec5f4"], [0.33, "#5598e7"],
    [0.50, "#2a78d6"], [0.67, "#1c5cab"], [0.83, "#104281"], [1.00, "#0d366b"],
]


def plot_prices(prices: pd.DataFrame, title: str = "Adj Closing Price of Stocks") -> go.Figure:
    """Combined line chart of all tickers' prices."""
    return px.line(prices, title=title)


def plot_single_ticker_price(prices: pd.DataFrame, ticker: str, color: str = "blue") -> go.Figure:
    return px.line(
        prices, x=prices.index, y=prices[ticker],
        title=f"Adj Closing Price of {ticker}",
        color_discrete_sequence=[color],
    )


def plot_log_returns(log_return_df: pd.DataFrame, title: str = "Log Return of Stocks") -> go.Figure:
    return px.line(log_return_df, title=title)


def plot_efficient_frontier(
    simulated_portfolios_df: pd.DataFrame,
    efficient_frontier_df: pd.DataFrame,
    min_vol_point: tuple[float, float],
    title: str = "Markowitz Efficient Frontier",
) -> go.Figure:
    """
    Parameters
    ----------
    simulated_portfolios_df : columns ["Return", "Volatility", "Sharpe"]
    efficient_frontier_df : columns ["Return", "Volatility"]
    min_vol_point : (volatility, return) tuple
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=simulated_portfolios_df["Volatility"],
        y=simulated_portfolios_df["Return"],
        mode="markers",
        marker=dict(
            size=5,
            color=simulated_portfolios_df["Sharpe"],
            colorscale=SEQUENTIAL_BLUE,
            colorbar=dict(title="Sharpe<br>Ratio"),
            opacity=0.55,
            line=dict(width=0),
        ),
        name="Simulated Portfolios",
        hovertemplate="Volatility: %{x:.2%}<br>Return: %{y:.2%}<br>Sharpe: %{marker.color:.2f}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=efficient_frontier_df["Volatility"],
        y=efficient_frontier_df["Return"],
        mode="lines",
        line=dict(color="#0b0b0b", width=2.5),
        name="Efficient Frontier",
        hovertemplate="Volatility: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=[min_vol_point[0]], y=[min_vol_point[1]],
        mode="markers+text",
        marker=dict(size=16, symbol="star", color="#1baf7a", line=dict(width=1, color="#0b0b0b")),
        text=["Min Volatility"], textposition="top center",
        name="Min Volatility Portfolio",
        hovertemplate="Volatility: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>",
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Annual Volatility (Risk)",
        yaxis_title="Expected Annual Return",
        xaxis_tickformat=".1%",
        yaxis_tickformat=".1%",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        width=900,
        height=600,
    )
    return fig
