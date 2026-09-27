"""
main.py
-------
Runs the full investment analysis end-to-end using the reusable
`investment_analysis` package:

  1. Read portfolio tickers from portfolio.xlsx, download their prices
     + benchmark + risk-free rate
  2. Compute simple/log returns at the asset level
  3. Run Markowitz optimization (min-vol, efficient frontier);
     portfolio weights are the minimum volatility portfolio
  4. Compute portfolio-level returns using those weights
  5. Compute performance & risk metrics (alpha, beta, Sharpe, Sortino,
     Calmar, Treynor, max drawdown, VaR, CVaR)
  6. Save charts as HTML files you can open in a browser

Usage:
    python main.py
Edit config.py to change the portfolio file path, dates, etc.
Edit portfolio.xlsx (a single "Ticker" column) to change portfolio holdings.
"""

from __future__ import annotations

import numpy as np

from config import AnalysisConfig
from investment_analysis import data, returns, risk_metrics, optimization, visualization


def run(cfg: AnalysisConfig) -> None:
    # 1. Portfolio tickers (from spreadsheet) + data
    tickers = data.load_tickers_from_excel(cfg.portfolio_file)
    print(f"Loaded tickers from {cfg.portfolio_file}: {tickers}")

    print(f"Downloading prices for {tickers}...")
    prices = data.download_prices(tickers, cfg.start_date, cfg.end_date)
    benchmark_prices = data.download_benchmark(cfg.benchmark_ticker, cfg.start_date, cfg.end_date)
    risk_free_rate = data.download_risk_free_rate(cfg.risk_free_ticker)
    print(f"Risk-free rate (annualized): {risk_free_rate:.4%}")

    # 2. Returns (asset-level)
    daily_returns = returns.simple_returns(prices)
    log_return_df = returns.log_returns(prices)
    benchmark_daily_returns = benchmark_prices.pct_change().dropna()

    # 3. Markowitz optimization - portfolio weights are the minimum
    # volatility portfolio rather than a fixed/equal weighting.
    print("\n--- Running Markowitz Optimization ---")
    expected_returns, cov_matrix_annual = optimization.annualized_inputs(log_return_df)
    pm = optimization.PortfolioMath(expected_returns, cov_matrix_annual, risk_free_rate)

    sim_df = optimization.monte_carlo_portfolios(pm, cfg.n_monte_carlo_portfolios, seed=cfg.random_seed)
    min_vol = optimization.minimum_volatility_portfolio(pm, tickers)

    target_returns = np.linspace(sim_df["Return"].min(), sim_df["Return"].max(), cfg.n_frontier_points)
    frontier_df = optimization.efficient_frontier(pm, target_returns)

    print("\nMinimum Volatility Portfolio (used as portfolio weights below):")
    print(min_vol.weights)
    print(f"Expected Return: {min_vol.expected_return:.4%}, Volatility: {min_vol.volatility:.4%}")

    weights = min_vol.weights

    # 4. Portfolio-level returns, using the min-vol weights
    portfolio_daily_returns = returns.portfolio_returns(daily_returns, weights)
    portfolio_log_return = returns.portfolio_returns(log_return_df, weights)

    annual_return = returns.annualize_simple_return(portfolio_daily_returns.mean())
    annual_log_return = returns.annualize_log_return(portfolio_log_return.mean())
    annual_volatility = returns.annualize_volatility(np.std(portfolio_daily_returns))

    # 5. Risk / performance metrics
    port_arr = portfolio_daily_returns.to_numpy().flatten()
    bench_arr = benchmark_daily_returns.to_numpy().flatten()

    port_beta = risk_metrics.beta(port_arr, bench_arr)
    port_alpha = risk_metrics.alpha(portfolio_daily_returns, benchmark_daily_returns, port_beta, risk_free_rate)
    downside_dev = risk_metrics.downside_deviation(portfolio_daily_returns)
    max_dd = risk_metrics.max_drawdown(portfolio_daily_returns)

    sharpe = risk_metrics.sharpe_ratio(annual_return, risk_free_rate, annual_volatility)
    sortino = risk_metrics.sortino_ratio(annual_return, risk_free_rate, downside_dev)
    calmar = risk_metrics.calmar_ratio(annual_return, max_dd)
    treynor = risk_metrics.treynor_ratio(annual_return, risk_free_rate, port_beta)

    print("\n--- Portfolio Metrics ---")
    print(f"Annualized Return (simple): {annual_return:.4%}")
    print(f"Annualized Return (log):    {annual_log_return:.4%}")
    print(f"Annualized Volatility:      {annual_volatility:.4%}")
    print(f"Beta:                       {port_beta:.4f}")
    print(f"Alpha (annualized):         {port_alpha:.4%}")
    print(f"Sharpe Ratio:               {sharpe:.4f}")
    print(f"Sortino Ratio:              {sortino:.4f}")
    print(f"Calmar Ratio:               {calmar:.4f}")
    print(f"Treynor Ratio:              {treynor:.4f}")
    print(f"Max Drawdown:               {max_dd:.4%}")

    print("\n--- Value at Risk / Expected Shortfall ---")
    for conf in cfg.var_confidence_levels:
        var = risk_metrics.value_at_risk(portfolio_daily_returns, cfg.portfolio_value, conf)
        print(f"VaR {conf:.0%}: {var:,.2f}")
    cvar_95 = risk_metrics.conditional_value_at_risk(portfolio_daily_returns, cfg.portfolio_value, 0.95)
    print(f"CVaR (Expected Shortfall) 95%: {cvar_95:,.2f}")

    # 6. Charts
    print("\nSaving charts...")
    price_fig = visualization.plot_prices(prices)
    price_fig.write_html("prices.html")

    frontier_fig = visualization.plot_efficient_frontier(
        sim_df,
        frontier_df,
        min_vol_point=(min_vol.volatility, min_vol.expected_return),
    )
    frontier_fig.write_html("efficient_frontier.html")
    print("Saved prices.html and efficient_frontier.html")


if __name__ == "__main__":
    run(AnalysisConfig())
