"""
config.py
---------
Single place to change tickers, date range, and portfolio settings
without touching the analysis code.
"""

from dataclasses import dataclass
import datetime as dt


@dataclass
class AnalysisConfig:
    # Portfolio tickers are read from this spreadsheet (a single "Ticker"
    # column). Weights are not read from config or the spreadsheet - they're
    # solved for via Markowitz max-Sharpe optimization in main.py.
    portfolio_file: str = "portfolio.xlsx"
    start_date: dt.datetime = dt.datetime(2024, 1, 1)
    end_date: dt.datetime | None = None  # None -> now

    benchmark_ticker: str = "^GSPC"
    risk_free_ticker: str = "^FVX"

    portfolio_value: float = 1_000_000
    var_confidence_levels: tuple[float, ...] = (0.90, 0.95, 0.99)

    n_monte_carlo_portfolios: int = 10_000
    n_frontier_points: int = 100
    random_seed: int | None = 42
