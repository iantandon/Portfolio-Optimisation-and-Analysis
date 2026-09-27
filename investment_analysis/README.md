# Investment Analysis Toolkit

A reusable Python package for equity portfolio analysis, refactored out of
the original `Project_1_Investment_Analysis_MarkowitzFrontier.ipynb` notebook.

Instead of one long notebook, the logic is split into focused modules you
can import individually, test, and reuse across different projects/notebooks.

## Project layout

```
investment_analysis/
├── config.py                      # Edit tickers, dates, weights here
├── main.py                        # Runs the full pipeline end-to-end
├── requirements.txt
├── investment_analysis/           # The reusable package
│   ├── __init__.py
│   ├── data.py                    # yfinance price/benchmark/risk-free downloads
│   ├── returns.py                 # simple/log returns, annualization
│   ├── risk_metrics.py            # beta, alpha, Sharpe, Sortino, Calmar,
│   │                               #   Treynor, max drawdown, VaR, CVaR
│   ├── optimization.py            # Markowitz: min-vol, max-Sharpe, frontier
│   └── visualization.py           # Plotly charts
└── notebooks/
    └── analysis.ipynb             # Thin notebook that imports the package
```

## Setup

```bash
pip install -r requirements.txt
```

## Run the full pipeline

```bash
python main.py
```

This downloads data for the tickers in `config.py`, prints all portfolio
metrics to the console, runs the Markowitz optimization, and saves two
charts (`prices.html`, `efficient_frontier.html`) you can open in a browser.

## Customize

Edit `config.py`:

```python
tickers = ["JPM", "MS", "BAC"]
weights = [1/3, 1/3, 1/3]
start_date = datetime(2024, 1, 1)
```

## Use the package pieces individually

```python
from investment_analysis import data, returns, risk_metrics, optimization, visualization
import datetime as dt

prices = data.download_prices(["AAPL", "MSFT"], dt.datetime(2023, 1, 1))
daily_returns = returns.simple_returns(prices)
log_returns = returns.log_returns(prices)

port_returns = returns.portfolio_returns(daily_returns, weights=[0.5, 0.5])
ann_return = returns.annualize_simple_return(port_returns.mean())
ann_vol = returns.annualize_volatility(port_returns.std())

sharpe = risk_metrics.sharpe_ratio(ann_return, annual_risk_free_rate=0.04, annual_volatility=ann_vol)
```

Each module has no dependency on the others' internal state -- functions
take plain DataFrames/Series/arrays in and return plain DataFrames/Series/
floats out, so you can mix and match, swap in a different data source, or
write unit tests against known inputs.

## Notes on choices made while refactoring

- `PortfolioMath` (in `optimization.py`) replaces the notebook's reliance on
  global variables (`expected_returns`, `cov_matrix_annual`, `risk_free_rate`)
  inside closures -- it's a small class bundling those three, so the
  optimizer functions are self-contained and testable.
- `risk_metrics.sortino_ratio` intentionally mirrors the original notebook's
  calculation (dividing by the *daily* downside deviation rather than an
  annualized one). If you want a conventionally annualized Sortino ratio,
  annualize `downside_deviation()` yourself before passing it in
  (multiply by `sqrt(252)`).
- Random Monte Carlo simulation now takes an explicit `seed` argument for
  reproducibility, instead of relying on global numpy random state.
