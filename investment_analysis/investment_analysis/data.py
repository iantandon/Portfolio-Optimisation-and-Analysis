"""
data.py
-------
Functions for pulling market data from Yahoo Finance via yfinance.

Kept separate from analysis logic so the rest of the package can be
tested/reused with data from any source (not just yfinance) as long as
it returns the same shape of DataFrame/Series.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import yfinance as yf


def load_tickers_from_excel(path: str, column: str = "Ticker") -> list[str]:
    """
    Read a list of portfolio tickers from an Excel spreadsheet.

    The spreadsheet needs one column (header given by `column`, default
    "Ticker") listing the assets to include; portfolio weights are not
    read from here — they're determined later via Markowitz optimization.

    Parameters
    ----------
    path : str
        Path to an .xlsx file.
    column : str
        Header of the column containing ticker symbols.

    Returns
    -------
    list[str]
        Upper-cased, de-duplicated (order-preserving) ticker symbols.
    """
    df = pd.read_excel(path)
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in {path}. Columns present: {list(df.columns)}")

    tickers = df[column].dropna().astype(str).str.strip().str.upper()
    tickers = tickers[tickers != ""]
    return list(dict.fromkeys(tickers))


def download_prices(
    tickers: list[str],
    start_date: dt.datetime,
    end_date: dt.datetime | None = None,
    price_field: str = "Adj Close",
) -> pd.DataFrame:
    """
    Download adjusted close prices for a list of tickers.

    Parameters
    ----------
    tickers : list[str]
        e.g. ["JPM", "MS", "BAC"]
    start_date, end_date : datetime
        end_date defaults to now if not supplied.
    price_field : str
        Column to extract from the yfinance MultiIndex download
        (typically "Adj Close" or "Close").

    Returns
    -------
    pd.DataFrame
        Columns = tickers, index = trading dates.
    """
    end_date = end_date or dt.datetime.now()
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)
    return data[price_field]


def download_benchmark(
    ticker: str,
    start_date: dt.datetime,
    end_date: dt.datetime | None = None,
    price_field: str = "Adj Close",
) -> pd.Series:
    """Download a single benchmark series (e.g. '^GSPC' for the S&P 500)."""
    end_date = end_date or dt.datetime.now()
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
    return data[price_field]


def download_risk_free_rate(
    ticker: str = "^FVX",
    start_date: dt.datetime | None = None,
    end_date: dt.datetime | None = None,
) -> float:
    """
    Download the most recent annualized risk-free rate implied by a
    treasury-yield ticker (e.g. '^FVX' = 5-year, '^TNX' = 10-year, '^IRX' = 13-week).

    Returns the rate as a decimal (e.g. 0.043 for 4.3%), taken from the
    latest available close, divided by 100.
    """
    start_date = start_date or dt.datetime.now() - dt.timedelta(days=30)
    end_date = end_date or dt.datetime.now()
    data = yf.download(ticker, start=start_date, end=end_date)["Close"]
    latest = float(data[ticker].iloc[-1]) if hasattr(data, "columns") else float(data.iloc[-1])
    return latest / 100
