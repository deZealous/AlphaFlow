"""
OHLCV ingestion from yfinance.

Downloads adjusted OHLCV data for all configured tickers and saves
a single flat Parquet file to data/raw/ohlcv/ohlcv_raw.parquet.

Column naming convention: {field}_{TICKER}  e.g. Close_AAPL, Volume_MSFT
"""

import yaml
import yfinance as yf
import pandas as pd
from pathlib import Path


def _parse_tickers(config: dict) -> list[str]:
    """Flatten the YAML list-of-comma-strings into a clean list of ticker symbols."""
    tickers = []
    for entry in config["tickers"]:
        tickers.extend([t.strip() for t in entry.split(",")])
    return tickers


def ingest_ohlcv(config: dict, force_refresh: bool = False) -> pd.DataFrame:
    """
    Download OHLCV data for all tickers in config and persist to Parquet.

    Parameters
    ----------
    config : dict
        Parsed phase1_config.yaml.

    Returns
    -------
    pd.DataFrame
        Flat DataFrame with columns {field}_{TICKER} and a DatetimeIndex.
    """
    tickers = _parse_tickers(config)
    start = config["date_range"]["start"]
    end = config["date_range"]["end"]
    out_path = Path(config["raw_data_path"]) / "ohlcv" / "ohlcv_raw.parquet"

    if not force_refresh and out_path.exists():
        print(f"OHLCV cache hit  -> loading from {out_path}")
        return pd.read_parquet(out_path)

    print(f"Downloading OHLCV for {len(tickers)} tickers: {start} to {end}")

    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,   # adjusts for splits and dividends
        progress=False,
        threads=True,
    )

    # yfinance returns a MultiIndex when multiple tickers are requested:
    # level-0 = field (Open/High/Low/Close/Volume)
    # level-1 = ticker
    # Flatten to {field}_{TICKER} so downstream code uses simple column names.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [f"{field}_{ticker}" for field, ticker in raw.columns]

    raw.index = pd.to_datetime(raw.index)
    raw = raw.sort_index()

    # Basic sanity check — warn if coverage is unexpectedly short
    expected_start = pd.Timestamp(start)
    expected_end = pd.Timestamp(end)
    actual_start = raw.index.min()
    actual_end = raw.index.max()

    if actual_start > expected_start + pd.Timedelta(days=10):
        print(f"WARNING: data starts at {actual_start.date()}, expected ~{expected_start.date()}")
    if actual_end < expected_end - pd.Timedelta(days=10):
        print(f"WARNING: data ends at {actual_end.date()}, expected ~{expected_end.date()}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(out_path)

    print(f"OHLCV saved  -> {out_path}")
    print(f"  Shape      : {raw.shape[0]} rows x {raw.shape[1]} columns")
    print(f"  Date range : {actual_start.date()} to {actual_end.date()}")
    print(f"  Null count : {raw.isna().sum().sum()}")

    return raw


if __name__ == "__main__":
    with open("configs/phase1_config.yaml") as f:
        cfg = yaml.safe_load(f)
    ingest_ohlcv(cfg)
