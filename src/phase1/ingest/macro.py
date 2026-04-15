"""
Macroeconomic data ingestion from FRED via fredapi.

Series ingested:
    DGS10    - 10-year Treasury yield (daily)
    VIXCLS   - CBOE VIX (daily)
    CPIAUCSL - CPI all-urban consumers (monthly -> forward-filled to business day)
    UNRATE   - Unemployment rate (monthly -> forward-filled)
    FEDFUNDS - Effective Federal Funds rate (monthly -> forward-filled)

Output: data/raw/macro/macro_raw.parquet
"""

import os
import yaml
import pandas as pd
from fredapi import Fred
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # loads FRED_API_KEY from .env if present


def ingest_macro(config: dict, force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch each FRED series individually, concatenate, resample to business-day
    frequency with forward-fill, and persist to Parquet.

    Uses fredapi (the official FRED Python client) rather than pandas_datareader,
    which is broken on Python 3.12+ due to distutils removal.

    Parameters
    ----------
    config : dict
        Parsed phase1_config.yaml.

    Returns
    -------
    pd.DataFrame
        DataFrame with one column per macro series, DatetimeIndex at business-day
        frequency, no leading NaNs after the first available observation.
    """
    series_ids = config["macro_series"]
    start = config["date_range"]["start"]
    end = config["date_range"]["end"]
    out_path = Path(config["raw_data_path"]) / "macro" / "macro_raw.parquet"

    # Load any cached series so we can fall back to them if FRED is flaky
    cached_series: dict[str, pd.Series] = {}
    if out_path.exists():
        try:
            cached_df = pd.read_parquet(out_path)
            for col in cached_df.columns:
                if col in series_ids:
                    cached_series[col] = cached_df[col]
        except Exception:
            pass

    if not force_refresh and set(series_ids).issubset(cached_series):
        print(f"Macro cache hit  -> loading from {out_path}")
        return pd.read_parquet(out_path)

    api_key = os.getenv("FRED_API_KEY")
    fred = Fred(api_key=api_key) if api_key else Fred()

    missing_ids = [s for s in series_ids if s not in cached_series] if not force_refresh else series_ids
    if missing_ids != series_ids:
        print(f"Macro partial cache -> fetching {len(missing_ids)} missing series from FRED")
    else:
        print(f"Fetching {len(series_ids)} FRED series: {start} to {end}")

    frames = list(cached_series.values()) if not force_refresh else []
    for sid in missing_ids:
        try:
            s = fred.get_series(sid, observation_start=start, observation_end=end)
            s.name = sid
            frames.append(s)
            print(f"  {sid}: {len(s)} observations")
        except Exception as exc:
            print(f"  WARNING: could not fetch {sid} — {exc}")
            if sid in cached_series:
                print(f"  Falling back to cached {sid}")
                frames.append(cached_series[sid])

    if not frames:
        raise RuntimeError("No macro series were fetched successfully.")

    macro = pd.concat(frames, axis=1)
    macro.index = pd.to_datetime(macro.index)

    # Resample to business-day frequency and forward-fill gaps.
    # Monthly series (CPI, UNRATE, FEDFUNDS) will be held constant between
    # release dates — this is the correct no-lookahead treatment.
    macro = macro.resample("B").last()
    macro = macro.ffill()

    # Trim to requested date range after resampling
    macro = macro.loc[start:end]

    null_pct = macro.isna().mean().round(4) * 100
    for col, pct in null_pct.items():
        if pct > 0:
            print(f"  WARNING: {col} has {pct:.1f}% nulls after forward-fill")

    out_path = Path(config["raw_data_path"]) / "macro" / "macro_raw.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    macro.to_parquet(out_path)

    print(f"Macro saved  -> {out_path}")
    print(f"  Shape      : {macro.shape[0]} rows x {macro.shape[1]} columns")
    print(f"  Date range : {macro.index.min().date()} to {macro.index.max().date()}")

    return macro


if __name__ == "__main__":
    with open("configs/phase1_config.yaml") as f:
        cfg = yaml.safe_load(f)
    ingest_macro(cfg)
