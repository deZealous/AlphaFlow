"""
Technical feature engineering for Phase 1.

For each ticker, computes:
    Momentum   : RSI-14, MACD, MACD signal, MACD diff
    Volatility : Bollinger Bands (high/low/width), ATR-14
    Returns    : rolling returns over 5, 10, 20, 60-day windows
    Volatility : rolling close-return std over 5, 10, 20, 60-day windows
    Volume     : volume ratio vs 20-day rolling mean

All feature columns are shifted forward by 1 trading day before being
returned — this is the primary lookahead guard.  The leakage checker in
validate/leakage_check.py provides a second independent verification.

Column naming convention: {TICKER}_{feature}_{window}
Examples: AAPL_rsi_14, MSFT_return_20d, SPY_bb_width, GLD_vol_ratio
"""

import pandas as pd
import numpy as np
import ta
from tqdm import tqdm


# Windows used for rolling statistics (configurable via config dict)
DEFAULT_WINDOWS = [5, 10, 20, 60]


def _get_series(ohlcv: pd.DataFrame, ticker: str) -> tuple:
    """
    Extract (close, high, low, volume) Series for a single ticker.

    Raises KeyError with a clear message if the ticker is absent.
    """
    required = ["Close", "High", "Low", "Volume"]
    missing = [f for f in required if f"{f}_{ticker}" not in ohlcv.columns]
    if missing:
        raise KeyError(
            f"Ticker '{ticker}' missing columns: {[f'{f}_{ticker}' for f in missing]}"
        )
    close  = ohlcv[f"Close_{ticker}"].astype(float)
    high   = ohlcv[f"High_{ticker}"].astype(float)
    low    = ohlcv[f"Low_{ticker}"].astype(float)
    volume = ohlcv[f"Volume_{ticker}"].astype(float)
    return close, high, low, volume


def engineer_features_for_ticker(
    ohlcv: pd.DataFrame,
    ticker: str,
    windows: list[int] = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    """
    Compute all technical features for a single ticker and lag them by 1 day.

    Parameters
    ----------
    ohlcv : pd.DataFrame
        Raw OHLCV DataFrame with columns {Field}_{TICKER}.
    ticker : str
        Ticker symbol (e.g. "AAPL").
    windows : list[int]
        Rolling windows for return and volatility features.

    Returns
    -------
    pd.DataFrame
        Feature columns for this ticker, all shifted forward by 1 trading day.
        Index matches ohlcv.index.
    """
    close, high, low, volume = _get_series(ohlcv, ticker)
    t = ticker  # shorthand for column name construction
    features = pd.DataFrame(index=ohlcv.index)

    # ── Momentum ──────────────────────────────────────────────────────────────
    features[f"{t}_rsi_14"] = ta.momentum.RSIIndicator(
        close=close, window=14
    ).rsi()

    macd_ind = ta.trend.MACD(close=close)
    features[f"{t}_macd"]        = macd_ind.macd()
    features[f"{t}_macd_signal"] = macd_ind.macd_signal()
    features[f"{t}_macd_diff"]   = macd_ind.macd_diff()

    # ── Volatility (price-based) ───────────────────────────────────────────────
    bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    features[f"{t}_bb_high"]  = bb.bollinger_hband()
    features[f"{t}_bb_low"]   = bb.bollinger_lband()
    features[f"{t}_bb_width"] = bb.bollinger_wband()

    features[f"{t}_atr_14"] = ta.volatility.AverageTrueRange(
        high=high, low=low, close=close, window=14
    ).average_true_range()

    # ── Rolling returns ────────────────────────────────────────────────────────
    daily_return = close.pct_change()
    for w in windows:
        features[f"{t}_return_{w}d"] = close.pct_change(w)
        features[f"{t}_vol_{w}d"]    = daily_return.rolling(w).std()

    # ── Volume ratio ──────────────────────────────────────────────────────────
    vol_ma = volume.rolling(20).mean()
    # Guard against division by zero on thinly-traded early rows
    features[f"{t}_vol_ratio"] = volume / vol_ma.replace(0, np.nan)

    # ── Lag all features by 1 trading day ─────────────────────────────────────
    features = features.shift(1)

    return features


def engineer_all_features(
    ohlcv: pd.DataFrame,
    tickers: list[str],
    windows: list[int] = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    """
    Run feature engineering for every ticker and concatenate into a single
    wide feature matrix.

    Tickers that are missing from the OHLCV DataFrame are skipped with a
    warning rather than raising — this prevents one bad ticker from aborting
    the entire pipeline.

    Parameters
    ----------
    ohlcv : pd.DataFrame
        Raw OHLCV DataFrame (from ingest_ohlcv).
    tickers : list[str]
        All ticker symbols to process.
    windows : list[int]
        Rolling windows for return and volatility features.

    Returns
    -------
    pd.DataFrame
        Wide feature matrix: ~1250 rows x (n_tickers * features_per_ticker) columns.
        All columns are lagged by 1 day.
    """
    frames = []
    skipped = []

    for ticker in tqdm(tickers, desc="Engineering features", unit="ticker"):
        try:
            feat = engineer_features_for_ticker(ohlcv, ticker, windows)
            frames.append(feat)
        except KeyError as exc:
            skipped.append(ticker)
            print(f"  SKIP {ticker}: {exc}")

    if not frames:
        raise RuntimeError("No features were engineered — all tickers failed.")

    if skipped:
        print(f"\nSkipped {len(skipped)} tickers: {skipped}")

    feature_matrix = pd.concat(frames, axis=1)

    # Features per ticker count (for logging)
    n_features = len(frames[0].columns) if frames else 0
    print(f"\nFeature engineering complete:")
    print(f"  Tickers processed : {len(frames)}")
    print(f"  Features/ticker   : {n_features}")
    print(f"  Matrix shape      : {feature_matrix.shape[0]} rows x {feature_matrix.shape[1]} columns")
    print(f"  Expected columns  : ~{len(tickers) * n_features} (={len(tickers)} tickers x {n_features} features)")

    return feature_matrix
