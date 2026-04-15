"""
Align NLP features to the Phase 1 trading day index.

NLP data arrives on irregular dates: earnings calls are quarterly, news
is sporadic, EDGAR filings cluster around reporting periods. This module
reindexes all NLP feature DataFrames to the exact set of trading days in
the OHLCV feature matrix and handles gaps sensibly:

    - Forward-fill up to `ffill_limit` days (default 5).
      Sentiment from Monday carries through to Friday; beyond that it is
      stale and should not be propagated.
    - Remaining NaNs (no news in the opening days of history, or gaps
      longer than ffill_limit) are filled with 0 (neutral).
"""

import pandas as pd


def align_to_trading_days(
    nlp_features: pd.DataFrame,
    trading_index: pd.DatetimeIndex,
    ffill_limit: int = 5,
) -> pd.DataFrame:
    """
    Reindex NLP features to match the trading day index.

    Parameters
    ----------
    nlp_features : pd.DataFrame
        Sparse NLP feature DataFrame (DatetimeIndex, possibly irregular).
    trading_index : pd.DatetimeIndex
        The exact index from the Phase 1 feature matrix.
    ffill_limit : int
        Maximum number of consecutive trading days to forward-fill.

    Returns
    -------
    pd.DataFrame
        Dense DataFrame with the same index as trading_index.
        No NaN values remain.
    """
    if nlp_features.empty:
        # Return an all-zero DataFrame with the correct index
        return pd.DataFrame(0.0, index=trading_index, columns=nlp_features.columns)

    # Reindex to trading days — introduces NaN wherever NLP data is absent
    aligned = nlp_features.reindex(trading_index)

    # Forward-fill with a cap: stale sentiment beyond ffill_limit days = noise
    aligned = aligned.ffill(limit=ffill_limit)

    # Any remaining NaN (long gaps or opening history) -> neutral (0)
    aligned = aligned.fillna(0.0)

    return aligned


def validate_alignment(
    nlp_features: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    name: str = "NLP",
) -> bool:
    """
    Verify that nlp_features shares its index exactly with feature_matrix.

    Parameters
    ----------
    nlp_features : pd.DataFrame
        Aligned NLP feature DataFrame.
    feature_matrix : pd.DataFrame
        Phase 1 feature matrix (the reference index).
    name : str
        Label used in log messages.

    Returns
    -------
    bool
        True if aligned, False otherwise.
    """
    if not nlp_features.index.equals(feature_matrix.index):
        print(
            f"ALIGNMENT FAIL [{name}]: index mismatch. "
            f"NLP has {len(nlp_features)} rows, feature_matrix has {len(feature_matrix)} rows."
        )
        return False

    null_cols = nlp_features.columns[nlp_features.isna().any()].tolist()
    if null_cols:
        print(f"ALIGNMENT WARN [{name}]: {len(null_cols)} columns still have NaN after fill.")

    print(
        f"Alignment OK [{name}]: {nlp_features.shape[1]} columns "
        f"aligned to {len(nlp_features)} trading days."
    )
    return True
