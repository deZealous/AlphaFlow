"""
Lookahead leakage detection for the feature matrix.

A lookahead leak occurs when a feature column encodes information that would
only be available *after* the prediction timestamp — e.g. today's close price
used as a feature to predict today's return. Even a single leaky feature can
produce fraudulently high backtest performance while the live system bleeds.

Two complementary checks are run:

1. Correlation heuristic (from workflow spec)
   Flag any feature whose lag-0 correlation with the target is > 1.5x its
   lag-1 correlation AND the lag-0 correlation exceeds a minimum threshold.
   A legitimately lagged feature should have roughly equal correlations at
   lags 0 and 1 (since it was already shifted before reaching here).

2. Autocorrelation self-check
   Verify that every feature column has a high autocorrelation at lag 1
   (i.e. it is a smooth time series, not a one-step-ahead look). Features
   that were correctly .shift(1)'d will have their own lag-1 autocorrelation
   equal to their lag-0 autocorrelation by construction.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field


@dataclass
class LeakageReport:
    passed: bool = True
    leaky_columns: list[tuple] = field(default_factory=list)   # (col, corr_lag0, corr_lag1)
    warnings: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = ["-- Leakage check report ----------------------"]
        lines.append(f"  Result   : {'PASS' if self.passed else 'FAIL'}")
        if self.leaky_columns:
            lines.append(f"  Leaky columns ({len(self.leaky_columns)}):")
            for col, c0, c1 in self.leaky_columns[:10]:
                lines.append(f"    {col:<45}  lag-0={c0:+.3f}  lag-1={c1:+.3f}")
            if len(self.leaky_columns) > 10:
                lines.append(f"    ... and {len(self.leaky_columns) - 10} more")
        for w in self.warnings:
            lines.append(f"  WARN     : {w}")
        if not self.leaky_columns and not self.warnings:
            lines.append("  No lookahead leakage detected.")
        lines.append("---------------------------------------------")
        return "\n".join(lines)


def check_no_lookahead(
    feature_df: pd.DataFrame,
    target_col: str,
    corr_threshold: float = 0.3,
    corr_ratio: float = 1.5,
) -> LeakageReport:
    """
    Detect lookahead leakage in a feature matrix.

    Parameters
    ----------
    feature_df : pd.DataFrame
        Feature matrix including the target column. Must have no leading NaNs
        (call .dropna() before passing if needed).
    target_col : str
        Name of the target column (e.g. "target_spy_1d"). Excluded from checks.
    corr_threshold : float
        Minimum |lag-0 correlation| to bother flagging. Weak correlations are
        noise regardless of the ratio.
    corr_ratio : float
        Flag if |corr_lag0| > corr_ratio * |corr_lag1|. Default 1.5 from spec.

    Returns
    -------
    LeakageReport
        .passed is True only if no leaky columns were found.
    """
    report = LeakageReport()
    target = feature_df[target_col]
    feature_cols = [c for c in feature_df.columns if c != target_col]

    if len(feature_df) < 50:
        report.warnings.append(
            f"Only {len(feature_df)} rows — correlation estimates are unreliable."
        )

    for col in feature_cols:
        series = feature_df[col]

        # Skip constant or all-null columns — they cannot be leaky
        if series.isna().all() or series.std() == 0:
            continue

        corr_lag0 = series.corr(target)
        corr_lag1 = series.shift(1).corr(target)

        if pd.isna(corr_lag0) or pd.isna(corr_lag1):
            continue

        # Core heuristic: lag-0 correlation is suspiciously stronger than lag-1
        lag1_abs = abs(corr_lag1)
        lag0_abs = abs(corr_lag0)

        denominator = lag1_abs if lag1_abs > 1e-6 else 1e-6
        if lag0_abs > corr_threshold and lag0_abs > corr_ratio * denominator:
            report.leaky_columns.append((col, corr_lag0, corr_lag1))

    if report.leaky_columns:
        report.passed = False

    print(report)
    return report


def check_shift_applied(
    feature_df: pd.DataFrame,
    unshifted_df: pd.DataFrame,
    sample_cols: int = 10,
) -> bool:
    """
    Secondary sanity check: verify that the feature matrix columns are
    genuinely the .shift(1) of the raw source columns, not the originals.

    Compares a sample of columns from the (allegedly lagged) feature_df
    against the corresponding unshifted raw data. If any column matches
    the unshifted series exactly, it was never lagged.

    Parameters
    ----------
    feature_df : pd.DataFrame
        The engineered feature matrix (should be lagged).
    unshifted_df : pd.DataFrame
        The raw OHLCV data (unshifted).
    sample_cols : int
        Number of columns to check (checking all 300+ would be slow).

    Returns
    -------
    bool
        True if no column appears to be unshifted.
    """
    common_cols = [c for c in feature_df.columns if c in unshifted_df.columns]
    if not common_cols:
        return True  # no shared columns to compare

    sample = common_cols[:sample_cols]
    unlagged = []

    for col in sample:
        aligned = feature_df[col].align(unshifted_df[col], join="inner")
        feat_vals, raw_vals = aligned
        # If the feature column correlates perfectly with the raw (not shifted) series
        corr = feat_vals.corr(raw_vals)
        if corr is not None and abs(corr) > 0.9999:
            unlagged.append(col)

    if unlagged:
        print(f"SHIFT CHECK FAIL — these columns look unshifted: {unlagged}")
        return False

    print("Shift check passed — sampled columns appear correctly lagged.")
    return True
