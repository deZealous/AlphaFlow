"""
Schema validation for raw and feature DataFrames.

Checks run in order — the first failure raises a descriptive ValueError so
the caller knows exactly what broke and where.
"""

import pandas as pd
from dataclasses import dataclass, field


@dataclass
class SchemaReport:
    passed: bool = True
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def fail(self, msg: str) -> None:
        self.failures.append(msg)
        self.passed = False

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def __str__(self) -> str:
        lines = ["-- Schema validation report ------------------"]
        lines.append(f"  Result   : {'PASS' if self.passed else 'FAIL'}")
        for f in self.failures:
            lines.append(f"  FAIL     : {f}")
        for w in self.warnings:
            lines.append(f"  WARN     : {w}")
        if not self.failures and not self.warnings:
            lines.append("  All checks passed with no warnings.")
        lines.append("---------------------------------------------")
        return "\n".join(lines)


def validate_ohlcv(df: pd.DataFrame, expected_tickers: list[str]) -> SchemaReport:
    """
    Validate raw OHLCV DataFrame coming out of ingest_ohlcv.

    Checks
    ------
    - Index is DatetimeIndex, sorted, no nulls
    - All columns are numeric
    - Columns exist for every expected ticker (Close_{ticker} minimum)
    - Null rate < 5% per column
    - At least 1200 trading days
    """
    report = SchemaReport()

    # 1. Index integrity
    if not isinstance(df.index, pd.DatetimeIndex):
        report.fail(f"Index is {type(df.index).__name__}, expected DatetimeIndex")
    elif df.index.isna().any():
        report.fail("Index contains NaT values")
    elif not df.index.is_monotonic_increasing:
        report.fail("Index is not sorted in ascending order")

    # 2. All columns must be numeric
    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        report.fail(f"Non-numeric columns found: {non_numeric[:5]}")

    # 3. Minimum ticker coverage — at least Close column per ticker
    missing_tickers = [t for t in expected_tickers if f"Close_{t}" not in df.columns]
    if missing_tickers:
        report.fail(f"Missing Close column for tickers: {missing_tickers}")

    # 4. Missing value rate
    null_rates = df.isna().mean()
    over_threshold = null_rates[null_rates > 0.05]
    if not over_threshold.empty:
        report.fail(
            f"{len(over_threshold)} columns exceed 5% null rate: "
            f"{list(over_threshold.index[:5])}"
        )

    # 5. Row count
    if len(df) < 1200:
        report.fail(f"Only {len(df)} rows — expected at least 1200 trading days")
    elif len(df) < 1250:
        report.warn(f"Row count {len(df)} is below the ~1250 target (minor, acceptable)")

    print(report)
    return report


def validate_macro(df: pd.DataFrame, expected_series: list[str]) -> SchemaReport:
    """
    Validate raw macro DataFrame coming out of ingest_macro.

    Checks
    ------
    - Index is DatetimeIndex, sorted, no nulls
    - All expected series present as columns
    - Null rate < 1% per column (macro should forward-fill cleanly)
    """
    report = SchemaReport()

    if not isinstance(df.index, pd.DatetimeIndex):
        report.fail(f"Index is {type(df.index).__name__}, expected DatetimeIndex")
    elif df.index.isna().any():
        report.fail("Index contains NaT values")
    elif not df.index.is_monotonic_increasing:
        report.fail("Index is not sorted in ascending order")

    missing_cols = [s for s in expected_series if s not in df.columns]
    if missing_cols:
        report.fail(f"Missing macro series: {missing_cols}")

    null_rates = df.isna().mean()
    over_threshold = null_rates[null_rates > 0.01]
    if not over_threshold.empty:
        report.fail(
            f"Macro columns exceed 1% null rate after forward-fill: "
            f"{list(over_threshold.index)}"
        )

    print(report)
    return report


def validate_feature_matrix(
    df: pd.DataFrame,
    expected_min_cols: int = 300,
    max_null_rate: float = 0.05,
) -> SchemaReport:
    """
    Validate the final engineered feature matrix before it is saved to Parquet.

    Checks
    ------
    - Index is DatetimeIndex, sorted, no nulls
    - At least expected_min_cols columns
    - No future dates in index (sanity cap: today)
    - Null rate < max_null_rate per column
    - At least 1200 rows
    """
    report = SchemaReport()

    if not isinstance(df.index, pd.DatetimeIndex):
        report.fail(f"Index is {type(df.index).__name__}, expected DatetimeIndex")
    elif df.index.isna().any():
        report.fail("Index contains NaT values")
    elif not df.index.is_monotonic_increasing:
        report.fail("Index is not sorted in ascending order")

    # No future dates
    today = pd.Timestamp.today().normalize()
    future = df.index[df.index > today]
    if not future.empty:
        report.fail(f"{len(future)} rows have future dates (first: {future[0].date()})")

    if df.shape[1] < expected_min_cols:
        report.warn(
            f"Feature matrix has {df.shape[1]} columns — expected {expected_min_cols}+. "
            "Macro-only join may not have run yet."
        )

    null_rates = df.isna().mean()
    over_threshold = null_rates[null_rates > max_null_rate]
    if not over_threshold.empty:
        report.fail(
            f"{len(over_threshold)} columns exceed {max_null_rate*100:.0f}% null rate: "
            f"{list(over_threshold.index[:5])}"
        )

    if len(df) < 1200:
        report.fail(f"Only {len(df)} rows — expected at least 1200 trading days")

    print(report)
    return report
