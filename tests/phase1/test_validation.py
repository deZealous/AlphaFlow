"""
Tests for schema validation and leakage detection.

Fixtures build minimal synthetic DataFrames so tests run without
hitting any network or touching data/ on disk.
"""

import pytest
import numpy as np
import pandas as pd

from src.phase1.validate.schema import validate_ohlcv, validate_macro, validate_feature_matrix
from src.phase1.validate.leakage_check import check_no_lookahead


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bdays(n: int = 1300) -> pd.DatetimeIndex:
    return pd.bdate_range("2019-01-01", periods=n)


def _make_ohlcv(tickers: list[str], n: int = 1300) -> pd.DataFrame:
    """Minimal OHLCV DataFrame with correct column naming."""
    idx = _bdays(n)
    data = {}
    for t in tickers:
        base = np.random.rand(n) * 100 + 50
        data[f"Close_{t}"]  = base
        data[f"Open_{t}"]   = base * np.random.uniform(0.99, 1.01, n)
        data[f"High_{t}"]   = base * np.random.uniform(1.00, 1.02, n)
        data[f"Low_{t}"]    = base * np.random.uniform(0.98, 1.00, n)
        data[f"Volume_{t}"] = np.random.randint(1_000_000, 10_000_000, n).astype(float)
    return pd.DataFrame(data, index=idx)


def _make_macro(series: list[str], n: int = 1300) -> pd.DataFrame:
    idx = _bdays(n)
    return pd.DataFrame(
        {s: np.random.rand(n) for s in series},
        index=idx,
    )


# ── validate_ohlcv ────────────────────────────────────────────────────────────

TICKERS = ["AAPL", "MSFT", "SPY"]


class TestValidateOhlcv:
    def test_passes_on_clean_data(self):
        df = _make_ohlcv(TICKERS)
        report = validate_ohlcv(df, TICKERS)
        assert report.passed

    def test_fails_on_non_datetime_index(self):
        df = _make_ohlcv(TICKERS)
        df.index = range(len(df))
        report = validate_ohlcv(df, TICKERS)
        assert not report.passed
        assert any("DatetimeIndex" in f for f in report.failures)

    def test_fails_on_unsorted_index(self):
        df = _make_ohlcv(TICKERS)
        df = df.iloc[::-1]  # reverse so index is descending
        report = validate_ohlcv(df, TICKERS)
        assert not report.passed
        assert any("sorted" in f for f in report.failures)

    def test_fails_on_missing_ticker(self):
        df = _make_ohlcv(TICKERS)
        df = df.drop(columns=["Close_SPY"])
        report = validate_ohlcv(df, TICKERS)
        assert not report.passed
        assert any("SPY" in f for f in report.failures)

    def test_fails_when_null_rate_exceeds_threshold(self):
        df = _make_ohlcv(TICKERS)
        # Inject 10% nulls into one column
        mask = np.random.choice(len(df), size=int(len(df) * 0.10), replace=False)
        df.iloc[mask, 0] = np.nan
        report = validate_ohlcv(df, TICKERS)
        assert not report.passed
        assert any("null rate" in f for f in report.failures)

    def test_fails_when_too_few_rows(self):
        df = _make_ohlcv(TICKERS, n=500)
        report = validate_ohlcv(df, TICKERS)
        assert not report.passed
        assert any("1200" in f for f in report.failures)


# ── validate_macro ────────────────────────────────────────────────────────────

MACRO_SERIES = ["DGS10", "VIXCLS", "CPIAUCSL"]


class TestValidateMacro:
    def test_passes_on_clean_data(self):
        df = _make_macro(MACRO_SERIES)
        report = validate_macro(df, MACRO_SERIES)
        assert report.passed

    def test_fails_on_missing_series(self):
        df = _make_macro(MACRO_SERIES)
        df = df.drop(columns=["VIXCLS"])
        report = validate_macro(df, MACRO_SERIES)
        assert not report.passed
        assert any("VIXCLS" in f for f in report.failures)

    def test_fails_when_macro_has_high_nulls(self):
        df = _make_macro(MACRO_SERIES)
        mask = np.random.choice(len(df), size=int(len(df) * 0.05), replace=False)
        df.iloc[mask, 0] = np.nan
        report = validate_macro(df, MACRO_SERIES)
        assert not report.passed
        assert any("null rate" in f for f in report.failures)


# ── validate_feature_matrix ───────────────────────────────────────────────────

class TestValidateFeatureMatrix:
    def _make_feature_matrix(self, n_cols: int = 310, n_rows: int = 1300) -> pd.DataFrame:
        idx = _bdays(n_rows)
        return pd.DataFrame(
            np.random.rand(n_rows, n_cols),
            index=idx,
            columns=[f"feat_{i}" for i in range(n_cols)],
        )

    def test_passes_on_clean_matrix(self):
        df = self._make_feature_matrix()
        report = validate_feature_matrix(df)
        assert report.passed

    def test_warns_on_low_column_count(self):
        df = self._make_feature_matrix(n_cols=150)
        report = validate_feature_matrix(df)
        # Warn, not fail
        assert report.passed
        assert any("columns" in w for w in report.warnings)

    def test_fails_on_future_dates(self):
        df = self._make_feature_matrix()
        future_idx = pd.bdate_range("2030-01-01", periods=len(df))
        df.index = future_idx
        report = validate_feature_matrix(df)
        assert not report.passed
        assert any("future" in f for f in report.failures)


# ── check_no_lookahead ────────────────────────────────────────────────────────

class TestLeakageCheck:
    def _make_clean_df(self, n: int = 500) -> pd.DataFrame:
        """Feature already shifted by 1 day — should pass the leakage check."""
        idx = _bdays(n)
        prices = pd.Series(np.cumsum(np.random.randn(n)) + 100, index=idx)
        feature = prices.pct_change().shift(1)   # correctly lagged
        target  = prices.pct_change().shift(-1)  # next-day return
        return pd.DataFrame({"feature": feature, "target": target}, index=idx).dropna()

    def _make_leaky_df(self, n: int = 500) -> pd.DataFrame:
        """Feature is perfectly correlated with target at lag 0 — must be flagged."""
        idx = _bdays(n)
        target = pd.Series(np.random.randn(n), index=idx)
        leaky  = target * 2.0 + np.random.randn(n) * 0.01  # near-perfect lag-0 correlation
        return pd.DataFrame({"leaky_feature": leaky, "target": target}, index=idx)

    def test_clean_data_passes(self):
        df = self._make_clean_df()
        report = check_no_lookahead(df, "target")
        assert report.passed

    def test_leaky_data_fails(self):
        df = self._make_leaky_df()
        report = check_no_lookahead(df, "target")
        assert not report.passed
        assert len(report.leaky_columns) >= 1

    def test_leaky_column_name_appears_in_report(self):
        df = self._make_leaky_df()
        report = check_no_lookahead(df, "target")
        flagged_names = [col for col, _, _ in report.leaky_columns]
        assert "leaky_feature" in flagged_names

    def test_constant_column_not_flagged(self):
        """Constant columns have zero variance — should be silently skipped."""
        df = self._make_clean_df()
        df["constant"] = 1.0
        report = check_no_lookahead(df, "target")
        assert report.passed

    def test_target_col_excluded_from_check(self):
        """The target itself must not be tested against itself."""
        df = self._make_clean_df()
        report = check_no_lookahead(df, "target")
        flagged_names = [col for col, _, _ in report.leaky_columns]
        assert "target" not in flagged_names
