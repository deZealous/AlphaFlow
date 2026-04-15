"""
Tests for technical feature engineering.

All tests use synthetic OHLCV data so no network or disk access is needed.
"""

import pytest
import numpy as np
import pandas as pd

from src.phase1.features.technical import (
    engineer_features_for_ticker,
    engineer_all_features,
    _get_series,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_ohlcv(tickers: list[str], n: int = 300) -> pd.DataFrame:
    """Synthetic OHLCV DataFrame with realistic price structure."""
    idx = pd.bdate_range("2020-01-01", periods=n)
    data = {}
    rng = np.random.default_rng(42)
    for t in tickers:
        log_returns = rng.normal(0.0005, 0.015, n)
        close = 100 * np.exp(np.cumsum(log_returns))
        data[f"Close_{t}"]  = close
        data[f"Open_{t}"]   = close * rng.uniform(0.99, 1.01, n)
        data[f"High_{t}"]   = close * rng.uniform(1.00, 1.02, n)
        data[f"Low_{t}"]    = close * rng.uniform(0.98, 1.00, n)
        data[f"Volume_{t}"] = rng.integers(1_000_000, 10_000_000, n).astype(float)
    return pd.DataFrame(data, index=idx)


TICKERS = ["AAPL", "MSFT", "SPY"]
WINDOWS = [5, 10, 20, 60]


# ── _get_series ───────────────────────────────────────────────────────────────

class TestGetSeries:
    def test_returns_four_series(self):
        ohlcv = _make_ohlcv(["AAPL"])
        close, high, low, volume = _get_series(ohlcv, "AAPL")
        assert isinstance(close, pd.Series)
        assert isinstance(volume, pd.Series)

    def test_raises_on_missing_ticker(self):
        ohlcv = _make_ohlcv(["AAPL"])
        with pytest.raises(KeyError, match="MISSING"):
            _get_series(ohlcv, "MISSING")


# ── engineer_features_for_ticker ──────────────────────────────────────────────

class TestEngineerFeaturesForTicker:
    def setup_method(self):
        self.ohlcv = _make_ohlcv(TICKERS)
        self.feat  = engineer_features_for_ticker(self.ohlcv, "AAPL", WINDOWS)

    def test_returns_dataframe(self):
        assert isinstance(self.feat, pd.DataFrame)

    def test_index_matches_ohlcv(self):
        assert self.feat.index.equals(self.ohlcv.index)

    def test_expected_columns_present(self):
        expected = [
            "AAPL_rsi_14",
            "AAPL_macd", "AAPL_macd_signal", "AAPL_macd_diff",
            "AAPL_bb_high", "AAPL_bb_low", "AAPL_bb_width",
            "AAPL_atr_14",
            "AAPL_return_5d",  "AAPL_vol_5d",
            "AAPL_return_10d", "AAPL_vol_10d",
            "AAPL_return_20d", "AAPL_vol_20d",
            "AAPL_return_60d", "AAPL_vol_60d",
            "AAPL_vol_ratio",
        ]
        for col in expected:
            assert col in self.feat.columns, f"Missing column: {col}"

    def test_feature_count_is_17(self):
        # 4 momentum + 4 volatility price + 8 rolling (4 windows x 2) + 1 volume = 17
        assert self.feat.shape[1] == 17

    def test_shift_applied_to_returns(self):
        """Feature on day N must equal raw value on day N-1."""
        raw_return = self.ohlcv["Close_AAPL"].pct_change(5)
        expected   = raw_return.shift(1)
        actual     = self.feat["AAPL_return_5d"]
        # Compare on rows where both are non-null
        mask = expected.notna() & actual.notna()
        pd.testing.assert_series_equal(
            actual[mask].round(10),
            expected[mask].round(10),
            check_names=False,
        )

    def test_shift_applied_to_rsi(self):
        """RSI feature should be 1 day behind the raw RSI series."""
        import ta
        close = self.ohlcv["Close_AAPL"].astype(float)
        raw_rsi = ta.momentum.RSIIndicator(close=close, window=14).rsi()
        expected = raw_rsi.shift(1)
        actual   = self.feat["AAPL_rsi_14"]
        mask = expected.notna() & actual.notna()
        pd.testing.assert_series_equal(
            actual[mask].round(8),
            expected[mask].round(8),
            check_names=False,
        )

    def test_all_columns_are_numeric(self):
        for col in self.feat.columns:
            assert pd.api.types.is_numeric_dtype(self.feat[col]), f"{col} is not numeric"

    def test_leading_nulls_are_expected(self):
        """
        Indicators need warmup rows. First value of RSI-14 must be null,
        and the 15th row (index 14, 0-based) must be non-null after shift.
        RSI warmup = 14 rows + 1 for shift = first non-null at row 15.
        """
        rsi = self.feat["AAPL_rsi_14"]
        assert rsi.iloc[0] != rsi.iloc[0]    # NaN check (NaN != NaN)
        assert rsi.iloc[15] == rsi.iloc[15]  # not NaN

    def test_vol_ratio_no_negative_values(self):
        """Volume ratio must be non-negative (volume / moving average)."""
        vol_ratio = self.feat["AAPL_vol_ratio"].dropna()
        assert (vol_ratio >= 0).all()

    def test_column_prefix_matches_ticker(self):
        for col in self.feat.columns:
            assert col.startswith("AAPL_"), f"Column {col} has wrong prefix"


# ── engineer_all_features ─────────────────────────────────────────────────────

class TestEngineerAllFeatures:
    def setup_method(self):
        self.ohlcv  = _make_ohlcv(TICKERS)
        self.matrix = engineer_all_features(self.ohlcv, TICKERS, WINDOWS)

    def test_returns_dataframe(self):
        assert isinstance(self.matrix, pd.DataFrame)

    def test_shape_rows_match_ohlcv(self):
        assert self.matrix.shape[0] == self.ohlcv.shape[0]

    def test_shape_cols_is_tickers_times_features(self):
        # 3 tickers x 17 features = 51 columns
        assert self.matrix.shape[1] == len(TICKERS) * 17

    def test_all_tickers_represented(self):
        for t in TICKERS:
            assert f"{t}_rsi_14" in self.matrix.columns

    def test_skips_missing_ticker_gracefully(self):
        """A ticker not in the OHLCV df should be skipped, not raise."""
        result = engineer_all_features(self.ohlcv, TICKERS + ["FAKE"], WINDOWS)
        # FAKE skipped — column count unchanged vs without FAKE
        assert result.shape[1] == len(TICKERS) * 17

    def test_index_is_datetime(self):
        assert isinstance(self.matrix.index, pd.DatetimeIndex)

    def test_no_ticker_cross_contamination(self):
        """AAPL features must not appear in MSFT columns."""
        msft_cols = [c for c in self.matrix.columns if c.startswith("MSFT_")]
        for col in msft_cols:
            assert "AAPL" not in col
