"""
Tests for OHLCV and macro ingestion.

Strategy
--------
- Pure functions (_parse_tickers) tested directly.
- Cache paths tested against the real parquet files already on disk
  (written by the live pipeline run) — no network required.
- Download paths tested with unittest.mock so the test suite never
  hits yfinance or FRED, keeping it fast and deterministic.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.phase1.ingest.ohlcv import ingest_ohlcv, _parse_tickers
from src.phase1.ingest.macro import ingest_macro


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def base_config(tmp_path):
    """Minimal config pointing at a temp directory."""
    return {
        "tickers": ["AAPL, MSFT", "SPY, GLD"],
        "date_range": {"start": "2022-01-01", "end": "2023-12-31"},
        "macro_series": ["DGS10", "VIXCLS", "CPIAUCSL"],
        "raw_data_path": str(tmp_path / "raw"),
        "processed_data_path": str(tmp_path / "processed"),
        "feature_store_path": str(tmp_path / "features"),
    }


def _make_fake_ohlcv_multiindex(tickers: list[str], n: int = 500) -> pd.DataFrame:
    """
    Mimics what yfinance returns for multiple tickers:
    a MultiIndex DataFrame with (field, ticker) column tuples.
    """
    idx = pd.bdate_range("2022-01-01", periods=n)
    fields = ["Close", "High", "Low", "Open", "Volume"]
    arrays = [
        np.repeat(fields, len(tickers)),
        np.tile(tickers, len(fields)),
    ]
    cols = pd.MultiIndex.from_arrays(arrays, names=["Price", "Ticker"])
    rng = np.random.default_rng(0)
    data = rng.uniform(50, 200, size=(n, len(fields) * len(tickers)))
    return pd.DataFrame(data, index=idx, columns=cols)


def _make_fake_macro_series(sid: str, n: int = 500) -> pd.Series:
    idx = pd.bdate_range("2022-01-01", periods=n)
    return pd.Series(np.random.rand(n), index=idx, name=sid)


# ── _parse_tickers ────────────────────────────────────────────────────────────

class TestParseTickers:
    def test_flattens_comma_separated_groups(self):
        config = {"tickers": ["AAPL, MSFT, GOOGL", "SPY, QQQ"]}
        result = _parse_tickers(config)
        assert result == ["AAPL", "MSFT", "GOOGL", "SPY", "QQQ"]

    def test_single_group(self):
        config = {"tickers": ["AAPL"]}
        assert _parse_tickers(config) == ["AAPL"]

    def test_strips_whitespace(self):
        config = {"tickers": ["  AAPL ,  MSFT  "]}
        result = _parse_tickers(config)
        assert result == ["AAPL", "MSFT"]

    def test_returns_list(self):
        config = {"tickers": ["SPY"]}
        assert isinstance(_parse_tickers(config), list)

    def test_full_30_ticker_config(self):
        config = {
            "tickers": [
                "AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, JPM, GS, BAC",
                "XOM, CVX, LLY, JNJ, UNH, WMT, PG, KO, PEP, MCD",
                "SPY, QQQ, IWM, GLD, TLT, USO, VNQ, HYG, EEM, DIA",
            ]
        }
        result = _parse_tickers(config)
        assert len(result) == 30
        assert "AAPL" in result
        assert "DIA" in result


# ── ingest_ohlcv ──────────────────────────────────────────────────────────────

class TestIngestOhlcv:
    def test_cache_hit_skips_download(self, base_config):
        """If parquet already exists and force_refresh=False, yfinance is never called."""
        out_path = Path(base_config["raw_data_path"]) / "ohlcv" / "ohlcv_raw.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Write a fake cached file
        tickers = _parse_tickers(base_config)
        fake = pd.DataFrame(
            {"Close_AAPL": [100.0, 101.0], "Close_MSFT": [200.0, 201.0]},
            index=pd.bdate_range("2022-01-03", periods=2),
        )
        fake.to_parquet(out_path)

        with patch("src.phase1.ingest.ohlcv.yf.download") as mock_dl:
            result = ingest_ohlcv(base_config, force_refresh=False)
            mock_dl.assert_not_called()

        pd.testing.assert_frame_equal(result, fake, check_freq=False)

    def test_force_refresh_always_downloads(self, base_config):
        """force_refresh=True must call yfinance even if cache exists."""
        out_path = Path(base_config["raw_data_path"]) / "ohlcv" / "ohlcv_raw.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        tickers = _parse_tickers(base_config)
        fake_raw = _make_fake_ohlcv_multiindex(tickers, n=50)

        with patch("src.phase1.ingest.ohlcv.yf.download", return_value=fake_raw):
            result = ingest_ohlcv(base_config, force_refresh=True)

        assert isinstance(result, pd.DataFrame)

    def test_columns_flattened_to_field_ticker(self, base_config):
        """Output columns must follow {Field}_{TICKER} naming convention."""
        tickers = _parse_tickers(base_config)
        fake_raw = _make_fake_ohlcv_multiindex(tickers, n=50)

        with patch("src.phase1.ingest.ohlcv.yf.download", return_value=fake_raw):
            result = ingest_ohlcv(base_config, force_refresh=True)

        for col in result.columns:
            assert "_" in col, f"Column '{col}' missing underscore separator"
            field, ticker = col.split("_", 1)
            assert field in {"Close", "High", "Low", "Open", "Volume"}, \
                f"Unknown field '{field}' in column '{col}'"
            assert ticker in tickers, f"Unknown ticker '{ticker}' in column '{col}'"

    def test_index_is_datetime(self, base_config):
        tickers = _parse_tickers(base_config)
        fake_raw = _make_fake_ohlcv_multiindex(tickers, n=50)

        with patch("src.phase1.ingest.ohlcv.yf.download", return_value=fake_raw):
            result = ingest_ohlcv(base_config, force_refresh=True)

        assert isinstance(result.index, pd.DatetimeIndex)

    def test_index_sorted_ascending(self, base_config):
        tickers = _parse_tickers(base_config)
        fake_raw = _make_fake_ohlcv_multiindex(tickers, n=50)

        with patch("src.phase1.ingest.ohlcv.yf.download", return_value=fake_raw):
            result = ingest_ohlcv(base_config, force_refresh=True)

        assert result.index.is_monotonic_increasing

    def test_parquet_written_to_correct_path(self, base_config):
        tickers = _parse_tickers(base_config)
        fake_raw = _make_fake_ohlcv_multiindex(tickers, n=50)
        expected_path = Path(base_config["raw_data_path"]) / "ohlcv" / "ohlcv_raw.parquet"

        with patch("src.phase1.ingest.ohlcv.yf.download", return_value=fake_raw):
            ingest_ohlcv(base_config, force_refresh=True)

        assert expected_path.exists()

    def test_result_matches_saved_parquet(self, base_config):
        """Return value must equal what was written to disk."""
        tickers = _parse_tickers(base_config)
        fake_raw = _make_fake_ohlcv_multiindex(tickers, n=50)
        expected_path = Path(base_config["raw_data_path"]) / "ohlcv" / "ohlcv_raw.parquet"

        with patch("src.phase1.ingest.ohlcv.yf.download", return_value=fake_raw):
            result = ingest_ohlcv(base_config, force_refresh=True)

        on_disk = pd.read_parquet(expected_path)
        pd.testing.assert_frame_equal(result, on_disk, check_freq=False)


# ── ingest_macro ──────────────────────────────────────────────────────────────

class TestIngestMacro:
    def _fred_side_effect(self, series_ids: list[str]):
        """Returns a side_effect function for Fred().get_series mock."""
        def side_effect(sid, **kwargs):
            if sid in series_ids:
                return _make_fake_macro_series(sid)
            raise ValueError(f"Unknown series {sid}")
        return side_effect

    def test_cache_hit_skips_fred(self, base_config):
        """All series present in cache -> FRED never called."""
        out_path = Path(base_config["raw_data_path"]) / "macro" / "macro_raw.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        idx = pd.bdate_range("2022-01-01", periods=100)
        fake = pd.DataFrame(
            {s: np.random.rand(100) for s in base_config["macro_series"]},
            index=idx,
        )
        fake.to_parquet(out_path)

        with patch("src.phase1.ingest.macro.Fred") as mock_fred_cls:
            result = ingest_macro(base_config, force_refresh=False)
            mock_fred_cls.assert_not_called()

        pd.testing.assert_frame_equal(result, fake, check_freq=False)

    def test_force_refresh_always_fetches(self, base_config):
        sids = base_config["macro_series"]
        mock_fred = MagicMock()
        mock_fred.get_series.side_effect = self._fred_side_effect(sids)

        with patch("src.phase1.ingest.macro.Fred", return_value=mock_fred):
            result = ingest_macro(base_config, force_refresh=True)

        assert mock_fred.get_series.call_count == len(sids)
        assert isinstance(result, pd.DataFrame)

    def test_all_series_present_as_columns(self, base_config):
        sids = base_config["macro_series"]
        mock_fred = MagicMock()
        mock_fred.get_series.side_effect = self._fred_side_effect(sids)

        with patch("src.phase1.ingest.macro.Fred", return_value=mock_fred):
            result = ingest_macro(base_config, force_refresh=True)

        for sid in sids:
            assert sid in result.columns, f"Series {sid} missing from output"

    def test_index_is_business_day_frequency(self, base_config):
        sids = base_config["macro_series"]
        mock_fred = MagicMock()
        mock_fred.get_series.side_effect = self._fred_side_effect(sids)

        with patch("src.phase1.ingest.macro.Fred", return_value=mock_fred):
            result = ingest_macro(base_config, force_refresh=True)

        assert isinstance(result.index, pd.DatetimeIndex)
        # Verify no weekends in index
        assert (result.index.dayofweek < 5).all(), "Index contains weekend dates"

    def test_no_nulls_after_forward_fill(self, base_config):
        """After resample + ffill, nulls should be negligible (≤1%)."""
        sids = base_config["macro_series"]
        mock_fred = MagicMock()
        mock_fred.get_series.side_effect = self._fred_side_effect(sids)

        with patch("src.phase1.ingest.macro.Fred", return_value=mock_fred):
            result = ingest_macro(base_config, force_refresh=True)

        null_rates = result.isna().mean()
        assert (null_rates <= 0.01).all(), \
            f"Unexpected null rates after ffill: {null_rates[null_rates > 0.01].to_dict()}"

    def test_parquet_written_to_correct_path(self, base_config):
        sids = base_config["macro_series"]
        mock_fred = MagicMock()
        mock_fred.get_series.side_effect = self._fred_side_effect(sids)
        expected_path = Path(base_config["raw_data_path"]) / "macro" / "macro_raw.parquet"

        with patch("src.phase1.ingest.macro.Fred", return_value=mock_fred):
            ingest_macro(base_config, force_refresh=True)

        assert expected_path.exists()

    def test_fallback_to_cache_on_fred_failure(self, base_config):
        """
        If FRED fails for a series that exists in cache, the cached value
        should be used rather than raising.
        """
        out_path = Path(base_config["raw_data_path"]) / "macro" / "macro_raw.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sids = base_config["macro_series"]

        # Write a cache with all series
        idx = pd.bdate_range("2022-01-01", periods=100)
        cached = pd.DataFrame(
            {s: np.random.rand(100) for s in sids}, index=idx
        )
        cached.to_parquet(out_path)

        # FRED fails for one series
        def flaky_fred(sid, **kwargs):
            if sid == sids[0]:
                raise ConnectionError("FRED down")
            return _make_fake_macro_series(sid)

        mock_fred = MagicMock()
        mock_fred.get_series.side_effect = flaky_fred

        with patch("src.phase1.ingest.macro.Fred", return_value=mock_fred):
            result = ingest_macro(base_config, force_refresh=True)

        # All columns should still be present despite one fetch failing
        for sid in sids:
            assert sid in result.columns


# ── Integration: cache round-trip against real on-disk data ───────────────────

class TestCacheRoundTrip:
    """
    These tests use the actual parquet files written by the live pipeline run.
    They test the cache path end-to-end without any mocking.
    Skipped if the files don't exist (e.g. fresh CI environment).
    """

    REAL_CONFIG = {
        "tickers": [
            "AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, JPM, GS, BAC",
            "XOM, CVX, LLY, JNJ, UNH, WMT, PG, KO, PEP, MCD",
            "SPY, QQQ, IWM, GLD, TLT, USO, VNQ, HYG, EEM, DIA",
        ],
        "date_range": {"start": "2019-01-01", "end": "2024-12-31"},
        "macro_series": ["DGS10", "VIXCLS", "CPIAUCSL", "UNRATE", "FEDFUNDS"],
        "raw_data_path": "data/raw",
        "processed_data_path": "data/processed",
        "feature_store_path": "data/features",
    }

    @pytest.mark.skipif(
        not Path("data/raw/ohlcv/ohlcv_raw.parquet").exists(),
        reason="Live OHLCV data not present — run the pipeline first",
    )
    def test_ohlcv_cache_loads_30_tickers(self):
        df = ingest_ohlcv(self.REAL_CONFIG, force_refresh=False)
        tickers = _parse_tickers(self.REAL_CONFIG)
        for t in tickers:
            assert f"Close_{t}" in df.columns

    @pytest.mark.skipif(
        not Path("data/raw/ohlcv/ohlcv_raw.parquet").exists(),
        reason="Live OHLCV data not present — run the pipeline first",
    )
    def test_ohlcv_cache_covers_full_date_range(self):
        df = ingest_ohlcv(self.REAL_CONFIG, force_refresh=False)
        assert df.index.min() <= pd.Timestamp("2019-01-10")
        assert df.index.max() >= pd.Timestamp("2024-12-20")

    @pytest.mark.skipif(
        not Path("data/raw/ohlcv/ohlcv_raw.parquet").exists(),
        reason="Live OHLCV data not present — run the pipeline first",
    )
    def test_ohlcv_has_at_least_1200_rows(self):
        df = ingest_ohlcv(self.REAL_CONFIG, force_refresh=False)
        assert len(df) >= 1200

    @pytest.mark.skipif(
        not Path("data/raw/macro/macro_raw.parquet").exists(),
        reason="Live macro data not present — run the pipeline first",
    )
    def test_macro_cache_has_all_five_series(self):
        df = ingest_macro(self.REAL_CONFIG, force_refresh=False)
        for sid in self.REAL_CONFIG["macro_series"]:
            assert sid in df.columns

    @pytest.mark.skipif(
        not Path("data/raw/macro/macro_raw.parquet").exists(),
        reason="Live macro data not present — run the pipeline first",
    )
    def test_macro_cache_no_weekends(self):
        df = ingest_macro(self.REAL_CONFIG, force_refresh=False)
        assert (df.index.dayofweek < 5).all()
