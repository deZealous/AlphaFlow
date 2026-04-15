"""
Tests for NLP feature alignment to the trading day index.
"""

import pytest
import numpy as np
import pandas as pd

from src.phase2.align import align_to_trading_days, validate_alignment


def _trading_idx(n: int = 100) -> pd.DatetimeIndex:
    return pd.bdate_range("2022-01-01", periods=n)


def _sparse_nlp(trading_idx: pd.DatetimeIndex, fill_fraction: float = 0.2) -> pd.DataFrame:
    """NLP features on only a fraction of trading days (sparse, like real data)."""
    n = len(trading_idx)
    selected = np.random.choice(n, size=int(n * fill_fraction), replace=False)
    selected_idx = trading_idx[sorted(selected)]
    return pd.DataFrame(
        {"AAPL_sentiment_mean": np.random.randn(len(selected_idx))},
        index=selected_idx,
    )


class TestAlignToTradingDays:
    def test_output_index_matches_trading_index(self):
        trading = _trading_idx(100)
        nlp = _sparse_nlp(trading)
        result = align_to_trading_days(nlp, trading)
        assert result.index.equals(trading)

    def test_no_nans_in_output(self):
        trading = _trading_idx(100)
        nlp = _sparse_nlp(trading)
        result = align_to_trading_days(nlp, trading)
        assert not result.isna().any().any()

    def test_forward_fill_propagates_values(self):
        """A sentiment on Monday should appear on Tuesday (day+1)."""
        trading = pd.bdate_range("2022-01-03", periods=5)   # Mon-Fri
        nlp = pd.DataFrame({"AAPL_s": [0.8]}, index=[trading[0]])
        result = align_to_trading_days(nlp, trading, ffill_limit=5)
        assert result.loc[trading[1], "AAPL_s"] == pytest.approx(0.8)

    def test_ffill_limit_respected(self):
        """After ffill_limit days of no new data, values reset to 0."""
        trading = pd.bdate_range("2022-01-03", periods=10)
        # Sentiment only on day 0
        nlp = pd.DataFrame({"AAPL_s": [1.0]}, index=[trading[0]])
        result = align_to_trading_days(nlp, trading, ffill_limit=3)
        # Days 1, 2, 3 should carry 1.0 (ffill_limit=3)
        assert result.loc[trading[3], "AAPL_s"] == pytest.approx(1.0)
        # Day 4 (beyond limit) should be 0
        assert result.loc[trading[4], "AAPL_s"] == pytest.approx(0.0)

    def test_empty_nlp_returns_zero_df(self):
        trading = _trading_idx(50)
        nlp = pd.DataFrame(columns=["AAPL_sentiment_mean"])
        result = align_to_trading_days(nlp, trading)
        assert result.shape == (50, 1)
        assert (result == 0.0).all().all()

    def test_columns_preserved(self):
        trading = _trading_idx(50)
        nlp = pd.DataFrame(
            {"AAPL_sentiment_mean": [0.5], "MSFT_sentiment_mean": [-0.3]},
            index=[trading[5]],
        )
        result = align_to_trading_days(nlp, trading)
        assert set(result.columns) == {"AAPL_sentiment_mean", "MSFT_sentiment_mean"}

    def test_all_columns_numeric(self):
        trading = _trading_idx(50)
        nlp = _sparse_nlp(trading)
        result = align_to_trading_days(nlp, trading)
        for col in result.columns:
            assert pd.api.types.is_numeric_dtype(result[col])


class TestValidateAlignment:
    def test_passes_when_indices_match(self):
        trading = _trading_idx(50)
        nlp = pd.DataFrame({"feat": np.zeros(50)}, index=trading)
        fm  = pd.DataFrame({"price": np.ones(50)}, index=trading)
        assert validate_alignment(nlp, fm, "test") is True

    def test_fails_when_indices_differ(self):
        trading_a = _trading_idx(50)
        trading_b = _trading_idx(60)
        nlp = pd.DataFrame({"feat": np.zeros(50)}, index=trading_a)
        fm  = pd.DataFrame({"price": np.ones(60)}, index=trading_b)
        assert validate_alignment(nlp, fm, "test") is False

    def test_warns_but_passes_on_residual_nans(self, capsys):
        trading = _trading_idx(10)
        nlp = pd.DataFrame({"feat": [np.nan] + [0.0] * 9}, index=trading)
        fm  = pd.DataFrame({"price": np.ones(10)}, index=trading)
        result = validate_alignment(nlp, fm, "test")
        assert result is True
        captured = capsys.readouterr()
        assert "WARN" in captured.out
