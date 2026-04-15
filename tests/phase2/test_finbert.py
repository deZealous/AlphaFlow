"""
Tests for FinBERT sentiment, forward-looking language, and NER modules.

FinBERT tests load the real model (downloads ~400 MB on first run, then
cached). Forward-looking and NER tests are pure-Python and instant.
"""

import pytest
import pandas as pd
import numpy as np

from src.phase2.nlp.forward_looking import score_forward_looking, compute_forward_looking_features
from src.phase2.nlp.ner import extract_entity_features, compute_ner_features


# ── forward_looking ───────────────────────────────────────────────────────────

class TestScoreForwardLooking:
    def test_positive_text_scores_positive(self):
        text = "We are confident about record revenue growth and expect to accelerate next quarter."
        result = score_forward_looking(text)
        assert result["fl_score"] > 0
        assert result["fl_positive"] > 0

    def test_negative_text_scores_negative(self):
        text = "We face significant headwinds and uncertainty. We are lowering our guidance due to challenging conditions."
        result = score_forward_looking(text)
        assert result["fl_score"] < 0
        assert result["fl_negative"] > 0

    def test_neutral_text_scores_zero(self):
        text = "The company reported results for the quarter."
        result = score_forward_looking(text)
        assert result["fl_score"] == 0.0

    def test_empty_string_returns_zeros(self):
        result = score_forward_looking("")
        assert result == {"fl_positive": 0, "fl_negative": 0, "fl_score": 0.0}

    def test_none_returns_zeros(self):
        result = score_forward_looking(None)
        assert result == {"fl_positive": 0, "fl_negative": 0, "fl_score": 0.0}

    def test_score_bounded(self):
        text = "We expect record growth, accelerating pipeline, confident strong record revenue."
        result = score_forward_looking(text)
        assert -1.0 <= result["fl_score"] <= 1.0

    def test_returns_all_keys(self):
        result = score_forward_looking("Some text.")
        assert set(result.keys()) == {"fl_positive", "fl_negative", "fl_score"}


class TestComputeForwardLookingFeatures:
    def _make_filings(self) -> pd.DataFrame:
        return pd.DataFrame({
            "ticker": ["AAPL", "AAPL", "MSFT"],
            "date":   pd.to_datetime(["2022-02-01", "2022-05-01", "2022-02-01"]),
            "text":   [
                "We are confident about record growth and expect acceleration.",
                "We face headwinds and uncertainty challenging conditions.",
                "We anticipate increase in revenue, confident outlook.",
            ],
        })

    def test_returns_dataframe(self):
        result = compute_forward_looking_features(self._make_filings())
        assert isinstance(result, pd.DataFrame)

    def test_index_is_datetime(self):
        result = compute_forward_looking_features(self._make_filings())
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_columns_follow_naming_convention(self):
        result = compute_forward_looking_features(self._make_filings())
        for col in result.columns:
            # Format: {TICKER}_{metric}
            parts = col.split("_", 1)
            assert len(parts) == 2, f"Column '{col}' does not follow TICKER_metric convention"

    def test_shift_applied(self):
        """First row should be NaN because of .shift(1)."""
        result = compute_forward_looking_features(self._make_filings())
        # After shift(1), the earliest date row should be all NaN
        assert result.iloc[0].isna().all()

    def test_empty_input_returns_empty(self):
        result = compute_forward_looking_features(pd.DataFrame())
        assert result.empty


# ── NER ───────────────────────────────────────────────────────────────────────

class TestExtractEntityFeatures:
    def test_returns_all_keys(self):
        result = extract_entity_features("Apple reported earnings.")
        expected = {
            "entity_mention_count", "org_mention_count",
            "product_mention_count", "gpe_mention_count", "person_mention_count",
        }
        assert expected.issubset(set(result.keys()))

    def test_org_detected_in_company_text(self):
        result = extract_entity_features(
            "Apple Inc and Microsoft Corporation announced a partnership."
        )
        assert result["org_mention_count"] >= 1

    def test_empty_text_returns_zeros(self):
        result = extract_entity_features("")
        assert all(v == 0 for v in result.values())

    def test_none_returns_zeros(self):
        result = extract_entity_features(None)
        assert all(v == 0 for v in result.values())

    def test_counts_are_non_negative(self):
        result = extract_entity_features("The FDA approved the drug in New York.")
        assert all(v >= 0 for v in result.values())


class TestComputeNerFeatures:
    def _make_filings(self) -> pd.DataFrame:
        return pd.DataFrame({
            "ticker": ["AAPL", "MSFT"],
            "date":   pd.to_datetime(["2022-02-01", "2022-02-01"]),
            "text":   [
                "Apple Inc reported strong results in California.",
                "Microsoft Azure grew 30% in the United States.",
            ],
        })

    def test_returns_dataframe(self):
        result = compute_ner_features(self._make_filings())
        assert isinstance(result, pd.DataFrame)

    def test_index_is_datetime(self):
        result = compute_ner_features(self._make_filings())
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_shift_applied(self):
        result = compute_ner_features(self._make_filings())
        assert result.iloc[0].isna().all()

    def test_empty_input_returns_empty(self):
        result = compute_ner_features(pd.DataFrame())
        assert result.empty


# ── FinBERT (requires model download ~400 MB) ─────────────────────────────────

@pytest.mark.slow
class TestFinBERT:
    """
    Marked slow -- skipped in fast CI. Run with: pytest -m slow
    Requires torch + transformers installed and HuggingFace cache available.
    """

    @pytest.fixture(scope="class")
    def finbert(self):
        from src.phase2.nlp.finbert import load_finbert
        return load_finbert()

    def test_output_shape(self, finbert):
        from src.phase2.nlp.finbert import score_texts
        tokenizer, model, device = finbert
        texts = [
            "Apple reported record quarterly earnings.",
            "Inflation fears weigh on markets.",
        ]
        scores = score_texts(texts, tokenizer, model, device)
        assert len(scores) == 2

    def test_output_keys(self, finbert):
        from src.phase2.nlp.finbert import score_texts
        tokenizer, model, device = finbert
        scores = score_texts(["Strong revenue growth."], tokenizer, model, device)
        assert set(scores[0].keys()) == {"positive", "negative", "neutral", "sentiment_score"}

    def test_probs_sum_to_one(self, finbert):
        from src.phase2.nlp.finbert import score_texts
        tokenizer, model, device = finbert
        scores = score_texts(["Quarterly results exceeded expectations."], tokenizer, model, device)
        s = scores[0]
        total = s["positive"] + s["negative"] + s["neutral"]
        assert abs(total - 1.0) < 1e-5

    def test_sentiment_direction(self, finbert):
        from src.phase2.nlp.finbert import score_texts
        tokenizer, model, device = finbert
        pos = score_texts(
            ["Company smashes earnings expectations with record revenue growth."],
            tokenizer, model, device
        )[0]["sentiment_score"]
        neg = score_texts(
            ["Company misses estimates, warns of severe headwinds and losses ahead."],
            tokenizer, model, device
        )[0]["sentiment_score"]
        assert pos > neg

    def test_empty_text_handled(self, finbert):
        from src.phase2.nlp.finbert import score_texts
        tokenizer, model, device = finbert
        scores = score_texts([""], tokenizer, model, device)
        assert len(scores) == 1
