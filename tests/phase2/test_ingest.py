"""
Tests for Phase 2 ingestion modules (news and EDGAR).
All network calls are mocked -- no live API hits.
"""

import json
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.phase2.ingest.news import ingest_news, _empty_news_df
from src.phase2.ingest.edgar import (
    get_company_filings,
    fetch_filing_text,
    ingest_edgar_filings,
    _format_accession,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def base_config(tmp_path):
    return {
        "ticker_to_company": {"AAPL": "Apple", "MSFT": "Microsoft"},
        "ticker_to_cik":     {"AAPL": "0000320193", "MSFT": "0000789019"},
        "date_range":        {"start": "2022-01-01", "end": "2023-12-31"},
        "raw_nlp_path":      str(tmp_path / "raw" / "nlp"),
        "edgar_filings_cap": 5,
    }


def _make_fake_articles(ticker: str, n: int = 3) -> list[dict]:
    return [
        {
            "publishedAt": f"2023-0{i+1}-15T10:00:00Z",
            "title":       f"{ticker} reports quarterly earnings",
            "description": f"{ticker} beats estimates",
            "content":     "Record revenue growth this quarter.",
            "source":      {"name": "Reuters"},
        }
        for i in range(n)
    ]


def _make_fake_submissions(n_8k: int = 3) -> dict:
    """Mimics the EDGAR /submissions/CIK*.json response."""
    return {
        "filings": {
            "recent": {
                "form":           ["8-K"] * n_8k + ["10-K"],
                "filingDate":     [f"2022-0{i+1}-10" for i in range(n_8k)] + ["2022-12-31"],
                "accessionNumber": [f"0001193125-22-00000{i}" for i in range(n_8k)] + ["0001193125-22-999999"],
            }
        }
    }


# ── _format_accession ─────────────────────────────────────────────────────────

class TestFormatAccession:
    def test_dashes_inserted_correctly(self):
        raw = "000119312522000001"
        result = _format_accession(raw)
        assert result == "0001193125-22-000001"

    def test_length_preserved(self):
        raw = "000119312522000001"
        result = _format_accession(raw)
        # format is XXXXXXXXXX-YY-ZZZZZZ (10 + 2 + 6 + 2 dashes = 20)
        assert len(result) == 20


# ── news ingestion ────────────────────────────────────────────────────────────

class TestIngestNews:
    def test_returns_empty_df_when_no_api_key(self, base_config):
        with patch.dict("os.environ", {"NEWS_API_KEY": ""}):
            result = ingest_news(base_config)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_returns_expected_columns_when_empty(self, base_config):
        with patch.dict("os.environ", {"NEWS_API_KEY": ""}):
            result = ingest_news(base_config)
        expected_cols = {"ticker", "date", "title", "description", "content", "source"}
        assert expected_cols.issubset(set(result.columns))

    def test_cache_hit_skips_api_call(self, base_config):
        """Cache present for all tickers -> get_everything() never called."""
        out_dir = Path(base_config["raw_nlp_path"]) / "news_raw"
        out_dir.mkdir(parents=True, exist_ok=True)
        articles = _make_fake_articles("AAPL")
        (out_dir / "AAPL_news.json").write_text(json.dumps(articles), encoding="utf-8")
        (out_dir / "MSFT_news.json").write_text(json.dumps([]), encoding="utf-8")

        mock_api_instance = MagicMock()
        with patch.dict("os.environ", {"NEWS_API_KEY": "fake_key"}):
            with patch("src.phase2.ingest.news.NewsApiClient", return_value=mock_api_instance):
                result = ingest_news(base_config)
                # Client may be instantiated, but the API must never be queried
                mock_api_instance.get_everything.assert_not_called()

        assert len(result) == len(articles)

    def test_date_column_is_datetime(self, base_config):
        out_dir = Path(base_config["raw_nlp_path"]) / "news_raw"
        out_dir.mkdir(parents=True, exist_ok=True)
        articles = _make_fake_articles("AAPL")
        (out_dir / "AAPL_news.json").write_text(json.dumps(articles), encoding="utf-8")
        (out_dir / "MSFT_news.json").write_text(json.dumps([]), encoding="utf-8")

        with patch.dict("os.environ", {"NEWS_API_KEY": "fake_key"}):
            with patch("src.phase2.ingest.news.NewsApiClient"):
                result = ingest_news(base_config)

        assert pd.api.types.is_datetime64_any_dtype(result["date"])

    def test_sorted_by_date(self, base_config):
        out_dir = Path(base_config["raw_nlp_path"]) / "news_raw"
        out_dir.mkdir(parents=True, exist_ok=True)
        articles = _make_fake_articles("AAPL", n=5)
        (out_dir / "AAPL_news.json").write_text(json.dumps(articles), encoding="utf-8")
        (out_dir / "MSFT_news.json").write_text(json.dumps([]), encoding="utf-8")

        with patch.dict("os.environ", {"NEWS_API_KEY": "fake_key"}):
            with patch("src.phase2.ingest.news.NewsApiClient"):
                result = ingest_news(base_config)

        assert result["date"].is_monotonic_increasing


# ── EDGAR ingestion ───────────────────────────────────────────────────────────

class TestGetCompanyFilings:
    def test_filters_to_8k_only(self):
        fake_resp = MagicMock()
        fake_resp.json.return_value = _make_fake_submissions(n_8k=3)
        fake_resp.raise_for_status = MagicMock()

        with patch("src.phase2.ingest.edgar.requests.get", return_value=fake_resp):
            result = get_company_filings("0000320193", form_type="8-K")

        assert all(f["form"] == "8-K" for f in result)
        assert len(result) == 3

    def test_returns_empty_on_request_error(self):
        with patch("src.phase2.ingest.edgar.requests.get", side_effect=Exception("timeout")):
            result = get_company_filings("0000320193")
        assert result == []

    def test_accession_number_has_no_dashes(self):
        fake_resp = MagicMock()
        fake_resp.json.return_value = _make_fake_submissions(n_8k=1)
        fake_resp.raise_for_status = MagicMock()

        with patch("src.phase2.ingest.edgar.requests.get", return_value=fake_resp):
            result = get_company_filings("0000320193")

        assert "-" not in result[0]["accession_number"]


class TestIngestEdgarFilings:
    def test_uses_cache_when_present(self, base_config, tmp_path):
        out_dir = Path(base_config["raw_nlp_path"]) / "edgar_raw"
        out_dir.mkdir(parents=True, exist_ok=True)

        fake_df = pd.DataFrame({
            "ticker":    ["AAPL"],
            "date":      [pd.Timestamp("2022-02-01")],
            "accession": ["000119312522000001"],
            "text":      ["Apple reports record earnings."],
        })
        fake_df.to_parquet(out_dir / "AAPL_filings.parquet")
        fake_df.to_parquet(out_dir / "MSFT_filings.parquet")

        with patch("src.phase2.ingest.edgar.requests.get") as mock_get:
            result = ingest_edgar_filings(base_config)
            mock_get.assert_not_called()

        assert len(result) >= 1

    def test_returns_empty_df_on_all_failures(self, base_config):
        with patch("src.phase2.ingest.edgar.requests.get", side_effect=Exception("EDGAR down")):
            result = ingest_edgar_filings(base_config)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_result_has_required_columns(self, base_config, tmp_path):
        out_dir = Path(base_config["raw_nlp_path"]) / "edgar_raw"
        out_dir.mkdir(parents=True, exist_ok=True)

        fake_df = pd.DataFrame({
            "ticker":    ["AAPL"],
            "date":      [pd.Timestamp("2022-02-01")],
            "accession": ["000119312522000001"],
            "text":      ["Strong quarter."],
        })
        fake_df.to_parquet(out_dir / "AAPL_filings.parquet")
        fake_df.to_parquet(out_dir / "MSFT_filings.parquet")

        with patch("src.phase2.ingest.edgar.requests.get"):
            result = ingest_edgar_filings(base_config)

        for col in ["ticker", "date", "accession", "text"]:
            assert col in result.columns
