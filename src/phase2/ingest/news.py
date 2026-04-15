"""
NewsAPI ingestion for Phase 2.

Fetches recent news articles per ticker and caches them as JSON so the
API is never hit twice for the same ticker.

IMPORTANT: The free NewsAPI tier returns at most 100 articles per query
and only covers the past 30 days. For the full 2019-2024 date range the
NLP feature columns will be sparse — this is expected and handled by
forward-fill (up to 5 days) + zero-fill in align.py.
"""

import os
import json
import time
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Top-level import so patch("src.phase2.ingest.news.NewsApiClient") works in tests.
# Set to None when the package is not installed; ingest_news() will warn and exit early.
try:
    from newsapi import NewsApiClient
except ImportError:
    NewsApiClient = None  # type: ignore[assignment,misc]


def ingest_news(config: dict) -> pd.DataFrame:
    """
    Fetch news articles for every ticker in config["ticker_to_company"].

    Requires NEWS_API_KEY in .env. If the key is absent, returns an empty
    DataFrame with the expected columns so downstream code keeps working.

    Parameters
    ----------
    config : dict
        Parsed phase2_config.yaml.

    Returns
    -------
    pd.DataFrame
        Columns: ticker, date, title, description, content, source.
        One row per article. Sorted by date ascending.
    """
    api_key = os.getenv("NEWS_API_KEY", "").strip()
    out_dir = Path(config["raw_nlp_path"]) / "news_raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not api_key:
        print("NEWS_API_KEY not set -- skipping NewsAPI ingestion.")
        print("  Add your key to .env to enable news sentiment features.")
        return _empty_news_df()

    if NewsApiClient is None:
        print("newsapi-python not installed. Run: pip install newsapi-python")
        return _empty_news_df()

    api = NewsApiClient(api_key=api_key)
    all_articles: list[dict] = []

    for ticker, company in config["ticker_to_company"].items():
        cache_path = out_dir / f"{ticker}_news.json"

        if cache_path.exists():
            with open(cache_path, encoding="utf-8") as f:
                articles = json.load(f)
            print(f"  {ticker}: {len(articles)} cached articles")
        else:
            articles = []
            try:
                response = api.get_everything(
                    q=company,
                    language="en",
                    sort_by="publishedAt",
                    page_size=100,
                )
                articles = response.get("articles", [])
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(articles, f, ensure_ascii=False)
                print(f"  {ticker}: fetched {len(articles)} articles")
                time.sleep(1)   # Respect free-tier rate limit
            except Exception as exc:
                print(f"  {ticker}: fetch failed -- {exc}")

        for art in articles:
            all_articles.append({
                "ticker":       ticker,
                "published_at": art.get("publishedAt", ""),
                "title":        art.get("title", "") or "",
                "description":  art.get("description", "") or "",
                "content":      art.get("content", "") or "",
                "source":       art.get("source", {}).get("name", ""),
            })

    if not all_articles:
        return _empty_news_df()

    df = pd.DataFrame(all_articles)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df["date"] = df["published_at"].dt.normalize().dt.tz_localize(None)
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    print(f"News ingestion complete: {len(df)} articles across {df['ticker'].nunique()} tickers")
    return df


def _empty_news_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["ticker", "date", "title", "description", "content", "source"])
