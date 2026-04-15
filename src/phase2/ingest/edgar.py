"""
SEC EDGAR 8-K filing scraper for Phase 2.

Fetches earnings-related 8-K filings for each ticker using the free
EDGAR REST API. No API key required — only a User-Agent email header.

Rate limit: EDGAR enforces max 10 requests/second. We sleep 0.5s
between requests to stay well under that limit.
"""

import re
import time
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm

EDGAR_SUBMISSIONS_BASE = "https://data.sec.gov"
EDGAR_ARCHIVE_BASE = "https://www.sec.gov"
HEADERS = {"User-Agent": "AlphaFlow hrm012@gmail.com"}

# Regex to skip metadata/XBRL viewer files when scanning filing indexes
_SKIP_FILE = re.compile(
    r"^(R\d+\.htm|.*\.(xml|xsd|zip|css|js|jpg|json)|"
    r"FilingSummary\.xml|Financial_Report\.xlsx|MetaLinks\.json|"
    r"Show\.js|report\.css|.*-index.*)",
    re.IGNORECASE,
)


def _format_accession(raw: str) -> str:
    """Convert '000119312524123456' -> '0001193125-24-123456'."""
    return f"{raw[:10]}-{raw[10:12]}-{raw[12:]}"


def get_company_filings(cik: str, form_type: str = "8-K") -> list[dict]:
    """
    Retrieve the list of recent 8-K filings for a company from EDGAR.

    Parameters
    ----------
    cik : str
        10-digit CIK (e.g. "0000320193" for Apple).
    form_type : str
        SEC form type to filter for.

    Returns
    -------
    list[dict]
        Each dict has keys: form, date, accession_number (dashes removed).
    """
    cik_int = str(int(cik))   # strip leading zeros for the URL
    url = f"{EDGAR_SUBMISSIONS_BASE}/submissions/CIK{cik.zfill(10)}.json"

    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
    except Exception as exc:
        print(f"  EDGAR: could not fetch submissions for CIK {cik} -- {exc}")
        return []

    data = r.json()
    recent = data.get("filings", {}).get("recent", {})

    forms   = recent.get("form", [])
    dates   = recent.get("filingDate", [])
    accnums = recent.get("accessionNumber", [])

    results = []
    for form, date, acc in zip(forms, dates, accnums):
        if form == form_type:
            results.append({
                "form":             form,
                "date":             date,
                "accession_number": acc.replace("-", ""),
                "cik_int":          cik_int,
            })

    return results


def fetch_filing_text(cik_int: str, accession_number: str) -> str:
    """
    Fetch the raw text of a single 8-K filing and strip HTML tags.

    Returns empty string on any failure so the caller can skip gracefully.
    """
    index_url = (
        f"{EDGAR_ARCHIVE_BASE}/Archives/edgar/data/{cik_int}"
        f"/{accession_number}/index.json"
    )

    try:
        r = requests.get(index_url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        index = r.json()
    except Exception:
        return ""

    # Find all .htm files, skip XBRL viewer (R1.htm etc.) and metadata files.
    # Pick the largest remaining file — that's typically the earnings release
    # (EX-99.1) or the primary 8-K document.
    candidates: list[tuple[int, str]] = []
    for item in index.get("directory", {}).get("item", []):
        name = item.get("name", "")
        if not name.lower().endswith(".htm"):
            continue
        if _SKIP_FILE.match(name):
            continue
        try:
            size = int(item.get("size", 0) or 0)
        except (ValueError, TypeError):
            size = 0
        candidates.append((size, name))

    if not candidates:
        return ""

    candidates.sort(reverse=True)   # largest first
    _, best_name = candidates[0]

    doc_url = (
        f"{EDGAR_ARCHIVE_BASE}/Archives/edgar/data/{cik_int}"
        f"/{accession_number}/{best_name}"
    )
    try:
        doc_r = requests.get(doc_url, headers=HEADERS, timeout=15)
        text = re.sub(r"<[^>]+>", " ", doc_r.text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    except Exception:
        return ""


def ingest_edgar_filings(config: dict) -> pd.DataFrame:
    """
    Scrape 8-K filings for every ticker in config["ticker_to_cik"].

    Results are cached per ticker as Parquet files. Re-running only
    fetches tickers whose cache does not yet exist.

    Parameters
    ----------
    config : dict
        Parsed phase2_config.yaml.

    Returns
    -------
    pd.DataFrame
        Columns: ticker, date, accession, text.
        Sorted by date ascending. Empty DataFrame if nothing was fetched.
    """
    out_dir = Path(config["raw_nlp_path"]) / "edgar_raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    start_date = config["date_range"]["start"]
    filings_cap = config.get("edgar_filings_cap", 20)

    all_frames: list[pd.DataFrame] = []

    for ticker, cik in tqdm(config["ticker_to_cik"].items(), desc="EDGAR scrape"):
        cache_path = out_dir / f"{ticker}_filings.parquet"

        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            print(f"  {ticker}: {len(df)} cached filings")
            all_frames.append(df)
            continue

        filings = get_company_filings(cik)
        # Filter to date range and cap
        end_date = config["date_range"].get("end", "2099-12-31")
        filings = [
            f for f in filings
            if start_date <= f["date"] <= end_date
        ][:filings_cap]

        records: list[dict] = []
        for filing in filings:
            text = fetch_filing_text(filing["cik_int"], filing["accession_number"])
            if text:
                records.append({
                    "ticker":    ticker,
                    "date":      pd.to_datetime(filing["date"]),
                    "accession": filing["accession_number"],
                    "text":      text[:5000],   # cap per filing to control memory
                })
            time.sleep(0.5)   # EDGAR rate limit

        if records:
            df = pd.DataFrame(records)
            df.to_parquet(cache_path)
            all_frames.append(df)
            print(f"  {ticker}: scraped {len(records)} filings")
        else:
            print(f"  {ticker}: no filings retrieved")

    if not all_frames:
        return pd.DataFrame(columns=["ticker", "date", "accession", "text"])

    result = pd.concat(all_frames, ignore_index=True)
    result = result.sort_values("date").reset_index(drop=True)
    print(f"EDGAR ingestion complete: {len(result)} filings across {result['ticker'].nunique()} tickers")
    return result
