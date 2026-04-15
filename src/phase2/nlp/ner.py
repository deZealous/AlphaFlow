"""
Named Entity Recognition features using spaCy.

Extracts entity mention counts from filing text. These act as proxy
signals for how much a document discusses organisations, products, and
geographies — which correlates with business complexity and news density.

spaCy model is lazy-loaded on first call so the module is importable
before `python -m spacy download en_core_web_sm` is run.
"""

import pandas as pd
from typing import Optional

ENTITY_TYPES = {"ORG", "PRODUCT", "GPE", "PERSON"}

_nlp = None   # lazy-loaded singleton


def _get_nlp(model_name: str = "en_core_web_sm"):
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load(model_name)
    return _nlp


def extract_entity_features(text: str, model_name: str = "en_core_web_sm") -> dict:
    """
    Run spaCy NER on text and return entity mention counts.

    Parameters
    ----------
    text : str
        Raw text (capped at 10 000 chars internally for speed).

    Returns
    -------
    dict
        entity_mention_count : total entities recognised
        org_mention_count    : ORG entities
        product_mention_count: PRODUCT entities
        gpe_mention_count    : GPE (geo-political) entities
        person_mention_count : PERSON entities
    """
    if not isinstance(text, str) or not text.strip():
        return {
            "entity_mention_count":  0,
            "org_mention_count":     0,
            "product_mention_count": 0,
            "gpe_mention_count":     0,
            "person_mention_count":  0,
        }

    nlp = _get_nlp(model_name)
    doc = nlp(text[:10_000])

    counts = {t: 0 for t in ENTITY_TYPES}
    for ent in doc.ents:
        if ent.label_ in ENTITY_TYPES:
            counts[ent.label_] += 1

    return {
        "entity_mention_count":  len(doc.ents),
        "org_mention_count":     counts["ORG"],
        "product_mention_count": counts["PRODUCT"],
        "gpe_mention_count":     counts["GPE"],
        "person_mention_count":  counts["PERSON"],
    }


def compute_ner_features(
    filings_df: pd.DataFrame,
    spacy_model: str = "en_core_web_sm",
) -> pd.DataFrame:
    """
    Run NER on every filing, aggregate to daily per-ticker counts,
    pivot wide, and lag 1 trading day.

    Parameters
    ----------
    filings_df : pd.DataFrame
        Must contain columns: date, ticker, text.

    Returns
    -------
    pd.DataFrame
        Wide DataFrame: index=date, columns={TICKER}_{metric}.
        All columns shifted forward by 1 day.
    """
    if filings_df.empty:
        return pd.DataFrame()

    records = []
    for _, row in filings_df.iterrows():
        feats = extract_entity_features(row["text"], spacy_model)
        feats["date"]   = row["date"]
        feats["ticker"] = row["ticker"]
        records.append(feats)

    df = pd.DataFrame(records)

    daily = (
        df.groupby(["date", "ticker"])
        .mean(numeric_only=True)
        .reset_index()
    )

    daily_wide = daily.pivot(index="date", columns="ticker")
    daily_wide.columns = [f"{ticker}_{feat}" for feat, ticker in daily_wide.columns]
    daily_wide.index = pd.to_datetime(daily_wide.index)
    daily_wide = daily_wide.sort_index()

    # Lag 1 day: filings available after market close on filing date
    daily_wide = daily_wide.shift(1)

    return daily_wide
