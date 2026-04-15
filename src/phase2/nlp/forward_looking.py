"""
Forward-looking language detector for earnings call text.

Management language about future expectations is a signal orthogonal to
historical price data. A bullish guidance statement ("we expect record
growth") carries information that momentum indicators cannot encode.

Two pattern sets are matched against each text:
    FORWARD_POSITIVE: optimistic forward language
    FORWARD_NEGATIVE: cautious / bearish forward language

The net score (fl_score) is (positive_hits - negative_hits) / total_hits,
normalised to [-1, +1]. Zero when no forward language is detected.
"""

import re
import pandas as pd

FORWARD_POSITIVE = [
    r"\bexpect(?:s|ed|ing)?\b.{0,60}\bgrowth\b",
    r"\banticipat(?:e|es|ed|ing)\b.{0,60}\bincrease\b",
    r"\bconfident\b",
    r"\bstrong pipeline\b",
    r"\brecord (?:revenue|earnings|growth|quarter)\b",
    r"\braccelerat(?:e|ing|ed)\b",
    r"\bupgrad(?:e|ing|ed)\b.{0,40}\bguidance\b",
    r"\brais(?:e|ing|ed)\b.{0,40}\bguidance\b",
    r"\boptimistic\b",
    r"\boutperform(?:ing|ed)?\b",
]

FORWARD_NEGATIVE = [
    r"\bheadwinds?\b",
    r"\buncertain(?:ty)?\b",
    r"\bchalleng(?:e|ing|es)\b",
    r"\bdecelerat(?:e|ing|ed)\b",
    r"\blower(?:ing|ed)?\b.{0,40}\bguidance\b",
    r"\bdown(?:ward|grade)\b",
    r"\brisk(?:s)?\b.{0,40}\bincrease\b",
    r"\bweaken(?:ing|ed)?\b",
    r"\bpressure(?:d|s)?\b",
    r"\bdisappoint(?:ing|ed|ment)?\b",
]


def score_forward_looking(text: str) -> dict:
    """
    Score a single text block for forward-looking language.

    Parameters
    ----------
    text : str
        Raw text (earnings call excerpt or news article).

    Returns
    -------
    dict
        fl_positive  : int  — count of positive forward-language matches
        fl_negative  : int  — count of negative forward-language matches
        fl_score     : float — net score in [-1, +1]; 0 when no match
    """
    if not isinstance(text, str) or not text.strip():
        return {"fl_positive": 0, "fl_negative": 0, "fl_score": 0.0}

    text_lower = text.lower()
    pos = sum(1 for p in FORWARD_POSITIVE if re.search(p, text_lower))
    neg = sum(1 for p in FORWARD_NEGATIVE if re.search(p, text_lower))
    total = pos + neg

    return {
        "fl_positive": pos,
        "fl_negative": neg,
        "fl_score": (pos - neg) / total if total > 0 else 0.0,
    }


def compute_forward_looking_features(filings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply forward-looking scoring to all filings, aggregate daily per
    ticker, pivot wide, and lag 1 trading day.

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

    scores = filings_df["text"].apply(score_forward_looking)
    scores_df = pd.DataFrame(scores.tolist())

    result = pd.concat(
        [filings_df[["date", "ticker"]].reset_index(drop=True), scores_df],
        axis=1,
    )

    daily = (
        result.groupby(["date", "ticker"])
        .agg(
            fl_score_mean=("fl_score",    "mean"),
            fl_positive  =("fl_positive", "sum"),
            fl_negative  =("fl_negative", "sum"),
        )
        .reset_index()
    )

    daily_wide = daily.pivot(index="date", columns="ticker")
    daily_wide.columns = [f"{ticker}_{feat}" for feat, ticker in daily_wide.columns]
    daily_wide.index = pd.to_datetime(daily_wide.index)
    daily_wide = daily_wide.sort_index()

    # Lag 1 day: earnings calls released after market close
    daily_wide = daily_wide.shift(1)

    return daily_wide
