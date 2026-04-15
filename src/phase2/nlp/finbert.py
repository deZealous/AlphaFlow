"""
FinBERT sentiment pipeline.

Uses ProsusAI/finbert — a BERT model fine-tuned on financial text.
Label order (from model config): 0=positive, 1=negative, 2=neutral.

Scores every text in batches, then aggregates to daily ticker-level
features and lags by 1 trading day to prevent lookahead.
"""

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax


def load_finbert(model_name: str = "ProsusAI/finbert"):
    """
    Load FinBERT tokenizer and model. Downloads from HuggingFace on
    first call; subsequent calls use the local cache.

    Returns
    -------
    tuple[tokenizer, model, device_str]
    """
    print(f"Loading FinBERT: {model_name}")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"  FinBERT loaded on {device}")
    return tokenizer, model, device


def score_texts(
    texts: list[str],
    tokenizer,
    model,
    device: str,
    batch_size: int = 16,
) -> list[dict]:
    """
    Run FinBERT on a list of texts.

    Parameters
    ----------
    texts : list[str]
        Raw text strings. Empty strings score as neutral (0.0 net).
    batch_size : int
        Number of texts per forward pass. Reduce if OOM on GPU.

    Returns
    -------
    list[dict]
        One dict per text with keys: positive, negative, neutral,
        sentiment_score (positive - negative).
    """
    results = []

    for i in tqdm(range(0, len(texts), batch_size), desc="FinBERT scoring", leave=False):
        batch = texts[i : i + batch_size]

        # Replace empty strings with a neutral placeholder to keep batch size stable
        batch = [t if t.strip() else "neutral" for t in batch]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        probs = softmax(outputs.logits, dim=-1).cpu().numpy()

        for prob in probs:
            results.append({
                "positive":       float(prob[0]),
                "negative":       float(prob[1]),
                "neutral":        float(prob[2]),
                "sentiment_score": float(prob[0] - prob[1]),
            })

    return results


def compute_daily_sentiment(
    df: pd.DataFrame,
    text_col: str,
    date_col: str,
    ticker_col: str,
    tokenizer,
    model,
    device: str,
) -> pd.DataFrame:
    """
    Score all texts in df, aggregate to daily per-ticker features,
    pivot wide, and lag 1 trading day (lookahead guard).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain date_col, ticker_col, and text_col.

    Returns
    -------
    pd.DataFrame
        Wide DataFrame: index=date, columns={TICKER}_{metric}.
        All columns shifted forward by 1 day.
    """
    if df.empty:
        return pd.DataFrame()

    texts = df[text_col].fillna("").tolist()
    scores = score_texts(texts, tokenizer, model, device)

    scored = df[[date_col, ticker_col]].copy().reset_index(drop=True)
    scored["positive"]        = [s["positive"]        for s in scores]
    scored["negative"]        = [s["negative"]        for s in scores]
    scored["neutral"]         = [s["neutral"]         for s in scores]
    scored["sentiment_score"] = [s["sentiment_score"] for s in scores]

    # Aggregate: mean sentiment + article count per (date, ticker)
    daily = (
        scored.groupby([date_col, ticker_col])
        .agg(
            sentiment_mean  =("sentiment_score", "mean"),
            sentiment_std   =("sentiment_score", lambda x: x.std() if len(x) > 1 else 0.0),
            positive_mean   =("positive",        "mean"),
            negative_mean   =("negative",        "mean"),
            article_count   =("sentiment_score", "count"),
        )
        .reset_index()
    )

    # Pivot: rows=date, cols=(metric, ticker) -> flatten to {ticker}_{metric}
    daily_wide = daily.pivot(index=date_col, columns=ticker_col)
    daily_wide.columns = [f"{ticker}_{feat}" for feat, ticker in daily_wide.columns]
    daily_wide.index = pd.to_datetime(daily_wide.index)
    daily_wide = daily_wide.sort_index()

    # Lag 1 day: today's features use yesterday's published text
    daily_wide = daily_wide.shift(1)

    return daily_wide
