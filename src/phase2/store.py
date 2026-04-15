"""
Phase 2 pipeline orchestrator.

Sequence:
    1. Load Phase 1 feature matrix (read-only)
    2. Ingest news (NewsAPI, cached)
    3. Ingest SEC EDGAR 8-K filings (cached)
    4. Run FinBERT on news text -> daily sentiment features
    5. Run FinBERT on earnings text -> daily earnings sentiment
    6. Compute forward-looking language features from filings
    7. Compute NER entity mention features from filings
    8. Align all NLP features to trading day index
    9. Validate alignment and run leakage check
   10. Merge into feature_matrix_v2.parquet
   11. Log to MLflow

Entry point:
    python -m src.phase2.store
"""

import yaml
import mlflow
import pandas as pd
from pathlib import Path

from src.phase2.ingest.news import ingest_news
from src.phase2.ingest.edgar import ingest_edgar_filings
from src.phase2.nlp.finbert import load_finbert, compute_daily_sentiment
from src.phase2.nlp.forward_looking import compute_forward_looking_features
from src.phase2.nlp.ner import compute_ner_features
from src.phase2.align import align_to_trading_days, validate_alignment
from src.phase1.validate.leakage_check import check_no_lookahead


def run_nlp_pipeline(config: dict) -> pd.DataFrame:
    """
    Execute the full Phase 2 pipeline and return the enriched feature matrix.

    Parameters
    ----------
    config : dict
        Parsed phase2_config.yaml.

    Returns
    -------
    pd.DataFrame
        feature_matrix_v2 (also written to disk).
    """
    feature_store = Path(config["feature_store_path"])
    ffill_limit   = config.get("sentiment_ffill_limit", 5)

    mlflow.set_experiment("phase2_nlp_pipeline")

    with mlflow.start_run(run_name="nlp_pipeline"):

        # ── Step 1: Load Phase 1 feature matrix ──────────────────────────────
        fm_path = feature_store / "feature_matrix.parquet"
        if not fm_path.exists():
            raise FileNotFoundError(
                f"Phase 1 feature matrix not found at {fm_path}. "
                "Run src.phase1.store first."
            )
        feature_matrix = pd.read_parquet(fm_path)
        trading_index  = feature_matrix.index
        print(f"[1/10] Phase 1 matrix loaded: {feature_matrix.shape}")

        # ── Step 2: Ingest news ───────────────────────────────────────────────
        print("\n[2/10] Ingesting news...")
        news_df = ingest_news(config)

        # ── Step 3: Ingest EDGAR filings ──────────────────────────────────────
        print("\n[3/10] Ingesting SEC EDGAR 8-K filings...")
        filings_df = ingest_edgar_filings(config)

        # ── Step 4 & 5: FinBERT sentiment ─────────────────────────────────────
        tokenizer, model, device = load_finbert(config["finbert_model"])

        print("\n[4/10] FinBERT scoring: news articles...")
        if not news_df.empty:
            news_df["text"] = (
                news_df["title"].fillna("") + " " + news_df["description"].fillna("")
            ).str.strip()
            news_sentiment = compute_daily_sentiment(
                news_df, "text", "date", "ticker", tokenizer, model, device
            )
        else:
            news_sentiment = pd.DataFrame()

        print("\n[5/10] FinBERT scoring: earnings call filings...")
        if not filings_df.empty:
            earnings_sentiment = compute_daily_sentiment(
                filings_df, "text", "date", "ticker", tokenizer, model, device
            )
        else:
            earnings_sentiment = pd.DataFrame()

        # ── Step 6: Forward-looking language ──────────────────────────────────
        print("\n[6/10] Forward-looking language features...")
        fl_features = compute_forward_looking_features(filings_df)

        # ── Step 7: NER features ──────────────────────────────────────────────
        print("\n[7/10] NER entity features...")
        ner_features = compute_ner_features(filings_df, config.get("spacy_model", "en_core_web_sm"))

        # ── Step 8: Align to trading day index ───────────────────────────────
        print("\n[8/10] Aligning NLP features to trading day index...")
        nlp_blocks = {
            "news_sentiment":     news_sentiment,
            "earnings_sentiment": earnings_sentiment,
            "forward_looking":    fl_features,
            "ner":                ner_features,
        }
        aligned: dict[str, pd.DataFrame] = {}
        for name, df in nlp_blocks.items():
            aligned[name] = align_to_trading_days(df, trading_index, ffill_limit)

        # ── Step 9: Validate alignment + leakage check ────────────────────────
        print("\n[9/10] Validating alignment...")
        for name, df in aligned.items():
            validate_alignment(df, feature_matrix, name)

        # Leakage check: build a temporary target column
        spy_close = pd.read_parquet(
            Path(config.get("raw_data_path", "data/raw")) / "ohlcv" / "ohlcv_raw.parquet"
        )["Close_SPY"]
        target = spy_close.pct_change().shift(-1).reindex(trading_index)

        combined_nlp = pd.concat(list(aligned.values()), axis=1)
        if not combined_nlp.empty:
            check_df = combined_nlp.copy()
            check_df["_target"] = target
            leakage_report = check_no_lookahead(check_df.dropna(how="all"), "_target")
            mlflow.log_param("leakage_check_passed", leakage_report.passed)
            if not leakage_report.passed:
                raise RuntimeError(
                    f"Leakage detected in NLP features: {leakage_report.leaky_columns[:3]}"
                )

        # ── Step 10: Merge and save ───────────────────────────────────────────
        print("\n[10/10] Merging and saving feature_matrix_v2...")
        feature_matrix_v2 = pd.concat(
            [feature_matrix] + list(aligned.values()), axis=1
        )

        out_path = feature_store / "feature_matrix_v2.parquet"
        feature_matrix_v2.to_parquet(out_path)

        nlp_cols_added = feature_matrix_v2.shape[1] - feature_matrix.shape[1]
        missing_rate   = float(feature_matrix_v2.isna().mean().mean())

        print(f"\nPhase 2 complete.")
        print(f"  feature_matrix_v2 : {feature_matrix_v2.shape}")
        print(f"  NLP columns added : {nlp_cols_added}")
        print(f"  Missing value rate: {missing_rate:.3f}")

        # ── MLflow logging ────────────────────────────────────────────────────
        mlflow.log_params({
            "finbert_model":    config["finbert_model"],
            "n_tickers":        len(config["ticker_to_company"]),
            "ffill_limit":      ffill_limit,
            "news_articles":    len(news_df),
            "edgar_filings":    len(filings_df),
        })
        mlflow.log_metrics({
            "nlp_columns_added":  nlp_cols_added,
            "total_features":     feature_matrix_v2.shape[1],
            "missing_value_rate": round(missing_rate, 4),
        })

    return feature_matrix_v2


if __name__ == "__main__":
    with open("configs/phase2_config.yaml") as f:
        cfg = yaml.safe_load(f)
    # Phase 1 raw path needed for leakage check target
    cfg["raw_data_path"] = "data/raw"
    run_nlp_pipeline(cfg)
