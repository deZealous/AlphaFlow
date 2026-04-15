"""
Phase 1 pipeline orchestrator.

Runs the full sequence:
    1. Ingest OHLCV (yfinance)
    2. Ingest macro (FRED)
    3. Validate raw data (schema checks)
    4. Engineer technical features (30 tickers x 17 features)
    5. Join macro features (shifted 1 day, forward-filled after join)
    6. Leakage check on full feature matrix
    7. Validate feature matrix (shape, nulls, no future dates)
    8. Save to data/features/feature_matrix.parquet
    9. Log run metadata to MLflow

Entry point:
    python -m src.phase1.store
    or imported and called as run_feature_pipeline(config)
"""

import yaml
import mlflow
import pandas as pd
from pathlib import Path

from src.phase1.ingest.ohlcv import ingest_ohlcv, _parse_tickers
from src.phase1.ingest.macro import ingest_macro
from src.phase1.validate.schema import validate_ohlcv, validate_macro, validate_feature_matrix
from src.phase1.validate.leakage_check import check_no_lookahead
from src.phase1.features.technical import engineer_all_features


def _join_macro(feature_matrix: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join macro series onto the feature matrix and forward-fill any gaps
    introduced by holidays that exist in the feature index but not in macro.

    Macro series are shifted by 1 trading day before joining — daily series
    like DGS10 and VIXCLS are published same-day, so they need the same
    lookahead guard as the technical features.

    Parameters
    ----------
    feature_matrix : pd.DataFrame
        Wide technical feature matrix (index = trading days).
    macro : pd.DataFrame
        Macro DataFrame (index = business days, already forward-filled).

    Returns
    -------
    pd.DataFrame
        feature_matrix with macro columns appended. No rows are dropped.
    """
    # Shift macro by 1 day — same lookahead guard as technical features
    macro_lagged = macro.shift(1)

    # Left join: keep only rows that exist in feature_matrix
    joined = feature_matrix.join(macro_lagged, how="left")

    # Forward-fill macro columns only — handles residual NaNs from
    # trading-day / business-day calendar misalignment (e.g. US holidays
    # that are business days in pandas but not in FRED data)
    macro_cols = macro.columns.tolist()
    joined[macro_cols] = joined[macro_cols].ffill()

    return joined


def run_feature_pipeline(config: dict, force_refresh: bool = False) -> pd.DataFrame:
    """
    Execute the full Phase 1 pipeline and return the saved feature matrix.

    Parameters
    ----------
    config : dict
        Parsed phase1_config.yaml.

    Returns
    -------
    pd.DataFrame
        The final feature matrix (also written to disk).

    Raises
    ------
    RuntimeError
        If any validation step fails hard (schema or leakage).
    """
    tickers = _parse_tickers(config)

    mlflow.set_experiment("phase1_data_pipeline")

    with mlflow.start_run(run_name="feature_pipeline"):

        # ── Step 1 & 2: Ingest ────────────────────────────────────────────────
        print("\n[1/7] Ingesting OHLCV data...")
        ohlcv = ingest_ohlcv(config, force_refresh=force_refresh)

        print("\n[2/7] Ingesting macro data...")
        macro = ingest_macro(config, force_refresh=force_refresh)

        # ── Step 3: Validate raw data ─────────────────────────────────────────
        print("\n[3/7] Validating raw data...")
        ohlcv_report = validate_ohlcv(ohlcv, tickers)
        macro_report = validate_macro(macro, config["macro_series"])

        if not ohlcv_report.passed:
            raise RuntimeError(f"OHLCV schema validation failed: {ohlcv_report.failures}")
        if not macro_report.passed:
            raise RuntimeError(f"Macro schema validation failed: {macro_report.failures}")

        # ── Step 4: Feature engineering ───────────────────────────────────────
        print("\n[4/7] Engineering technical features...")
        windows = config.get("feature_windows", [5, 10, 20, 60])
        feature_matrix = engineer_all_features(ohlcv, tickers, windows)

        # ── Step 5: Join macro ────────────────────────────────────────────────
        print("\n[5/7] Joining macro features...")
        feature_matrix = _join_macro(feature_matrix, macro)
        print(f"  Matrix after macro join: {feature_matrix.shape}")

        # ── Step 6: Leakage check ─────────────────────────────────────────────
        print("\n[6/7] Running leakage check...")

        # Build a target column purely for the leakage check — next-day SPY return.
        # It is NOT saved into the feature matrix; Phase 5 builds its own targets.
        spy_close = ohlcv["Close_SPY"]
        target_series = spy_close.pct_change().shift(-1)
        check_df = feature_matrix.copy()
        check_df["_target_spy_1d"] = target_series

        leakage_report = check_no_lookahead(
            check_df.dropna(), target_col="_target_spy_1d"
        )
        mlflow.log_param("leakage_check_passed", leakage_report.passed)
        if leakage_report.leaky_columns:
            mlflow.log_param(
                "leaky_columns",
                [col for col, _, _ in leakage_report.leaky_columns]
            )
        if not leakage_report.passed:
            raise RuntimeError(
                f"Leakage detected in {len(leakage_report.leaky_columns)} columns. "
                "Pipeline aborted. Review leakage_report for details."
            )

        # ── Step 7: Validate feature matrix ───────────────────────────────────
        print("\n[7/7] Validating feature matrix...")
        fm_report = validate_feature_matrix(feature_matrix)
        if not fm_report.passed:
            raise RuntimeError(f"Feature matrix validation failed: {fm_report.failures}")

        # ── Save ──────────────────────────────────────────────────────────────
        out_path = Path(config["feature_store_path"]) / "feature_matrix.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        feature_matrix.to_parquet(out_path)

        null_count   = int(feature_matrix.isna().sum().sum())
        null_pct_max = float(feature_matrix.isna().mean().max() * 100)

        print(f"\nFeature matrix saved -> {out_path}")
        print(f"  Shape      : {feature_matrix.shape[0]} rows x {feature_matrix.shape[1]} columns")
        print(f"  Date range : {feature_matrix.index.min().date()} to {feature_matrix.index.max().date()}")
        print(f"  Total nulls: {null_count}  (max column null rate: {null_pct_max:.1f}%)")

        # ── MLflow logging ────────────────────────────────────────────────────
        mlflow.log_params({
            "n_tickers":          len(tickers),
            "n_macro_series":     len(config["macro_series"]),
            "feature_windows":    str(windows),
            "date_start":         config["date_range"]["start"],
            "date_end":           config["date_range"]["end"],
        })
        mlflow.log_metrics({
            "n_rows":             feature_matrix.shape[0],
            "n_cols":             feature_matrix.shape[1],
            "null_count_total":   null_count,
            "null_pct_max_col":   round(null_pct_max, 4),
        })

        print("\nMLflow run logged.")

    return feature_matrix


if __name__ == "__main__":
    with open("configs/phase1_config.yaml") as f:
        cfg = yaml.safe_load(f)
    run_feature_pipeline(cfg)
