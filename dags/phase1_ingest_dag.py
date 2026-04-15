"""
Airflow DAG — Phase 1: daily data ingestion, validation, and feature engineering.

Schedule : weekdays at 06:00 UTC (markets closed, pre-open data available)
Tasks    : ingest_ohlcv >> validate_and_engineer
           ingest_macro  ──┘
Retries  : 2 per task, 5-minute back-off

To run locally (development):
    airflow standalone
    airflow dags trigger phase1_data_pipeline

To backfill a single day:
    airflow dags backfill phase1_data_pipeline \
        --start-date 2024-01-15 --end-date 2024-01-15
"""

from __future__ import annotations

import os
import yaml
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator


# ── Config ────────────────────────────────────────────────────────────────────

# Resolve config path relative to the repo root, not the DAG file location.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _REPO_ROOT / "configs" / "phase1_config.yaml"


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ── Task callables ────────────────────────────────────────────────────────────
# Defined as module-level functions — Airflow serialises tasks by import path,
# so lambdas and closures cannot be used with the default pickle serialiser.

def task_ingest_ohlcv(**context) -> None:
    """
    Download fresh OHLCV data from yfinance and persist to Parquet.
    Always force-refreshes on the scheduled daily run so the file
    reflects today's new trading day.
    """
    import sys
    sys.path.insert(0, str(_REPO_ROOT))

    from src.phase1.ingest.ohlcv import ingest_ohlcv
    config = _load_config()
    df = ingest_ohlcv(config, force_refresh=True)

    # Push shape to XCom so validate_and_engineer can log it
    context["ti"].xcom_push(key="ohlcv_shape", value=str(df.shape))


def task_ingest_macro(**context) -> None:
    """
    Fetch macro series from FRED and persist to Parquet.
    Uses the cached-with-fallback logic: if FRED is temporarily down for
    a series, the last known value is used rather than aborting the run.
    """
    import sys
    sys.path.insert(0, str(_REPO_ROOT))

    from src.phase1.ingest.macro import ingest_macro
    config = _load_config()
    df = ingest_macro(config, force_refresh=True)

    context["ti"].xcom_push(key="macro_shape", value=str(df.shape))


def task_validate_and_engineer(**context) -> None:
    """
    Run the full validation + feature engineering pipeline using the
    raw Parquet files just written by the two ingest tasks.

    force_refresh=False so the pipeline reads from disk (the freshly
    written files) rather than triggering another download.
    """
    import sys
    sys.path.insert(0, str(_REPO_ROOT))

    from src.phase1.store import run_feature_pipeline
    config = _load_config()

    # Retrieve upstream metadata from XCom for logging
    ti = context["ti"]
    ohlcv_shape = ti.xcom_pull(task_ids="ingest_ohlcv", key="ohlcv_shape")
    macro_shape  = ti.xcom_pull(task_ids="ingest_macro",  key="macro_shape")
    print(f"Upstream shapes — OHLCV: {ohlcv_shape}, Macro: {macro_shape}")

    feature_matrix = run_feature_pipeline(config, force_refresh=False)
    print(f"Feature matrix written: {feature_matrix.shape}")


# ── DAG definition ────────────────────────────────────────────────────────────

default_args = {
    "owner": "alphaflow",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": False,
    "email_on_failure": False,
    "email_on_retry": False,
}

with DAG(
    dag_id="phase1_data_pipeline",
    default_args=default_args,
    description=(
        "Daily ingestion of OHLCV + macro data, schema validation, "
        "leakage check, and feature engineering for 30 tickers."
    ),
    schedule_interval="0 6 * * 1-5",   # weekdays at 06:00 UTC
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,                  # prevent overlapping daily runs
    tags=["phase1", "data", "features"],
) as dag:

    ingest_ohlcv = PythonOperator(
        task_id="ingest_ohlcv",
        python_callable=task_ingest_ohlcv,
    )

    ingest_macro = PythonOperator(
        task_id="ingest_macro",
        python_callable=task_ingest_macro,
    )

    validate_and_engineer = PythonOperator(
        task_id="validate_and_engineer",
        python_callable=task_validate_and_engineer,
    )

    # ingest_ohlcv --+
    #                +--> validate_and_engineer
    # ingest_macro --+
    [ingest_ohlcv, ingest_macro] >> validate_and_engineer
