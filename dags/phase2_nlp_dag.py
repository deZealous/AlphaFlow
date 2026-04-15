"""
Airflow DAG -- Phase 2: weekly NLP feature enrichment.

Schedule : every Monday at 07:00 UTC
           (after weekend news and any Friday earnings releases are available)
Tasks    : ingest_news ----+
                           +--> run_nlp_pipeline
           ingest_edgar ---+
Retries  : 2 per task, 10-minute back-off
"""

from __future__ import annotations

import yaml
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

_REPO_ROOT   = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _REPO_ROOT / "configs" / "phase2_config.yaml"


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    cfg["raw_data_path"] = str(_REPO_ROOT / "data" / "raw")
    return cfg


def task_ingest_news(**context) -> None:
    import sys
    sys.path.insert(0, str(_REPO_ROOT))
    from src.phase2.ingest.news import ingest_news
    config = _load_config()
    df = ingest_news(config)
    context["ti"].xcom_push(key="news_articles", value=len(df))


def task_ingest_edgar(**context) -> None:
    import sys
    sys.path.insert(0, str(_REPO_ROOT))
    from src.phase2.ingest.edgar import ingest_edgar_filings
    config = _load_config()
    df = ingest_edgar_filings(config)
    context["ti"].xcom_push(key="edgar_filings", value=len(df))


def task_run_nlp_pipeline(**context) -> None:
    import sys
    sys.path.insert(0, str(_REPO_ROOT))
    from src.phase2.store import run_nlp_pipeline
    config = _load_config()
    ti = context["ti"]
    news_n  = ti.xcom_pull(task_ids="ingest_news",  key="news_articles")
    edgar_n = ti.xcom_pull(task_ids="ingest_edgar", key="edgar_filings")
    print(f"Upstream counts -- news: {news_n}, edgar: {edgar_n}")
    feature_matrix_v2 = run_nlp_pipeline(config)
    print(f"feature_matrix_v2 shape: {feature_matrix_v2.shape}")


default_args = {
    "owner":              "alphaflow",
    "depends_on_past":    False,
    "retries":            2,
    "retry_delay":        timedelta(minutes=10),
    "email_on_failure":   False,
    "email_on_retry":     False,
}

with DAG(
    dag_id="phase2_nlp_pipeline",
    default_args=default_args,
    description=(
        "Weekly NLP feature enrichment -- news sentiment (FinBERT), "
        "earnings call tone, forward-looking language, and NER features."
    ),
    schedule_interval="0 7 * * 1",   # every Monday at 07:00 UTC
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["phase2", "nlp", "finbert"],
) as dag:

    ingest_news_task = PythonOperator(
        task_id="ingest_news",
        python_callable=task_ingest_news,
    )

    ingest_edgar_task = PythonOperator(
        task_id="ingest_edgar",
        python_callable=task_ingest_edgar,
    )

    nlp_pipeline_task = PythonOperator(
        task_id="run_nlp_pipeline",
        python_callable=task_run_nlp_pipeline,
    )

    [ingest_news_task, ingest_edgar_task] >> nlp_pipeline_task
