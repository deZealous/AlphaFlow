# AlphaFlow

An end-to-end production ML platform for algorithmic trading under market regime uncertainty.

AlphaFlow is not a trading bot. It is a production ML system that solves a specific and well-defined problem: **how do you make portfolio decisions when the statistical structure of the market keeps changing?** It does this by combining regime detection, meta-learning, multi-agent reinforcement learning, ensemble methods, uncertainty quantification, causal inference, and adversarial robustness testing — all deployed on a production MLOps stack.

---

## The problem

Most algorithmic trading systems are trained on one market regime and deployed into another. A momentum model trained on 2020-2021 bull conditions bleeds capital in a 2022 bear market. The system has no mechanism to know the world has changed, no way to adapt, and no fallback. This project solves three nested subproblems simultaneously:

1. **Regime awareness** — know what kind of market you are in
2. **Fast model adaptation** — update to a new regime without full retraining
3. **Multi-asset decision making under uncertainty** — allocate capital across asset classes when models disagree

---

## Architecture overview

```
Data layer
  OHLCV · FRED macro · FinBERT news sentiment
          |
Layer 1 — Regime detector
  HMM + causal graph structure (DoWhy)
          |
Layer 2 — Meta-learner (MAML)
  Fast-adapts 4 base models to current regime
  Triggered by uncertainty spike from Layer 3
          |
Layer 3 — Uncertainty quantification
  MC Dropout + Deep Ensembles on LSTM & Transformer
  Posterior distributions, not point estimates
          |
Layer 4 — Multi-agent system (MADDPG)
  One agent per asset class, shared critic
  Agents penalised for trading under high epistemic uncertainty
          |
Layer 5 — Stacked ensemble blender
  Ridge blender trained on inter-model disagreement
  Outputs calibrated confidence score per signal
          |
Layer 6 — Hierarchical RL execution agent
  High-level DQN: direction (long/short/flat)
  Low-level SAC: size, entry timing, stop-loss
  Learns to sit flat when uncertainty is high
          |
Adversarial stress testing
  Generator RL policy creates worst-case scenarios
  Red-team report logged per model version
          |
MLOps layer
  MLflow · Airflow · Kubernetes · GitHub Actions
  Evidently AI drift detection · Grafana monitoring
```

---

## Project phases

| Phase | Name | Status |
|-------|------|--------|
| 1 | Data pipeline + feature store | Complete |
| 2 | Alternative data — FinBERT NLP | Not started |
| 3 | Regime detector (HMM) | Not started |
| 4 | Causal inference layer | Not started |
| 5 | Base models + MLflow registry | Not started |
| 6 | Uncertainty quantification (Bayesian) | Not started |
| 7 | Meta-learner (MAML) | Not started |
| 8 | Multi-agent system (MADDPG) | Not started |
| 9 | Stacked ensemble blender | Not started |
| 10 | Hierarchical RL execution agent | Not started |
| 11 | Adversarial stress testing | Not started |
| 12 | Explainability dashboard | Not started |
| 13 | Production deployment | Not started |

---

## Phase 1 — Data pipeline + feature store

### What was built

| Module | Purpose |
|--------|---------|
| `src/phase1/ingest/ohlcv.py` | Downloads 30 tickers via yfinance, flattens MultiIndex to `{Field}_{TICKER}`, caches to Parquet |
| `src/phase1/ingest/macro.py` | Fetches 5 FRED series via fredapi, resamples to business-day frequency with ffill, cache-with-fallback on FRED outages |
| `src/phase1/validate/schema.py` | Schema validators for OHLCV, macro, and feature matrix — DatetimeIndex, null rates, row count, future-date check |
| `src/phase1/validate/leakage_check.py` | Correlation heuristic leakage detector — flags any feature where lag-0 correlation with target exceeds 1.5x lag-1 |
| `src/phase1/features/technical.py` | 17 features per ticker (RSI, MACD, Bollinger, ATR, rolling returns/vol, volume ratio), all shifted 1 day |
| `src/phase1/store.py` | Pipeline orchestrator — 7-step sequence with MLflow logging |
| `dags/phase1_ingest_dag.py` | Airflow DAG — weekdays 06:00 UTC, 2 retries, `[ingest_ohlcv, ingest_macro] >> validate_and_engineer` |

### Feature matrix (on disk)

```
data/features/feature_matrix.parquet
  Shape      : 1509 rows x 515 columns
  Date range : 2019-01-02 to 2024-12-30
  Tickers    : 30 (AAPL, MSFT, GOOGL, ... DIA)
  Features   : 510 technical (30 x 17) + 5 macro
  Null rate  : max 4.0% per column (indicator warmup rows only)
  Leakage    : PASS
```

### Tests

```
pytest tests/phase1/    # 60 tests, 60 pass
  test_ingest.py        — 24 tests (parse, cache, download, round-trip)
  test_validation.py    — 17 tests (schema, leakage detection)
  test_features.py      — 19 tests (feature shape, shift, naming, no cross-contamination)
```

### Running Phase 1

```bash
# Activate environment
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

# Install Phase 1 dependencies
pip install yfinance fredapi pandas numpy pyarrow ta pyyaml python-dotenv mlflow pytest

# Set FRED API key
cp .env.example .env
# edit .env: FRED_API_KEY=your_key_here

# Run the pipeline
python -m src.phase1.store

# Run tests
pytest tests/phase1/

# View MLflow run
mlflow ui --port 5000
```

---

## Tech stack

### ML and data
- `yfinance` — OHLCV price data
- `fredapi` — macroeconomic features from FRED
- `FinBERT` (HuggingFace) — earnings call and news sentiment
- `ta` — technical indicators (RSI, MACD, Bollinger, ATR)
- `hmmlearn` — Hidden Markov Model regime detection
- `DoWhy` / `CausalML` — causal inference and confound removal
- `PyTorch` — LSTM, Transformer, deep RL
- `learn2learn` / `higher` — MAML meta-learning
- `LightGBM`, `XGBoost` — tabular base models
- `Stable-Baselines3` — PPO, DQN, SAC reinforcement learning
- `PettingZoo` — multi-agent environment framework
- `scikit-learn` — stacking, calibration, Ridge blender
- `SHAP` — model explainability
- `Evidently AI` — data and model drift detection

### MLOps and infrastructure
- `MLflow` — experiment tracking and model registry
- `Apache Airflow` — pipeline orchestration and scheduled retraining
- `FastAPI` — REST inference API
- `Docker` — containerisation
- `Kubernetes` — deployment, scaling, HPA
- `GitHub Actions` — CI/CD, testing, deployment gates
- `Grafana` — live portfolio and model monitoring
- `Streamlit` — explainability dashboard

---

## Repository structure

```
alphaflow/
|-- configs/                  # YAML configs per phase
|-- dags/                     # Airflow DAGs
|-- data/
|   |-- raw/                  # Immutable raw data (gitignored)
|   |-- processed/            # Validated, cleaned data (gitignored)
|   |-- features/             # Feature store (Parquet, gitignored)
|-- models/                   # Serialised model artefacts (gitignored)
|-- notebooks/                # Exploratory analysis (not production)
|-- src/
|   |-- phase1/               # Data pipeline
|   |-- phase2/               # FinBERT NLP
|   |-- phase3/               # Regime detector
|   |-- phase4/               # Causal inference
|   |-- phase5/               # Base models
|   |-- phase6/               # Uncertainty quantification
|   |-- phase7/               # Meta-learner
|   |-- phase8/               # Multi-agent system
|   |-- phase9/               # Ensemble blender
|   |-- phase10/              # Hierarchical RL
|   |-- phase11/              # Adversarial testing
|   |-- phase12/              # Explainability dashboard
|   |-- serving/              # FastAPI app
|-- tests/                    # pytest test suite
|-- k8s/                      # Kubernetes manifests
|-- .github/workflows/        # GitHub Actions CI/CD
|-- Dockerfile
|-- requirements.txt
|-- setup.py
|-- CHECKLIST.md
|-- README.md
```

---

## Design principles

**No lookahead.** Every feature is lagged by at least one trading day. The leakage checker runs before any feature matrix is written to disk.

**Regime-conditioned everything.** No model runs without a regime tag. All training is stratified by regime. All evaluation is reported per regime.

**Uncertainty as a first-class signal.** Every prediction carries a posterior distribution. High epistemic uncertainty reduces position size automatically.

**Adversarial by default.** The stress testing layer is not optional. Every model version in the registry carries a red-team report before it can be promoted to champion.

**Explainable by design.** Every trade signal is traceable to the features that drove it, the model that generated it, the regime it was produced under, and the causal attribution of the macro factors involved.

---

## Disclaimer

This project is for research and portfolio demonstration purposes only. It does not constitute financial advice. No real capital is traded.
