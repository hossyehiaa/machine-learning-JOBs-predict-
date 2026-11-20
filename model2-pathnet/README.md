# PathNet0: Career Path Network & Trajectory Prediction System

**PathNet0** is an end-to-end machine-learning system and FastAPI serving layer (`app.py`) trained and evaluated on the full **1,140,000-record** longitudinal career transition dataset (`pathnet0_training_features.parquet`, spanning `801,603` unique `person_id`s across ESCO/ISCO, JobHop `JH_*`, and LinkedIn `LD_*` cohorts) and **230,333** deduplicated career skill-gap roadmaps (`pathnet0_rag_knowledge_base.parquet`).

---

## 1. Dataset Provenance & Packaged Artifacts

All packaged Parquet datasets and serialized model checkpoints in this repository are built from the **full 1,140,000-record dataset**:

- `pathnet0_training_features.parquet` — **`1,140,000` rows × `18` columns** (`801,603` unique individuals, `58,253` unique current roles, `103,207` unique target roles):
  - **ESCO/ISCO Occupational Transitions (`person_id < 1M`)**: `567,372` records
  - **JobHop Longitudinal Career Histories (`2B <= person_id < 3B`, `JH_TR*`)**: `284,628` records
  - **LinkedIn Career Promotions (`person_id >= 3B`, `LD_*`)**: `288,000` records
- `pathnet0_rag_knowledge_base.parquet` — **`230,333` rows × `6` columns** (deduplicated target role descriptions and phased skill-gap roadmaps).
- `pathnet0_deep_model.pt`, `pathnet0_hgb_regressor.joblib`, `pathnet0_hgb_classifier.joblib`, `pathnet0_isotonic_calibrator.joblib`, `pathnet0_feature_pipeline.joblib` — Trained strictly on the `GroupShuffleSplit` training partition (`n = 798,192` rows) and calibrated on the validation partition (`n = 170,996` rows), with `n = 170,812` records strictly held out for testing (`0` `person_id` overlap).

---

## 2. Held-Out Test Set Benchmarks (`n = 170,812` Test Records, `0` Person Overlap)

### A. Overall Pooled Test Set Performance (`n = 170,812`)

| Model Architecture | Probability MAE | Probability RMSE | Probability $R^2$ | Transition Type Acc | Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Deep Multi-Task PathNet (PyTorch)** | `0.0125` | `0.0252` | `0.9849` | `85.56%` | `0.7964` |
| **HistGradientBoosting (Trees)** | `0.0100` | `0.0203` | `0.9902` | `87.47%` | `0.8184` |
| **Unified Ensemble (Isotonic-Calibrated)** | **`0.0107`** | **`0.0207`** | **`0.9898`** | **`86.85%`** | **`0.8107`** |

### B. Within-Source Cohort Breakdown on Held-Out Test Set (`n = 170,812`)

Because the 1.14M dataset merges empirical branching frequencies from ESCO/JobHop (`mean = 0.011–0.019`) with LinkedIn promotion feasibility scores (`mean = 0.461`), pooled $R^2$ (`0.9898`) largely reflects cross-regime separation. Within-cohort test metrics isolate fine-grained accuracy within each data source:

| Data Cohort (Held-Out Test Partition) | Test Records (`n`) | Probability MAE | Probability RMSE | Transition Type Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **ESCO/ISCO Occupational Transitions (`<1M`)** | `84,588` | `0.0041` | `0.0114` | `84.86%` |
| **JobHop Longitudinal Histories (`2B..3B`)** | `42,524` | `0.0036` | `0.0135` | `77.29%` |
| **LinkedIn Career Promotions (`>=3B`)** | `43,700` | `0.0304` | `0.0354` | `100.00%` |

---

## 3. Repository Structure & Included Files

- [`PathNet0_Production_Ready.ipynb`](./PathNet0_Production_Ready.ipynb) — Complete, self-contained Jupyter Notebook:
  - **Section I (Cells 1–9)**: Original 1,140,000-row preprocessing pipeline and preserved cell outputs (`LD_*` and `JH_*` records, `230,333` deduplicated roadmaps).
  - **Section II (Cells 10–18)**: Self-contained module generation, `GroupShuffleSplit` evaluation on the 1.14M dataset (`170,812` held-out test records), post-hoc `IsotonicRegression` calibration curves, multi-dimensional subgroup fairness & error parity audit, and automated 11-check FastAPI & security test suite.
- [`train_pathnet0.py`](./train_pathnet0.py) — Full training, calibration, and cohort evaluation pipeline.
- [`pathnet0_inference.py`](./pathnet0_inference.py) — Standalone CLI and Python inference engine with automatic pickle version-mismatch recovery.
- [`calibration_and_fairness.py`](./calibration_and_fairness.py) — Held-out test set ECE and Brier Score evaluation + multi-dimensional subgroup error parity audit across **Occupational Sectors**, **Experience Tenure Brackets**, and **Seniority Tiers**.
- [`app.py`](./app.py) — Hardened FastAPI microservice exposing `GET /health`, `POST /v1/predict-transition`, `POST /v1/recommend-next-steps`, and `GET /metrics` (standard Prometheus `0.0.4` text exposition format), with explicit CORS allowlisting, optional `X-API-Key` authentication, per-IP sliding-window rate limiting, and sanitized `reference_id` error responses.
- [`test_pathnet0.py`](./test_pathnet0.py) — Automated 11-check verification suite validating endpoints, Prometheus exposition format, input validation (`422`), authentication (`401`), rate limiting (`429`), error sanitization (`500`), deterministic `==` dependency pinning, `python:3.12-slim` Docker configuration, and full-scale Parquet dimensions (`1,140,000 × 18` and `230,333 × 6`).
- [`Dockerfile`](./Dockerfile) & [`requirements.txt`](./requirements.txt) — Deterministically pinned dependencies (`scikit-learn==1.7.2`, `torch==2.13.0`, `httpx==0.28.1`, etc.) on `python:3.12-slim` running a single Uvicorn worker per replica as a non-root user.

---

## 4. Quickstart & Verification Commands

```bash
# 1. Train models, fit isotonic calibrator, and evaluate on 170,812 held-out test records
python3 train_pathnet0.py

# 2. Run held-out test calibration & multi-dimensional subgroup fairness audit (n=170,812)
python3 calibration_and_fairness.py

# 3. Run automated 11-check API, Prometheus metrics, auth, rate-limit, Docker & 1.14M data test suite
python3 test_pathnet0.py

# 4. Run CLI career transition inference
python3 pathnet0_inference.py --current "Data Analyst" --tenure 3.0 --target "Data Scientist"

# 5. Launch FastAPI microservice
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
```
