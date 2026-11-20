# FitNet (Model 1): Resume-to-Job & Career Matching System

**FitNet (Model 1)** is an end-to-end machine-learning system and FastAPI serving layer (`app_fitnet.py`) trained and evaluated on the full **221,903-record** resume-to-job matching dataset (`fitnet_training_features.parquet`, spanning `37,044` unique candidate resumes and `84,508` unique target job postings).

- **FitNet (Model 1 — Job & Career Matching)**: Evaluates how well a candidate's current resume, role, and skill set match a target job posting (`Low`, `Medium`, `High` match tier + calibrated fit percentage + matched skills vs. missing skill gap).
- **PathNet0 (Model 2 — Future-Self Career Modeling)**: Evaluates longitudinal career trajectory progression (`Vertical_Promotion`, `Adjacent_Pivot`, `Career_Shift` + transition feasibility probability + RAG roadmap retrieval across `1,140,000` records).

---

## 1. Key Engineering Improvement Over Original `FitNet_Preprocessing_Pipeline.ipynb`

In Cell 8 of the original preprocessing notebook (`FitNet_Preprocessing_Pipeline.ipynb`), applying `RARE_THRESHOLD = 50` to `target_future_role` (`job_title`, which has `84,508` unique titles across `221,903` rows) collapsed over `85%` of job titles into `"other"`.

In [`fitnet_training_features.parquet`](./fitnet_training_features.parquet) (`221,903` rows × `26` columns):
1. All **`84,508` unique `target_future_role` (`job_title`) strings** and `resume_id`s are restored from `final_model_training_dataset.csv`.
2. Each unique candidate resume is assigned a `candidate_id` (`37,044` unique candidates) so that `GroupShuffleSplit` enforces **zero candidate resume leakage** across Train (`155,087`), Validation (`32,794`), and Held-Out Test (`34,022`) partitions.

---

## 2. Held-Out Test Set Benchmarks (`n = 34,022` Unseen Candidate Resume-Job Pairs, `0` Candidate Overlap)

| Model Architecture | Fit Score MAE | Fit Score RMSE | Fit Score $R^2$ | Match Tier Accuracy (`Low`/`Med`/`High`) | Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Deep Multi-Task FitNet (PyTorch)** | `0.0826` | `0.1337` | `0.7392` | `82.73%` | `0.8263` |
| **HistGradientBoosting (Trees)** | `0.1131` | `0.1549` | `0.6499` | `80.36%` | `0.8022` |
| **Unified FitNet Ensemble (Calibrated)** | **`0.0899`** | **`0.1373`** | **`0.7250`** | **`82.73%`** | **`0.8259`** |

---

## 3. Quickstart & Verification Commands

```bash
# 1. Train FitNet models, fit isotonic calibrator, and evaluate on 34,022 held-out test records
python3 train_fitnet.py

# 2. Run held-out test calibration & multi-dimensional subgroup fairness audit (n=34,022)
python3 calibration_and_fairness_fitnet.py

# 3. Run automated API, Prometheus metrics, auth, rate-limit, and 221,903-row data test suite
python3 test_fitnet.py

# 4. Launch FitNet FastAPI microservice
uvicorn app_fitnet:app --host 0.0.0.0 --port 8000 --workers 1
```
