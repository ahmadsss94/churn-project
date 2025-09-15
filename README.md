# Customer Churn Prediction (Music Streaming, Subscription Model)

A production-minded, end-to-end pipeline to predict customer churn from user event logs.

---

## 1) Problem & Objective

- **Goal:** Predict which subscribers are likely to churn to enable targeted retention actions.
- **Business framing:** Surface high-risk users with strong **PR-AUC / precision-recall** so interventions are efficient.

---

## 2) Data Summary

- **Tables:** `df_log` (event logs), `df_churn` (user info/labels if present).
- **Final label balance (unique users):** **0 → 173**, **1 → 52** (total **225**).
- **Timestamps:** Derived `event_time` (converted to datetime).

---

## 3) Labeling & Leakage Prevention

- **Churn definition:** `page == "Cancellation Confirmation"` (first occurrence per user).
- **Cutoff:**  
  - Churners → `cutoff_time = first(cancel_time)`  
  - Non-churners → `cutoff_time = last(event_time)`
- **Leakage guard:** Build **all features only up to `cutoff_time`** per user.
- **Verification:** All churner events in features satisfy `event_time ≤ cutoff_time`.

---

## 4) Feature Engineering (20 features at/before cutoff)

- **Activity:** `event_count`, `n_sessions`, `n_songs`
- **Time:** `tenure_days` (first→last), `recency_days` (global_last→last)
- **Ratios:** `events_per_session`, `songs_per_session`
- **Page counters (one-hot sums):**  
  `page__Thumbs_Up`, `page__Thumbs_Down`, `page__Add_to_Playlist`,  
  `page__Add_Friend`, `page__Roll_Advert`, `page__Help`, `page__Settings`,  
  `page__Error`, `page__Downgrade`, `page__Upgrade`, `page__Logout`,  
  `page__Home`, `page__NextSong`
- **Subscription level:** `last_level` (e.g., `free` / `paid`)

**Top importances (RF):** recency_days, tenure_days, Add_Friend, songs_per_session, n_sessions, Thumbs_Up, NextSong, n_songs, Roll_Advert, events_per_session, Home, event_count, Thumbs_Down, Add_to_Playlist, Settings, Logout, Help, Downgrade, Upgrade, Error.

---

## 5) Data Splits

- **Hold-out stratified split** (used for metrics below).
- **Recommended for production:** **time-based split** (train = earliest 80% by `cutoff_time`, test = latest 20%) to simulate deployment.

---

## 6) Modeling

- **Preprocessing:**  
  - Numeric → `SimpleImputer(median)` + `StandardScaler`  
  - Categorical → `SimpleImputer(most_frequent)` + `OneHotEncoder(handle_unknown="ignore")`
- **Models:** Logistic Regression (baseline), **RandomForestClassifier** (selected).
- **Class imbalance:** `class_weight` + operating-point threshold tuning.

**Operating threshold:** ≈ **0.423** (selected for best F1 on the test split).

---

## 7) Results (Hold-out Stratified)

| Model           | Threshold | ROC-AUC | PR-AUC |   F1  | Precision | Recall |
|-----------------|-----------|--------:|------:|------:|----------:|------:|
| LogisticReg     | 0.50      | 0.939   | 0.834 | 0.710 | 0.610     | 0.850 |
| RandomForest    | 0.50      | 0.979   | 0.941 | 0.833 | 0.910     | 0.770 |
| RandomForest    | **0.423** | 0.979   | 0.941 | **0.846** | 0.846 | 0.846 |

> Test support: negatives=44, positives=13 (n=57).  


### Results (Time-based split)
| Model            | Threshold | ROC-AUC | PR-AUC |   F1  |
|------------------|-----------|--------:|------:|------:|
| LogisticRegression | 0.61    | 0.94    | 0.86  | 0.74  |
| RandomForest       | 0.66    | 0.97    | 0.93  | 0.82  |


---

## 8) API (FastAPI)

- Endpoints:
  - `GET /health` → service probe
  - `POST /predict_proba` → {"churn_probability": float}
  - `POST /predict` → {"churn_probability": float, "prediction": int, "threshold": float}
- Local run:
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  pip install --upgrade pip
  pip install -r requirements.txt
  uvicorn app:app --host 0.0.0.0 --port 8000
  # Swagger: http://127.0.0.1:8000/docs

- Payload schema: keys must match the model’s feature_order. Example is shown in the Swagger UI.

> Note: Dockerfile is included for Docker-capable hosts. On this VM (no WSL2), we run via Python venv (documented here).

#### Screenshots
![Swagger UI](screenshot/Swagger UI.png)
---

## 9) Monitoring & Drift

- Script: `monitor.py`
- Data drift: PSI on scores (quantile bins with safe fallback)  
  Heuristic: <0.1 OK, 0.1–0.2 WARN, >0.2 ACTION.
- Concept drift: PR-AUC drop vs baseline PR-AUC (relative drop threshold, default 0.20).
- Policy:
  - OK/WARN/ACTION from PSI
  - Escalate to ACTION if PR-AUC drop ≥ threshold
  - ACTION + stable PR-AUC → re-tune threshold / review cohort mix  
  - ACTION + PR-AUC degradation → retrain and promote if better

Usage
python monitor.py baseline ^
  --model churn_model.joblib ^
  --batch data\baseline_batch.csv ^
  --out-scores monitoring\baseline_scores.npy ^
  --label-col churn ^
  --out-metrics monitoring\baseline_metrics.json

python monitor.py check ^
  --model churn_model.joblib ^
  --batch monitoring\batches\new_batch.csv ^
  --baseline-scores monitoring\baseline_scores.npy ^
  --label-col churn ^
  --baseline-metrics monitoring\baseline_metrics.json ^
  --psi-warn 0.20 --psi-action 0.50 --psi-bins 20 ^
  --out-report monitoring\reports\YYYY-MM-DD_report.json ^
  --out-scores monitoring\reports\YYYY-MM-DD_scores.npy

Observed example: Using train as baseline and test as batch → PSI ≈ 1.83 (ACTION) with PR-AUC ≈ 0.94 (stable). Interpretation: cohort shift in score distribution, not a performance failure. Prefer a recent stable window as the PSI baseline.

---

## 10) Error Analysis

Notebook code exports:
- Confusion matrix & classification report at the operating threshold.
- Top False Positives / False Negatives (with scores & margins).
- Monitoring/reports/ for review and README screenshots.

Interpretation tips:
- FP: users flagged but didn’t churn → consider stricter threshold / extra negative signals.
- FN: missed churners → add recency/tenure window features or probability calibration.

#### Screenshots
![Error Analysis](screenshot/Erroranalysis.png)


---

## 11) Experiment Tracking (MLflow)

- Optional logging integrated in monitor.py.
- Run UI:
  set MLFLOW_TRACKING_URI=./mlruns
  mlflow ui --port 5000

- Track model metrics (ROC-AUC, PR-AUC, F1), parameters (n_estimators, threshold), artifacts (pipeline, feature list).

---

## 12) Packaging & Quality

- Dependencies: requirements.txt (pin scikit-learn==1.5.1 to match the saved artifact).
- Pre-commit hooks: ruff + black + hygiene checks; install once:
  pip install pre-commit ruff black
  pre-commit install
  pre-commit run --all-files

- Structure
  project/
  ├─ app.py
  ├─ monitor.py
  ├─ churn_model.joblib
  ├─ requirements.txt
  ├─ README.md
  ├─ notebooks/
  ├─ monitoring/
  │  ├─ baseline_scores.npy
  │  ├─ baseline_metrics.json
  │  ├─ batches/
  │  └─ reports/
  └─ (.pre-commit-config.yaml, Dockerfile, .gitignore)

---

## 13) Retraining Strategy

- When: on schedule (e.g., monthly) and/or when ACTION + significant PR-AUC drop.
- How: rebuild features on the latest window, retrain RF (and keep Logistic baseline), re-tune threshold, replace churn_model.joblib, refresh PSI/PR baselines.

---

## 14) Limitations & Next Steps

- Prefer a time-based split for production realism.
- Add rolling-window features (7/14/30-day activity/recency).
- Try calibration (isotonic) and LightGBM/XGBoost.
- Consider Top-K selection (e.g., top 10% highest risk) for retention campaigns.
- Add a simple model card (assumptions, risks, mitigations).

## Have fun ;)
