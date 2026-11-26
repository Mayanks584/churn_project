# Netflix Customer Churn & Watch-Time Prediction — Complete AIML Syllabus Project

This repository contains **two independent Machine Learning modules** built on the same approved Netflix customer dataset (`50,000` unique customers, `29` columns), plus a local Version 1 churn-prediction web frontend:

- **Module 1 (Customer Churn Prediction — Classification & Clustering):** Predicts `churn_status` (`Active = 0`, `Cancelled = 1`) and segments customers via K-Means, Hierarchical Clustering, and DBSCAN.
- **Module 2 (Customer Watch-Time Prediction — Regression):** Independent CE-2 regression module predicting continuous `watch_time_minutes` (`10`–`180` minutes) using Simple Linear Regression, Multiple Linear Regression, and Polynomial Regression evaluated with `MAE`, `MSE`, `RMSE`, and `R²`.

---

## Project Files

| File | Purpose |
|---|---|
| [Netflix_Churn_and_Watch_Time_Regression_v3.ipynb](./Netflix_Churn_and_Watch_Time_Regression_v3.ipynb) | **Latest Combined Deliverable (Modules 1 + 2)** (`250` cells: `130` Markdown + `120` freshly executed Code cells, `0` errors). Module 1 is preserved intact; Module 2 adds Phases 1–10 with 9-point English + Hinglish explanations before every code cell. |
| [Netflix_Churn_and_Watch_Time_Regression_v3_Code_Only.ipynb](./Netflix_Churn_and_Watch_Time_Regression_v3_Code_Only.ipynb) | Code-only companion (`120` code cells in exact order from `v3.ipynb`, `0` Markdown cells, cleared outputs). |
| [Netflix_Churn_Syllabus_Implementation_v2.ipynb](./Netflix_Churn_Syllabus_Implementation_v2.ipynb) | Completed Module 1 explained notebook — **preserved unmodified**. |
| [Netflix_Churn_Syllabus_Implementation_v2_Code_Only.ipynb](./Netflix_Churn_Syllabus_Implementation_v2_Code_Only.ipynb) | Completed Module 1 code-only notebook — **preserved unmodified**. |
| [backups/Untitled_final_fully_explained_backup.ipynb](./backups/Untitled_final_fully_explained_backup.ipynb) | Verified SHA-256 backup copy of the original documented notebook — **preserved unmodified**. |
| [backups/README_before_module2_backup.md](./backups/README_before_module2_backup.md) | SHA-256 backup of `README.md` prior to Module 2 additions. |
| [data/original_data/netflix_large_dataset_cleaned.csv](./data/original_data/netflix_large_dataset_cleaned.csv) | Approved dataset (`50,000` unique customers, `29` columns) — **preserved unmodified**. |

---

## Module 1 Summary: Customer Churn Prediction (Classification & Clustering)

- **Target (`churn_status`)**: `Active = 41,391 (82.78%)` $\rightarrow 0$; `Cancelled = 8,609 (17.22%)` $\rightarrow 1$.
- **Train / Test Split**: Stratified 80/20 split (`random_state=42`): `40,000` training rows (`X_train`), `10,000` test rows (`X_test`).
- **Excluded Leakage**: `subscription_end_date` (`41,391` `Active` rows use date-only ISO `"2026-02-07"`, while `Cancelled` rows use `%Y-%m-%d %H:%M:%S` timestamps; default `pd.to_datetime(..., errors="coerce")` coerces `"2026-02-07"` to `NaT`, whereas `format="mixed"` parses both — either way, 100% target leakage).

### Module 1 Held-Out Test Comparison (`X_test`, $n = 10,000$)

| Model | Train Scope | Accuracy | Precision | Recall | F1 | ROC-AUC | Average Precision (AP) | PR-AUC (Trapezoidal) | Confusion Matrix (`TN, FP, FN, TP`) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| **Tuned Decision Tree (Balanced)** | Full `X_train` (40k) | 0.6303 | 0.2549 | 0.5964 | **0.3572** | 0.6383 | 0.2341 | 0.2498 | `5276, 3002, 695, 1027` |
| **Tuned Random Forest (Balanced)** | Full `X_train` (40k) | 0.6294 | 0.2543 | 0.5964 | 0.3566 | **0.6384** | **0.2442** | 0.2437 | `5267, 3011, 695, 1027` |
| **Tuned LinearSVC (Balanced)** | Full `X_train` (40k) | 0.6226 | 0.2519 | 0.6051 | 0.3558 | 0.6340 | 0.2372 | 0.2367 | `5184, 3094, 680, 1042` |
| **Balanced Logistic Regression** *(Historical)* | Full `X_train` (40k) | 0.6198 | 0.2506 | **0.6069** | 0.3547 | 0.6340 | 0.2373 | 0.2369 | `5153, 3125, 677, 1045` |
| **Gaussian Naive Bayes** | Full `X_train` (40k) | 0.6347 | 0.2529 | 0.5738 | 0.3510 | 0.6375 | 0.2375 | 0.2372 | `5359, 2919, 734, 988` |
| **SVC (RBF Kernel, Balanced)** | Stratified 6k `X_train` | 0.6203 | 0.2391 | 0.5523 | 0.3337 | 0.6213 | 0.2363 | 0.2358 | `5252, 3026, 771, 951` |
| **KNN (`K=5`)** *(Historical)* | Full `X_train` (40k) | 0.8006 | 0.2405 | 0.0732 | 0.1122 | 0.5594 | 0.1992 | 0.2148 | `7880, 398, 1596, 126` |
| **Logistic Regression** *(Historical)* | Full `X_train` (40k) | 0.8278 | 0.0000 | 0.0000 | 0.0000 | 0.6345 | 0.2379 | 0.2374 | `8278, 0, 1722, 0` |

---

## Module 2 Summary: Customer Watch-Time Prediction (Regression)

- **Target (`y_reg = watch_time_minutes`)**: Continuous range `10`–`180` minutes (`162` unique values, `mean = 85.00`, `std = 28.09`, `median = 83.00`, `skew = 0.1710`).
- **Train / Test Split**: Independent 80:20 split (`random_state=42`): `X_train_reg` (`40,000`), `X_test_reg` (`10,000`, protected until Phase 10 final evaluation).
- **Engineered Event-Time Feature**: `tenure_at_watch = (date_watched - subscription_start_date).dt.days` (`0`–`300` days, replacing Module 1's global-max `tenure_days`).

### Module 2 Leakage & Temporal-Validity Audit
- **Excluded due to Target Leakage**:
  - `session_count`: `max(1, watch_time_minutes // 30)` matches **50,000 / 50,000 rows (`100%`)**.
  - `watch_time_per_session`: directly computed from `watch_time_minutes / session_count`.
  - `completion_percentage`: matches `min(100, floor(watch_time_minutes / plan_denom * 100))` on **99.98%** of rows (`Basic=60`, `Standard=90`, `Premium=120`).
- **Excluded due to Unverified Pre-Event Window / Post-Watch Timing**:
  - `days_since_last_watch`: exceeds `tenure_at_watch` on **5,036 / 50,000 rows (`10.07%`)**.
  - `content_diversity_score`: spans `0.21`–`0.99` even on all `162` Day-0 rows (`tenure_at_watch == 0`), with no pre-event window timestamp.
  - `avg_weekly_watch_time`: no observation-window timestamp proving pre-session availability (near-zero within-plan correlation with target).
  - `rating` & `liked`: post-watch feedback (`liked` deterministic `rating >= 4`).
  - `tenure_days` (Module 1 global-max `2025-04-14`), `churn_status`, `subscription_end_date`, `user_id`, `title`, raw dates, `release_year`.
- **Safe Candidate Pool (`15` Columns)**: `monthly_fee`, `tenure_at_watch`, plus `13` pre-event/context categoricals (`subscription_plan`, `country`, `region`, `age_group`, `gender`, `payment_method`, `discount_applied`, `content_type`, `genre`, `language`, `device_type`, `time_of_day`, `recommendation_source`).
- **Collinearity Resolution (Phase 5)**: Full `drop="first"` encoded design matrix (`Spec A`) exhibited `VIF = inf` due to cross-group exact dependencies (`monthly_fee` $\leftrightarrow$ `subscription_plan` and `country` $\leftrightarrow$ `region`). Removing redundant `monthly_fee` and `region` (`Spec B`, `45` encoded features) reduced Max VIF to **`1.8438`**.

### Module 2 Cross-Validation (`X_train_reg`) & Final Held-Out Test Results (`X_test_reg`, $n = 10,000$)

#### 5-Fold Cross-Validation (`X_train_reg`, Mean ± Std)
| Model | CV MAE | CV MSE | CV RMSE | CV R² |
|---|---:|---:|---:|---:|
| **Simple Linear Regression (`monthly_fee`)** | 12.0412 ± 0.1014 | 227.7041 | 15.0892 ± 0.1370 | 0.7104 ± 0.0030 |
| **Multiple Linear Regression (`Spec B`: Plan + Country + Context)** | 11.9818 ± 0.0901 | 225.4547 | 15.0146 ± 0.1327 | 0.7133 ± 0.0029 |
| **Polynomial Regression (`Degree 2` on `monthly_fee`)** | 11.9750 ± 0.0924 | 225.1307 | 15.0038 ± 0.1346 | 0.7137 ± 0.0029 |
| *Polynomial Regression (`Degree 2` on continuous `tenure_at_watch`)* | 23.3989 ± 0.1218 | 786.6202 | 28.0463 ± 0.1547 | -0.0003 ± 0.0005 |

#### Final Held-Out Test Comparison (`X_test_reg`, $n = 10,000$)
| Model | MAE (min) | MSE ($\text{min}^2$) | RMSE (min) | R² |
|---|---:|---:|---:|---:|
| **Simple Linear Regression (`monthly_fee`)** | 12.0739 | 230.1009 | 15.1691 | 0.7117 |
| **Multiple Linear Regression (`Spec B`: Non-Redundant)** | 12.0110 | 227.7942 | 15.0929 | 0.7146 |
| **Polynomial Regression (`Degree 2` on `monthly_fee`; Deg 3 rank-redundant)** | **11.9986** | **227.3737** | **15.0789** | **0.7151** |

- **Key Regression Takeaways**:
  - `monthly_fee` / `subscription_plan` drives virtually all predictable variance (`R² ≈ 0.71`): `Basic` averages `~59.6 min`, `Standard` `~89.4 min`, and `Premium` `~119.4 min`.
  - **Degree 3** on `monthly_fee` is **rank-redundant** (`matrix rank = 3 < 4` columns because `monthly_fee` has only 3 unique values), while **Degree 2** captures the slight non-constant step between plans (`R²: 0.7117 -> 0.7151`).
  - Residual diagnostics show symmetric, zero-mean, homoscedastic errors across all plans (`std ≈ 15.0 min`), confirming remaining variance stems from within-plan behavioral noise not captured by static pre-event attributes.

---

## CE-2 Syllabus Coverage Summary

- **Module 1 (Classification & Unsupervised Learning)**: Data audit, NumPy, Logistic Regression, Balanced LR, KNN, Decision Tree, Random Forest, LinearSVC, sampled RBF SVC, Gaussian Naive Bayes, 5-fold Stratified CV, `GridSearchCV`/`RandomizedSearchCV`, Bias-Variance curves, `SelectKBest`, `RFE`, MDI vs Permutation Importance, PCA comparison, K-Means (Full vs Reduced), Agglomerative Ward, DBSCAN, Subgroup Fairness Audit.
- **Module 2 (Regression)**: Target & temporal leakage audit (`tenure_at_watch`), Simple Linear Regression ($y = mx + c$), Multiple Linear Regression ($y = b_0 + \sum b_i x_i$) with full design-matrix VIF (`drop="first"`), Polynomial Regression (`PolynomialFeatures` Degrees 1–3 + discrete rank analysis + continuous `tenure_at_watch` check), `MAE`, `MSE`, `RMSE`, `R²`, 5-fold CV, residual & overfitting diagnostics.
- **Not Suitable / Future Scope**: Multi-table Data Integration, live GenAI narrative assistant, General/Super AI, Reinforcement Learning, Apriori/FP-Growth, Recommendation Systems.

---

## Environment, Notebook Reproduction & V1 Web Frontend

Verified environment (`/home/ishu/ml/ml_env`): Python `3.12.3`, pandas `3.0.3`, numpy `2.5.1`, scipy `1.18.0`, scikit-learn `1.9.0`, matplotlib `3.11.0`, seaborn `0.13.2`, nbconvert `7.17.1`, nbclient `0.11.0`.

To execute the combined Modules 1 + 2 notebook into a separate output file:
```bash
/home/ishu/ml/ml_env/bin/jupyter nbconvert --to notebook --execute \
  Netflix_Churn_and_Watch_Time_Regression_v3.ipynb \
  --output Netflix_Churn_and_Watch_Time_Regression_v3_executed.ipynb
```

### Version 1 Churn Frontend (Local Demo)
- [export_model_from_notebook.py](./export_model_from_notebook.py), [validator.py](./validator.py), [app.py](./app.py), [frontend/](./frontend/), [tests/test_frontend_backend.py](./tests/test_frontend_backend.py).
```bash
/home/ishu/ml/ml_env/bin/python -m unittest discover -s tests -v
/home/ishu/ml/ml_env/bin/python app.py --port 8000
```
