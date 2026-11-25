# Netflix Customer Churn Prediction — Complete AI & ML Syllabus Project

This project extends the original Netflix Customer Churn binary classification workflow (`Active = 0`, `Cancelled = 1`) into a complete, college-syllabus-aligned Machine Learning notebook with beginner-friendly English + Hinglish explanations before every code cell.

---

## Project Files

| File | Purpose |
|---|---|
| [Netflix_Churn_Syllabus_Implementation_v2.ipynb](./Netflix_Churn_Syllabus_Implementation_v2.ipynb) | **Latest Verified Deliverable** (`233` cells: `121` Markdown + `112` freshly executed Code cells, `0` errors, `0` SVC deprecation warnings). Includes environment verification, cleaned Part A explanations, corrected Phase 3 checks, `X_train`-only Phase 11 clustering, and Phases 3–16. |
| [Netflix_Churn_Syllabus_Implementation.ipynb](./Netflix_Churn_Syllabus_Implementation.ipynb) | Initial v1 extended notebook preserved prior to v2 revisions. |
| [Untitled_final_fully_explained.ipynb](./Untitled_final_fully_explained.ipynb) | Original documented notebook — **preserved unmodified**. |
| [backups/Untitled_final_fully_explained_backup.ipynb](./backups/Untitled_final_fully_explained_backup.ipynb) | Verified SHA-256 backup copy of the original notebook. |
| [data/original_data/netflix_large_dataset_cleaned.csv](./data/original_data/netflix_large_dataset_cleaned.csv) | Approved dataset (`50,000` unique customers, `29` columns) — **preserved unmodified**. |

---

## Dataset & Feature Overview

- **Rows / Customers**: `50,000` unique `user_id`s (`0` duplicate rows, `0` original missing values, `0` whitespace/casing errors).
- **Target (`churn_status`)**: `Active = 41,391 (82.78%)` $\rightarrow 0$; `Cancelled = 8,609 (17.22%)` $\rightarrow 1$.
- **Train / Test Split**: Stratified 80/20 split (`random_state=42`): `40,000` training rows (`X_train`), `10,000` test rows (`X_test`).
- **Approved Features (`24`)**:
  - **Numeric (`10`)**: `monthly_fee`, `watch_time_minutes`, `session_count`, `completion_percentage`, `rating`, `days_since_last_watch`, `avg_weekly_watch_time`, `content_diversity_score`, plus engineered `tenure_days` and `watch_time_per_session`.
  - **Categorical (`14`)**: `age_group`, `gender`, `country`, `region`, `subscription_plan`, `payment_method`, `discount_applied`, `content_type`, `genre`, `language`, `device_type`, `time_of_day`, `liked`, `recommendation_source`.
- **Excluded Columns**: `user_id` (identifier), `title` (20,987 unique values), `subscription_end_date`, `churn_status` (target), `release_year`, and raw dates (`date_watched`, `subscription_start_date`).
  - **Why `subscription_end_date` is excluded (100% Target Leakage)**: `Cancelled` rows store full timestamps (`%Y-%m-%d %H:%M:%S`), whereas all `41,391` `Active` rows store the valid date-only ISO placeholder `"2026-02-07"` (`%Y-%m-%d`). Under **default date-format inference with `errors="coerce"`**, `pd.to_datetime` infers `%Y-%m-%d %H:%M:%S` and coerces `"2026-02-07"` to `NaT`; under **explicit `format="mixed"` parsing**, both formats parse into valid timestamps. In both cases, all `41,391` placeholder rows align 100% with `Active` customers, confirming severe target leakage.
- **Data Audit Note (`liked` vs `rating`)**: Cross-tabulation shows a deterministic mapping (`rating` 1–3 $\rightarrow$ `liked="No"`, `rating` 4–5 $\rightarrow$ `liked="Yes"`), indicating redundant information between these two columns while keeping original data unmodified.

---

## Methodology Guardrails

1. **Leakage Prevention**: All `ColumnTransformer` preprocessing (`SimpleImputer`, `StandardScaler`, `OneHotEncoder`), `SelectKBest`, `RFE`, `PCA`, and Phase 11 `StandardScaler`/K-Means selection are fitted strictly on `X_train` (and inside CV folds during `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`).
2. **`X_test` Protection**: `X_test` was used for the preserved historical baseline models and was protected from all new model selection, feature selection, clustering $K$ selection, and hyperparameter tuning decisions. Selected final models were evaluated once on `X_test` at the Final Evaluation stage.
3. **Separate PR Metrics**: **Average Precision (AP)** (`average_precision_score`) and **Trapezoidal PR-AUC** (`auc(recall, precision)` via `precision_recall_curve`) are computed and reported separately.
4. **No Artificial Inflation**: No rows were deleted for statistical outliers, no synthetic churn labels were created, and SMOTE was not applied.

---

## Model Comparison Results (Held-Out `X_test`, $n = 10,000$)

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

### Key Analytical Findings
- **5-Fold Stratified CV (`X_train`) & Hyperparameter Tuning**: Tuned Random Forest (`n_estimators=100, max_depth=6, min_samples_split=10, min_samples_leaf=5, class_weight='balanced'`) achieved the highest CV `F1` (`0.3520`), while Tuned Decision Tree (`max_depth=4, min_samples_leaf=5`) achieved `CV F1 = 0.3502`.
- **Bias-Variance Trade-Off**: Unconstrained Decision Tree (`max_depth=None`) overfit with `Train F1 = 0.4884` vs `Validation F1 = 0.2796` (`gap = +0.2088`), whereas depth-capped Tuned Random Forest (`max_depth=6`) kept the Train–Validation F1 gap at `+0.0039`.
- **Feature Selection & Importance**: `SelectKBest(f_classif, k=10)` achieved `CV F1 = 0.3519` (outperforming all 73 features at `0.3493`). Validation Permutation Importance confirmed `subscription_plan` (`+0.04064` F1 drop) and `monthly_fee` (`+0.00549`) as top predictors while exposing MDI inflation on collinear `session_count`.
- **PCA Comparison**: Retaining 95% variance (`73 -> ~44` components) slightly improved KNN CV `F1` (`0.1010 -> 0.1070`) and runtime, slightly hurt Random Forest CV `F1` (`0.3520 -> 0.3492`) and more than doubled RF CV runtime (`1.76s -> 4.22s`), and left linear models nearly unchanged.
- **Customer Segmentation (`X_train` Only, Phase 11)**: Removing redundant watch-volume features (`watch_time_minutes`, `session_count`, $r = 0.952$) shifted `X_train` K-Means ($K=2$) from `Silhouette = 0.2468` (Full, 10 features: `22.20%` vs `10.58%` post-hoc `y_train` churn) to `0.2000` (Reduced, 8 features: `24.47%` vs `11.72%` post-hoc `y_train` churn). Agglomerative Ward (`n=2,000` `X_train` sample) scored `0.1986` at $K=2$, and DBSCAN (`eps=1.6, min_samples=25`, `n=5,000` `X_train` sample) found 2 clusters with `3.74%` noise (`0.2273` noise-excluded Silhouette).
- **Ethical AI & Fairness (Phase 12)**: Every demographic and geographic subgroup (`gender`, `age_group`, `region`, `country`) met sufficiency thresholds (`N >= 30`, `Active >= 20`, `Cancelled >= 20`), with modest regional Recall variations (`0.5294` Canada to `0.6543` UK) attributable to sample variance and plan/engagement mix rather than proven causal bias.

---

## Syllabus Coverage Summary

- **Already Implemented (Part A)**: Data loading & inspection, target imbalance & `subscription_end_date` leakage detection, OHE / MinMax / StandardScaler / SimpleImputer / ColumnTransformer, feature engineering (`tenure_days`, `watch_time_per_session`) & ablation, Baseline & Balanced Logistic Regression, KNN (`K=5`), PCA demo.
- **Newly Implemented (Part B)**: Environment verification cell, Data Quality Re-Audit (`X_train`-only target checks), practical NumPy implementation, Decision Tree, Random Forest, `LinearSVC`, sampled `RBF SVC` (`decision_function` without deprecated `probability` warning), Gaussian Naive Bayes, 5-fold Stratified CV pipelines, multi-metric `GridSearchCV` / `RandomizedSearchCV(n_iter=12)`, Validation & Learning Curves, `SelectKBest`, `RFE`, RF MDI vs Permutation Importance, controlled PCA Before-vs-After CV experiment, `X_train`-only K-Means (Full vs Reduced) + Hierarchical + DBSCAN clustering, Ethical AI & Fairness audit.
- **Not Suitable / Theoretical / Future Scope**: Conditional multi-table Data Integration (only 1 CSV supplied), Generative AI retention assistant (documented grounded template), General/Super AI & Reactive Machines, Linear/Polynomial Regression, Reinforcement Learning, Apriori/FP-Growth, and Recommendation Systems.

---

## Environment & Reproduction

Verified execution environment (`/home/ishu/ml/ml_env`):
- Python `3.12.3`, pandas `3.0.3`, numpy `2.5.1`, scipy `1.18.0`, scikit-learn `1.9.0`, matplotlib `3.11.0`, seaborn `0.13.2`, nbconvert `7.17.1`, nbclient `0.11.0`.

To execute the notebook into a separate output file without overwriting the saved deliverable:
```bash
/home/ishu/ml/ml_env/bin/jupyter nbconvert --to notebook --execute \
  Netflix_Churn_Syllabus_Implementation_v2.ipynb \
  --output Netflix_Churn_Syllabus_Implementation_v2_executed.ipynb
```
