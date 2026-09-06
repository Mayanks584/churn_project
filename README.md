# Netflix Customer Churn Prediction

## Project Overview

This project builds a machine-learning system to predict whether a Netflix customer is likely to churn.

### Problem Type

**Supervised Binary Classification**

- `0` → Active
- `1` → Cancelled

### Dataset

**File:** `netflix_large_dataset_cleaned.csv`

**Dataset size:** 50,000 rows × 29 columns

Each row represents one customer because `user_id` is unique for every row.

---

# 1. Overall Data Preprocessing Flow

```text
netflix_large_dataset_cleaned.csv
        ↓
Data quality checks
        ↓
Convert dates
        ↓
Clean / validate data
        ↓
Check repeated users
        ↓
Check churn target
        ↓
Check target leakage
        ↓
Select useful features
        ↓
Create customer-level features
        ↓
Remove IDs / leakage-prone raw fields
        ↓
Prepare final ML dataset
```

### Important note about aggregation

A common customer-churn workflow may aggregate event/session data into one row per customer.

**For this dataset, aggregation is NOT required**, because our analysis confirmed:

- 50,000 rows
- 50,000 unique users
- 0 repeated `user_id` values

Therefore, the existing dataset is already one row per customer.

---

# 2. Complete Project Roadmap

```text
1. Dataset Identification
        ↓
2. Input & Output Definition
        ↓
3. Data Understanding / EDA
        ↓
4. Data Preprocessing
        ↓
5. Feature Extraction / Feature Engineering
        ↓
6. Train-Test Preparation
        ↓
7. Binary Classification Models
        ↓
8. Multiple Experiments
   (Permutations & Combinations)
        ↓
9. Model Comparison
        ↓
10. Confusion Matrix + Other Metrics
        ↓
11. Best Model Selection
        ↓
12. Final Prediction System
        ↓
13. Deployment
```

---

# 3. Real Work Pipeline

```text
RAW NETFLIX CSV
       ↓
1. UNDERSTAND DATA
       ↓
2. CHECK DATA QUALITY
       ↓
3. CHECK TARGET
       ↓
4. CHECK LEAKAGE
       ↓
5. SELECT FEATURES
       ↓
6. CONVERT DATATYPES
       ↓
7. FEATURE ENGINEERING
       ↓
8. FINAL CLEAN DATASET
       ↓
9. TRAIN/TEST SPLIT
       ↓
10. ENCODING + SCALING
       ↓
11. MODEL TRAINING
       ↓
12. MODEL COMPARISON
       ↓
13. EVALUATION
       ↓
14. INTERPRETATION
       ↓
15. DEPLOYMENT
```

---

# 4. Current Project Status

## Stage 1 — Dataset Understanding ✅ COMPLETE

### Basic Dataset Understanding ✅

```text
50,000 customers
        ↓
29 columns
        ↓
No missing values
        ↓
No duplicate rows
        ↓
No repeated users
        ↓
Existing target: churn_status
        ↓
Categorical + numerical + date-like columns
        ↓
Potential target leakage: subscription_end_date
        ↓
High-cardinality feature: title
```

### Completed checks

```text
Dataset Identification           ✅
Basic EDA                       ✅
Target Analysis                 ✅
Duplicate Analysis              ✅
User Analysis                   ✅
Categorical Analysis            ✅
Numerical Range Inspection      ✅
```

---

# 5. Categorical Features Identified

```text
gender → 2 categories
country → 10 countries
region → 5 regions
subscription_plan → 3 plans
payment_method → 3 methods
discount_applied → Yes/No
content_type → Movie/Series
genre → 8 categories
language → 7 categories
device_type → 4 categories
time_of_day → 4 categories
liked → Yes/No
recommendation_source → 5 categories
```

### High-cardinality observation

`title` has **20,987 unique values**.

Because it is highly granular, it was excluded from the initial ML feature set instead of directly one-hot encoding thousands of titles.

---

# 6. Numerical Range Inspection

```text
age_group                 17 → 60
monthly_fee              8.99 → 17.99
release_year             2000 → 2025
watch_time_minutes         10 → 180
session_count                1 → 6
completion_percentage      20 → 100
rating                       1 → 5
days_since_last_watch       0 → 60
avg_weekly_watch_time      33 → 499
content_diversity_score    0.2 → 1.0
```

### Numerical quality

No invalid business-range values were found in the validated numerical checks.

### Outlier analysis

IQR analysis detected some statistical outliers, mainly in:

- `completion_percentage`
- `rating`
- `session_count`
- `watch_time_minutes`

These values were **not automatically removed**, because the flagged observations remained within valid business ranges.

Therefore:

> Statistical outlier ≠ invalid value.

---

# 7. Stage 2 — Date Conversion & Validation ✅ COMPLETE

### Date columns identified

```text
subscription_start_date
subscription_end_date
date_watched
```

### Date conversion

The date columns were converted from strings to proper datetime values.

A mixed-format issue was found in `subscription_end_date`, so mixed-format parsing was used.

### Date range checks

```text
subscription_start_date
2022-01-01 → 2024-06-19

date_watched
2022-01-05 → 2025-04-14
```

### Date logic validation

```text
Watch date before subscription start → 0
End date before subscription start   → 0
```

---

# 8. Target Leakage Investigation ✅ COMPLETE

`subscription_end_date` was investigated because it contains information strongly related to whether a customer cancelled.

The analysis showed a direct relationship between `subscription_end_date` and `churn_status`.

Therefore:

```text
subscription_end_date
        ↓
Potential target leakage
        ↓
EXCLUDE FROM MODEL
```

### Important rule

A feature must be available at prediction time.

We must not give the model information that effectively reveals the answer we are trying to predict.

---

# 9. Churn Target

### Original target distribution

```text
Active      = 41,391 → 82.78%
Cancelled   =  8,609 → 17.22%
```

The target is therefore **imbalanced**.

### Target encoding

```text
Active      → 0
Cancelled   → 1
```

### How imbalance was handled

We did not modify the original dataset distribution.

We used:

```text
Stratified train/test split
        +
class_weight="balanced"
```

This preserved the original class proportions during splitting and gave the minority churn class more importance during balanced Logistic Regression training.

---

# 10. Feature Engineering ✅ COMPLETE

## Engineered Feature 1 — `tenure_days`

Derived from:

```text
subscription_start_date
        ↓
tenure_days
```

Reference date used:

```text
2025-04-14
```

This was selected from the latest `date_watched` value in the dataset rather than using the current real-world date.

Observed range:

```text
299 → 1,199 days
```

## Engineered Feature 2 — `watch_time_per_session`

Formula:

```text
watch_time_per_session =
watch_time_minutes / session_count
```

Observed range:

```text
10 → 59 minutes
```

These features were retained as candidate engineered features and tested experimentally.

---

# 11. Feature Selection ✅ COMPLETE

## Excluded from the ML feature set

```text
user_id
```

Reason:

- identifier only
- retained in the original dataset for traceability
- not used as a predictive input

```text
title
```

Reason:

- 20,987 unique values
- very high-cardinality
- not used directly in the first model setup

```text
subscription_end_date
```

Reason:

- target leakage

```text
churn_status
```

Reason:

- target variable, not an input feature

```text
release_year
```

Reason:

- not selected for the first ML feature set

```text
date_watched
```

Reason:

- raw date is not used directly in the first model feature set
- relevant recency/usage information is already represented by other features such as `days_since_last_watch`

---

# 12. Final Feature Sets

### Original numerical features used

```text
monthly_fee
watch_time_minutes
session_count
completion_percentage
rating
days_since_last_watch
avg_weekly_watch_time
content_diversity_score
```

### Engineered numerical features

```text
tenure_days
watch_time_per_session
```

### Categorical features

```text
age_group
gender
country
region
subscription_plan
payment_method
discount_applied
content_type
genre
language
device_type
time_of_day
liked
recommendation_source
```

Final feature count before encoding:

```text
10 numerical + 14 categorical = 24 features
```

---

# 13. Train/Test Split ✅ COMPLETE

```text
Training set → 40,000 rows
Testing set  → 10,000 rows
```

Used:

```text
test_size = 0.20
stratify = y
random_state = 42
```

Target proportions remained approximately:

```text
Training:
Active      82.78%
Cancelled   17.22%

Testing:
Active      82.78%
Cancelled   17.22%
```

---

# 14. Encoding & Scaling ✅ COMPLETE

## Categorical Encoding

Used:

```text
OneHotEncoder
handle_unknown="ignore"
```

Original categorical features:

```text
14
```

Encoded categorical features:

```text
63
```

## Numerical Scaling

Used:

```text
StandardScaler
```

The final preprocessing output was:

```text
X_train_processed → 40,000 × 73
X_test_processed  → 10,000 × 73
```

The preprocessing pipeline was fitted only on training data and then applied to the test data.

---

# 15. Evaluation 1 — Experiments ✅ COMPLETE

For the first evaluation, three model setups were selected.

## Experiment 1 — Logistic Regression

Purpose:

```text
Baseline binary-classification model
```

Observed behavior:

```text
Accuracy  = 82.78%
Precision = 0%
Recall    = 0%
F1 Score  = 0%
```

The model predicted all test customers as `Active`.

This demonstrated the effect of class imbalance on a basic baseline model.

---

## Experiment 2 — Balanced Logistic Regression

Configuration:

```text
class_weight = "balanced"
```

Observed results:

```text
Accuracy  = 61.98%
Precision = 25.06%
Recall    = 60.69%
F1 Score  = 35.47%
```

This model detected substantially more churners than the unweighted baseline.

Confusion matrix:

```text
[[5153, 3125],
 [ 677, 1045]]
```

Interpretation:

```text
TN = 5153
FP = 3125
FN =  677
TP = 1045
```

---

## Experiment 3 — KNN

Configuration:

```text
n_neighbors = 5
```

Observed results:

```text
Accuracy  = 80.06%
Precision = 24.05%
Recall    = 7.32%
F1 Score  = 11.22%
```

Confusion matrix:

```text
[[7880, 398],
 [1596, 126]]
```

Interpretation:

```text
TN = 7880
FP =  398
FN = 1596
TP =  126
```

---

# 16. Model Comparison

### Current comparison

| Experiment | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 82.78% | 0% | 0% | 0% |
| Balanced Logistic Regression | 61.98% | 25.06% | 60.69% | 35.47% |
| KNN | 80.06% | 24.05% | 7.32% | 11.22% |

### Important observation

The highest accuracy is **not automatically the best churn model**.

For churn prediction, identifying actual churners is important, so recall and F1 are important evaluation measures.

Based on the current three experiments:

> **Balanced Logistic Regression is the strongest setup for detecting the minority churn class.**

This conclusion is based on the observed results, not on accuracy alone.

---

# 17. Feature Engineering Impact ✅ COMPLETE

To satisfy the professor's requirement, we compared the same Balanced Logistic Regression setup:

```text
WITHOUT engineered features
        VS
WITH engineered features
```

### Before feature engineering

```text
Accuracy  = 62.12%
Precision = 25.11%
Recall    = 60.51%
F1 Score  = 35.49%
```

### After feature engineering

```text
Accuracy  = 61.98%
Precision ≈ 25.06%
Recall    = 60.69%
F1 Score  = 35.47%
```

### Observed change

```text
Accuracy  → slightly decreased
Precision → slightly decreased
Recall    → slightly increased
F1 Score  → slightly decreased
```

Therefore:

> The engineered features produced a very small recall improvement but did not produce an overall improvement in accuracy or F1 for this experiment.

This is a valid experimental finding and should be reported honestly.

---

# 18. Professor Requirements Checklist

```text
Dataset clearly identified                 ✅
Input parameters identified                ✅
Output parameter identified               ✅
Binary classification                     ✅
Missing-value checking                    ✅
Duplicate checking                        ✅
Data preprocessing                        ✅
Encoding                                  ✅
Scaling                                   ✅
Feature extraction / engineering          ✅
Before/after feature comparison           ✅
Multiple experiments                      ✅
Permutations / combinations                ✅
Class imbalance handling                  ✅
Confusion matrix                          ✅
Performance comparison                    ✅
Model comparison graph                    ✅
```

---

# 19. What Is Still Planned

## Evaluation 1 Finalization

```text
Final experiment ranking
        ↓
Best-model justification
        ↓
Final Evaluation 1 summary
        ↓
Presentation-ready results
```

## Evaluation 2 — Future Work

Potential additions:

```text
Random Forest
        ↓
Additional model comparison
        ↓
Hyperparameter experiments
        ↓
Feature-selection experiments
        ↓
Possible additional imbalance techniques
        ↓
Improved evaluation
```

Possible imbalance methods for future experimentation include oversampling techniques such as SMOTE, but they are **not part of Evaluation 1 yet**.

## Final System

```text
Best trained model
        ↓
Customer input
        ↓
Same preprocessing
        ↓
Churn probability
        ↓
Churn / No Churn prediction
        ↓
Explain prediction
        ↓
Deployment
```

---

# 20. Important Project Rules

### Rule 1 — Do not use leakage

Never use:

```text
subscription_end_date
```

as a model input.

### Rule 2 — Do not use identifiers as predictive features

Keep:

```text
user_id
```

for identification and traceability, but exclude it from `X`.

### Rule 3 — Do not judge the model only by accuracy

Always inspect:

```text
Accuracy
Precision
Recall
F1 Score
Confusion Matrix
```

### Rule 4 — Use actual experimental results

Do not claim a feature or model improved performance unless the measured results show the improvement.

### Rule 5 — Avoid unnecessary feature engineering

Only create features that have a clear business or mathematical meaning and can be tested experimentally.

### Rule 6 — Prevent preprocessing leakage

Fit preprocessing on training data and transform the test data using the already-fitted preprocessing objects.

---

# 21. Project Status Snapshot

```text
DATASET
✅ Identified

DATA QUALITY
✅ Checked

TARGET
✅ Defined
✅ Imbalance identified
✅ Imbalance handled for Evaluation 1

DATES
✅ Converted
✅ Validated

LEAKAGE
✅ Investigated
✅ Leakage-prone field excluded

PREPROCESSING
✅ Encoding
✅ Scaling
✅ Train/test split

FEATURE ENGINEERING
✅ tenure_days
✅ watch_time_per_session
✅ Before/after comparison

MODELS — EVALUATION 1
✅ Logistic Regression
✅ Balanced Logistic Regression
✅ KNN

EVALUATION
✅ Correct / incorrect predictions
✅ Accuracy
✅ Precision
✅ Recall
✅ F1
✅ Confusion matrices
✅ Comparison graph

NEXT
⏳ Finalize Evaluation 1
⏳ Best-model selection
⏳ Presentation summary
⏳ Evaluation 2
⏳ Random Forest
⏳ Final prediction system
⏳ Deployment
```

---

# 22. One-Line Project Flow

```text
Raw Dataset
→ Data Understanding
→ Data Quality
→ Target & Leakage Check
→ Feature Selection
→ Date Handling
→ Feature Engineering
→ Train/Test Split
→ Encoding & Scaling
→ Model Training
→ Multiple Experiments
→ Model Comparison
→ Confusion Matrix & Metrics
→ Best Model
→ Final Prediction System
→ Deployment
```
