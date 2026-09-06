                 ┌──────────────────────────┐
                 │     1. DATASET           │
                 │  Netflix Churn Dataset   │
                 │  50,000 rows, 29 cols    │
                 └────────────┬─────────────┘
                              ↓
                 ┌──────────────────────────┐
                 │ 2. INPUT & OUTPUT        │
                 │                          │
                 │ Input  → customer data   │
                 │ Output → churn_status    │
                 │ Active / Cancelled       │
                 └────────────┬─────────────┘
                              ↓
              ╔══════════════════════════════════╗
              ║ 3. DATA PREPROCESSING            ║
              ╚══════════════════════════════════╝
                              ↓
          ┌────────────────────────────────────────┐
          │ 3.1 Missing Value Check         ✅     │
          │ 3.2 Duplicate Check             ✅     │
          │ 3.3 Date Validation / Leakage   ✅     │
          │ 3.4 Categorical Encoding        ✅     │
          │ 3.5 Numerical Scaling           ✅     │
          │ 3.6 Outlier Analysis            ✅     │
          └──────────────────┬─────────────────────┘
                             ↓
              ╔══════════════════════════════════╗
              ║ 4. FEATURE ENGINEERING           ║
              ╚══════════════════════════════════╝
                             ↓
              ┌──────────────────────────────┐
              │ Create meaningful NEW        │
              │ features from existing data  │
              │                              │
              │ Example:                    │
              │ subscription_start_date     │
              │          ↓                  │
              │       tenure_days           │
              └──────────────┬───────────────┘
                             ↓
                ┌─────────────────────────┐
                │ 5. FEATURE SELECTION     │
                │                         │
                │ Remove: user_id         │
                │         title           │
                │         leakage columns  │
                │ Keep useful features     │
                └────────────┬────────────┘
                             ↓
              ╔══════════════════════════════════╗
              ║ 6. TRAIN / TEST SPLIT            ║
              ║    Stratified Split              ║
              ╚══════════════════════════════════╝
                             ↓
        ┌────────────────────────────────────────────┐
        │ 7. MACHINE LEARNING MODELS                 │
        │                                            │
        │ Logistic Regression                        │
        │ Random Forest                              │
        │ XGBoost / other suitable classifier       │
        └────────────────────┬───────────────────────┘
                             ↓
              ╔══════════════════════════════════╗
              ║ 8. MULTIPLE EXPERIMENTS          ║
              ╚══════════════════════════════════╝
                             ↓
       ┌──────────────────────────────────────────────┐
       │ Compare different combinations               │
       │                                              │
       │ Base Features                                │
       │       vs                                     │
       │ Engineered Features                          │
       │                                              │
       │ Model A vs Model B vs Model C                │
       │                                              │
       │ Different preprocessing combinations        │
       └─────────────────────┬────────────────────────┘
                             ↓
              ╔══════════════════════════════════╗
              ║ 9. EVALUATION                    ║
              ╚══════════════════════════════════╝
                             ↓
        ┌────────────────────────────────────────────┐
        │ Accuracy                                   │
        │ Precision                                  │
        │ Recall                                     │
        │ F1 Score                                   │
        │ ROC-AUC                                    │
        │                                            │
        │ ⭐ CONFUSION MATRIX                         │
        └────────────────────┬───────────────────────┘
                             ↓
              ┌─────────────────────────────┐
              │ 10. BEST MODEL              │
              │                             │
              │ Select based on actual      │
              │ experimental results        │
              └──────────────┬──────────────┘
                             ↓
              ┌─────────────────────────────┐
              │ 11. FEATURE IMPORTANCE      │
              │ / SHAP                     │
              │                             │
              │ Why did customer churn?    │
              └──────────────┬──────────────┘
                             ↓
              ┌─────────────────────────────┐
              │ 12. DEPLOYMENT              │
              │                             │
              │ Streamlit / Flask          │
              │ User → inputs → prediction  │
              └─────────────────────────────┘



              model final structure 

                               ORIGINAL DATA
                       │
        ┌──────────────┼──────────────┐
        ↓              ↓              ↓
     KEEP          TRANSFORM         DROP
        │              │              │
   useful cols      start_date      user_id
                    → tenure        title
                    → watch/session end_date
                                      release_year
                                      date_watched
        │
        └──────────────┐
                       ↓
                FINAL FEATURES
                       │
             categorical + numeric
                       │
                       ↓
                  ML MODEL
                       │
                       ↓
                churn_status
              Active / Cancelled