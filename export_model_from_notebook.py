"""Controlled notebook-driven exporter for the Netflix Customer Churn V1 frontend.

Executes ONLY the minimal verified dependency chain from
Netflix_Churn_Syllabus_Implementation_v2.ipynb, verifies exact and tolerance-based
reproducibility against the notebook's held-out X_test outputs, computes single-model
X_train 5-fold out-of-fold risk-band statistics, and exports the fitted Pipeline and
strict 24-feature input contract.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import joblib
import nbformat
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict

PROJECT_ROOT = Path(__file__).resolve().parent
NB_PATH = PROJECT_ROOT / "Netflix_Churn_Syllabus_Implementation_v2.ipynb"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_OUT = ARTIFACTS_DIR / "netflix_churn_rf_pipeline.joblib"
CONTRACT_OUT = ARTIFACTS_DIR / "model_contract.json"

# Exact minimal cell indices from Netflix_Churn_Syllabus_Implementation_v2.ipynb
DEPENDENCY_CELLS = [2, 6, 36, 90, 103, 107, 109, 113, 115, 117, 119, 121, 200]

# Verified X_test metrics from Cell 224 (ec=109) of Netflix_Churn_Syllabus_Implementation_v2.ipynb
EXPECTED_TEST_CM = np.array([[5267, 3011], [695, 1027]], dtype=int)
EXPECTED_TEST_METRICS = {
    "accuracy": 0.6294,
    "precision": 0.2543,
    "recall": 0.5964,
    "f1": 0.3566,
    "roc_auc": 0.6384,
    "average_precision": 0.2442,
    "pr_auc_trapezoidal": 0.2437,
}
METRIC_ATOL = 1e-4


def run_notebook_dependency_chain() -> dict:
    """Execute only the exact notebook dependency cells in a clean namespace."""
    nb = nbformat.read(NB_PATH, as_version=4)

    ns: dict = {
        "__name__": "__notebook_export__",
        "display": lambda *args, **kwargs: None,
    }
    # Metric & RF imports needed by Cell 200 / final_rf_model without running Cell 134's y_pred_lr
    exec(
        "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score",
        ns,
    )

    old_cwd = os.getcwd()
    os.chdir(PROJECT_ROOT)
    try:
        for cell_idx in DEPENDENCY_CELLS:
            cell = nb.cells[cell_idx]
            if cell.cell_type != "code":
                raise RuntimeError(f"Expected code cell at index {cell_idx}, got {cell.cell_type}")
            exec(compile(cell.source, f"{NB_PATH.name}#cell_{cell_idx}", "exec"), ns)
    finally:
        os.chdir(old_cwd)

    return ns


def verify_one_to_one_mappings(h_df) -> tuple[dict[str, float], dict[str, str]]:
    """Verify 100% deterministic subscription_plan->monthly_fee and country->region maps."""
    plan_fee_pairs = h_df[["subscription_plan", "monthly_fee"]].drop_duplicates()
    if len(plan_fee_pairs) != h_df["subscription_plan"].nunique():
        raise AssertionError("subscription_plan -> monthly_fee is not 1-to-1 in dataset!")
    plan_to_fee = {
        str(row.subscription_plan): float(row.monthly_fee)
        for row in plan_fee_pairs.itertuples(index=False)
    }

    country_region_pairs = h_df[["country", "region"]].drop_duplicates()
    if len(country_region_pairs) != h_df["country"].nunique():
        raise AssertionError("country -> region is not 1-to-1 in dataset!")
    country_to_region = {
        str(row.country): str(row.region)
        for row in country_region_pairs.itertuples(index=False)
    }
    return plan_to_fee, country_to_region


def main() -> None:
    print(f"1. Executing minimal dependency chain from {NB_PATH.name} (cells {DEPENDENCY_CELLS})...")
    ns = run_notebook_dependency_chain()

    h = ns["h"]
    X_train = ns["X_train"]
    y_train = ns["y_train"]
    X_test = ns["X_test"]
    y_test = ns["y_test"]
    numeric_features = ns["numeric_features"]
    categorical_features = ns["categorical_features"]
    final_features = ns["final_features"]
    reference_date = ns["reference_date"]
    tree_preprocessor = ns["tree_preprocessor"]
    cv_5fold = ns["cv_5fold"]
    pr_auc_trap_func = ns["pr_auc_trap_func"]
    accuracy_score = ns["accuracy_score"]
    precision_score = ns["precision_score"]
    recall_score = ns["recall_score"]
    f1_score = ns["f1_score"]
    roc_auc_score = ns["roc_auc_score"]
    average_precision_score = ns["average_precision_score"]

    assert list(X_train.columns) == final_features, "X_train column order mismatch!"
    assert len(final_features) == 24, f"Expected 24 features, got {len(final_features)}"

    # 2. Verify dataset 1-to-1 mappings
    plan_to_fee, country_to_region = verify_one_to_one_mappings(h)
    print("2. Verified 1-to-1 mappings:", {"plan_to_fee": plan_to_fee, "countries": len(country_to_region)})

    # 3. Construct & fit exact Phase 7 winning Tuned Random Forest Pipeline
    best_rf_pipe = ns["Pipeline"]([
        ("prep", clone(tree_preprocessor)),
        (
            "clf",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ])
    final_rf_model = clone(best_rf_pipe).fit(X_train, y_train)

    # 4. Assert exact confusion matrix & tolerance-based metric parity on X_test
    y_pred = final_rf_model.predict(X_test)
    y_prob = final_rf_model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)

    if not np.array_equal(cm, EXPECTED_TEST_CM):
        raise AssertionError(f"Confusion matrix mismatch! Got:\n{cm}\nExpected:\n{EXPECTED_TEST_CM}")

    actual_metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "average_precision": float(average_precision_score(y_test, y_prob)),
        "pr_auc_trapezoidal": float(pr_auc_trap_func(y_test, y_prob)),
    }
    for k, exp_v in EXPECTED_TEST_METRICS.items():
        diff = abs(actual_metrics[k] - exp_v)
        if diff > METRIC_ATOL:
            raise AssertionError(f"Metric {k} mismatch: got {actual_metrics[k]:.6f}, expected {exp_v:.4f} (diff={diff:.6f})")

    print("3. Verified exact X_test confusion matrix and metric parity within atol=1e-4:")
    print("   CM:", cm.tolist())
    print("   Metrics:", {k: round(v, 4) for k, v in actual_metrics.items()})

    # 5. Export fitted Pipeline and verify bit-for-bit reload parity
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_rf_model, MODEL_OUT)
    reloaded = joblib.load(MODEL_OUT)

    y_pred_reloaded = reloaded.predict(X_test)
    y_prob_reloaded = reloaded.predict_proba(X_test)[:, 1]
    if not np.array_equal(y_pred_reloaded, y_pred):
        raise AssertionError("Reloaded model predict(X_test) does not match exactly!")
    if not np.allclose(y_prob_reloaded, y_prob, atol=1e-12):
        raise AssertionError("Reloaded model predict_proba(X_test) exceeds atol=1e-12!")
    print(f"4. Exported {MODEL_OUT.relative_to(PROJECT_ROOT)} and verified exact reload parity.")

    # 6. Single-model 5-fold CV on X_train ONLY for risk-band calibration/documentation
    print("5. Running single-model 5-fold cross_val_predict on X_train for risk-band documentation...")
    oof_probs_train = cross_val_predict(
        clone(best_rf_pipe), X_train, y_train, cv=cv_5fold, method="predict_proba", n_jobs=-1
    )[:, 1]
    y_tr_np = y_train.to_numpy()

    def band_stats(mask: np.ndarray) -> dict:
        cnt = int(mask.sum())
        churn_rate = float(y_tr_np[mask].mean()) if cnt > 0 else 0.0
        return {"train_cv_count": cnt, "train_cv_share_pct": round(cnt / len(y_tr_np) * 100, 2), "observed_churn_rate_pct": round(churn_rate * 100, 2)}

    risk_bands = [
        {
            "name": "Low",
            "min_prob_inclusive": 0.0,
            "max_prob_exclusive": 0.40,
            "basis": "Demonstration tier boundary below the balanced model's 0.50 classification cutoff.",
            **band_stats(oof_probs_train < 0.40),
        },
        {
            "name": "Moderate",
            "min_prob_inclusive": 0.40,
            "max_prob_exclusive": 0.50,
            "basis": "Demonstration watch-zone approaching the balanced model's 0.50 classification cutoff.",
            **band_stats((oof_probs_train >= 0.40) & (oof_probs_train < 0.50)),
        },
        {
            "name": "High",
            "min_prob_inclusive": 0.50,
            "max_prob_exclusive": 1.0001,
            "basis": "Exact binary classification decision boundary of final_rf_model.predict() (class_weight='balanced').",
            **band_stats(oof_probs_train >= 0.50),
        },
    ]

    # 7. Training-only descriptive statistics & global context (from X_train, never X_test)
    train_num_stats = {}
    active_m = y_train == 0
    cancel_m = y_train == 1
    for col in numeric_features:
        s = X_train[col]
        train_num_stats[col] = {
            "min": float(s.min()),
            "max": float(s.max()),
            "overall_mean": round(float(s.mean()), 2),
            "active_mean": round(float(s[active_m].mean()), 2),
            "cancelled_mean": round(float(s[cancel_m].mean()), 2),
        }

    plan_churn_rates = {}
    for plan_name, grp_y in y_train.groupby(X_train["subscription_plan"]):
        plan_churn_rates[str(plan_name)] = {
            "train_count": int(len(grp_y)),
            "churn_rate_pct": round(float(grp_y.mean() * 100), 2),
        }

    cat_allowed = {}
    for col in categorical_features:
        vals = X_train[col].unique().tolist()
        if col == "age_group":
            cat_allowed[col] = sorted(int(v) for v in vals)
        else:
            cat_allowed[col] = sorted(str(v) for v in vals)

    # Synthetic UI test presets (explicitly marked as synthetic for interface testing only)
    synthetic_presets = {
        "high_risk_demo": {
            "label": "Synthetic Demo A: Basic Low-Weekly-Watch Profile",
            "note": "Synthetic interface test example (not a real customer record).",
            "inputs": {
                "subscription_plan": "Basic",
                "country": "India",
                "age_group": 21,
                "gender": "Male",
                "payment_method": "Wallet",
                "discount_applied": "No",
                "content_type": "Series",
                "genre": "Drama",
                "language": "Hindi",
                "device_type": "Mobile",
                "time_of_day": "Night",
                "rating": 3,
                "liked": "No",
                "recommendation_source": "Ads",
                "watch_time_minutes": 45,
                "session_count": 2,
                "completion_percentage": 68,
                "days_since_last_watch": 35,
                "avg_weekly_watch_time": 120,
                "content_diversity_score": 0.45,
                "tenure_days": 450,
            },
        },
        "moderate_risk_demo": {
            "label": "Synthetic Demo B: Standard Mid-Engagement Profile",
            "note": "Synthetic interface test example (not a real customer record).",
            "inputs": {
                "subscription_plan": "Standard",
                "country": "USA",
                "age_group": 30,
                "gender": "Female",
                "payment_method": "Card",
                "discount_applied": "Yes",
                "content_type": "Movie",
                "genre": "Thriller",
                "language": "English",
                "device_type": "Laptop",
                "time_of_day": "Evening",
                "rating": 4,
                "liked": "Yes",
                "recommendation_source": "Algorithm",
                "watch_time_minutes": 85,
                "session_count": 2,
                "completion_percentage": 92,
                "days_since_last_watch": 18,
                "avg_weekly_watch_time": 240,
                "content_diversity_score": 0.60,
                "tenure_days": 740,
            },
        },
        "low_risk_demo": {
            "label": "Synthetic Demo C: Premium High-Engagement Profile",
            "note": "Synthetic interface test example (not a real customer record).",
            "inputs": {
                "subscription_plan": "Premium",
                "country": "UK",
                "age_group": 40,
                "gender": "Female",
                "payment_method": "Bank Transfer",
                "discount_applied": "No",
                "content_type": "Series",
                "genre": "Sci-Fi",
                "language": "English",
                "device_type": "TV",
                "time_of_day": "Evening",
                "rating": 5,
                "liked": "Yes",
                "recommendation_source": "Friend",
                "watch_time_minutes": 140,
                "session_count": 4,
                "completion_percentage": 98,
                "days_since_last_watch": 5,
                "avg_weekly_watch_time": 400,
                "content_diversity_score": 0.78,
                "tenure_days": 950,
            },
        },
    }

    contract = {
        "model_name": "Tuned Random Forest (Balanced)",
        "model_version": "v1.0.0-rf-v2nb",
        "notebook_source": NB_PATH.name,
        "executed_dependency_cells": DEPENDENCY_CELLS,
        "sklearn_version": __import__("sklearn").__version__,
        "pandas_version": __import__("pandas").__version__,
        "reference_date": reference_date.strftime("%Y-%m-%d"),
        "training_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "feature_order": final_features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "derived_features": {
            "monthly_fee": "Derived from verified 1-to-1 subscription_plan -> monthly_fee mapping.",
            "region": "Derived from verified 1-to-1 country -> region mapping.",
            "watch_time_per_session": "watch_time_minutes / session_count (requires session_count >= 1).",
            "tenure_days": "(reference_date - subscription_start_date).days if subscription_start_date supplied, else direct tenure_days.",
        },
        "plan_to_fee": plan_to_fee,
        "country_to_region": country_to_region,
        "allowed_categories": cat_allowed,
        "numeric_bounds": {
            "monthly_fee": {"min": 8.99, "max": 17.99, "type": "float"},
            "watch_time_minutes": {"min": 1.0, "max": 1440.0, "type": "float"},
            "session_count": {"min": 1, "max": 100, "type": "int"},
            "completion_percentage": {"min": 0.0, "max": 100.0, "type": "float"},
            "rating": {"min": 1.0, "max": 5.0, "type": "float"},
            "days_since_last_watch": {"min": 0.0, "max": 3650.0, "type": "float"},
            "avg_weekly_watch_time": {"min": 0.0, "max": 10080.0, "type": "float"},
            "content_diversity_score": {"min": 0.0, "max": 1.0, "type": "float"},
            "tenure_days": {"min": 0.0, "max": 5000.0, "type": "float"},
            "watch_time_per_session": {"min": 0.01, "max": 1440.0, "type": "float"},
        },
        "risk_bands": risk_bands,
        "verified_test_metrics": {k: round(v, 4) for k, v in actual_metrics.items()},
        "verified_test_confusion_matrix": {"TN": int(cm[0, 0]), "FP": int(cm[0, 1]), "FN": int(cm[1, 0]), "TP": int(cm[1, 1])},
        "train_numeric_reference_stats": train_num_stats,
        "train_plan_churn_rates": plan_churn_rates,
        "global_permutation_importance_top5": [
            {"feature": "subscription_plan", "perm_f1_drop": 0.04064},
            {"feature": "monthly_fee", "perm_f1_drop": 0.00549},
            {"feature": "avg_weekly_watch_time", "perm_f1_drop": 0.00134},
            {"feature": "completion_percentage", "perm_f1_drop": 0.00087},
            {"feature": "days_since_last_watch", "perm_f1_drop": 0.00081},
        ],
        "synthetic_presets": synthetic_presets,
    }

    with open(CONTRACT_OUT, "w", encoding="utf-8") as f:
        json.dump(contract, f, indent=2)

    print(f"6. Exported {CONTRACT_OUT.relative_to(PROJECT_ROOT)} with X_train CV risk bands:")
    for rb in risk_bands:
        print(f"   - {rb['name']:8s}: count={rb['train_cv_count']}, share={rb['train_cv_share_pct']}%, observed_churn={rb['observed_churn_rate_pct']}%")


if __name__ == "__main__":
    main()

