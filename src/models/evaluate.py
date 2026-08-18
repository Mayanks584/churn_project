"""
src/models/evaluate.py
Model evaluation metrics and reporting.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve
)
import logging

logger = logging.getLogger(__name__)


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Comprehensive model evaluation.

    Args:
        model: Fitted sklearn model
        X_test: Test features
        y_test: True labels
        threshold: Classification threshold

    Returns:
        dict: All evaluation metrics
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "f1_score": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
    }

    logger.info(f"Evaluation Results:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    return metrics


def get_classification_report(model, X_test, y_test, threshold: float = 0.5) -> str:
    """Return formatted classification report string."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return classification_report(y_test, y_pred, target_names=["No Churn", "Churn"])


def compare_models(results: dict) -> pd.DataFrame:
    """
    Create comparison DataFrame of multiple model results.

    Args:
        results: dict of {model_name: metrics_dict}

    Returns:
        pd.DataFrame: Sorted comparison table
    """
    df = pd.DataFrame(results).T
    df = df.sort_values("roc_auc", ascending=False)
    df = df.round(4)
    logger.info("Model comparison:\n" + df.to_string())
    return df


def save_metrics(metrics: dict, filepath: str) -> None:
    """Save metrics dictionary to JSON."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {filepath}")
