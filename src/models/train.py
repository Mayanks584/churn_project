"""
src/models/train.py
Model training with multiple algorithms + MLflow tracking.
"""

import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import logging
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

logger = logging.getLogger(__name__)


def get_models(config: dict) -> dict:
    """Return dictionary of model instances from config."""
    m = config["models"]
    return {
        "logistic_regression": LogisticRegression(**m["logistic_regression"]),
        "random_forest": RandomForestClassifier(**m["random_forest"]),
        "xgboost": XGBClassifier(**m["xgboost"]),
        "lightgbm": LGBMClassifier(**m["lightgbm"]),
    }


def apply_smote(X: np.ndarray, y: np.ndarray, random_state: int = 42):
    """Apply SMOTE oversampling to handle class imbalance."""
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    logger.info(f"SMOTE applied. Samples: {len(y)} → {len(y_resampled)}")
    return X_resampled, y_resampled


def train_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str,
    config: dict,
    cv_folds: int = 5,
) -> object:
    """
    Train a single model with cross-validation and MLflow logging.

    Args:
        model: sklearn-compatible model
        X_train: Training features
        y_train: Training labels
        model_name: Name for logging
        config: Project config dict
        cv_folds: Number of CV folds

    Returns:
        Fitted model
    """
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=model_name):
        # Cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        logger.info(f"[{model_name}] CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Full train
        model.fit(X_train, y_train)

        # Log to MLflow
        mlflow.log_metric("cv_roc_auc_mean", cv_scores.mean())
        mlflow.log_metric("cv_roc_auc_std", cv_scores.std())
        mlflow.sklearn.log_model(model, artifact_path=model_name)

        logger.info(f"[{model_name}] Training complete")

    return model


def save_model(model, model_name: str, save_dir: str = "models/saved/") -> str:
    """Save trained model using joblib."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    filepath = Path(save_dir) / f"{model_name}.joblib"
    joblib.dump(model, filepath)
    logger.info(f"Model saved: {filepath}")
    return str(filepath)
