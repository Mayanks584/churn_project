"""
src/models/predict.py
Load saved model and run inference.
"""

import joblib
import numpy as np
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_model(model_path: str):
    """Load a saved joblib model."""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")
    model = joblib.load(model_path)
    logger.info(f"Model loaded from {model_path}")
    return model


def predict_proba(model, X: np.ndarray) -> np.ndarray:
    """Return churn probabilities."""
    return model.predict_proba(X)[:, 1]


def predict_churn(model, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Return binary churn predictions (0 = No Churn, 1 = Churn)."""
    proba = predict_proba(model, X)
    return (proba >= threshold).astype(int)


def predict_with_confidence(model, X: np.ndarray, threshold: float = 0.5) -> pd.DataFrame:
    """
    Return a DataFrame with predictions, probabilities, and risk level.

    Risk levels:
        Low:    probability < 0.3
        Medium: 0.3 <= probability < 0.6
        High:   probability >= 0.6
    """
    proba = predict_proba(model, X)
    predictions = (proba >= threshold).astype(int)

    risk_levels = pd.cut(
        proba,
        bins=[0, 0.3, 0.6, 1.0],
        labels=["Low", "Medium", "High"],
        include_lowest=True
    )

    result = pd.DataFrame({
        "churn_probability": proba.round(4),
        "prediction": predictions,
        "churn_label": ["Yes" if p == 1 else "No" for p in predictions],
        "risk_level": risk_levels,
    })

    logger.info(f"Predictions generated for {len(result)} samples")
    logger.info(f"Churn rate: {predictions.mean():.2%}")
    return result
