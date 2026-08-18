"""
src/features/build_features.py
Feature engineering for churn prediction.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def create_tenure_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Bin tenure into risk groups."""
    df = df.copy()
    bins = [0, 12, 24, 48, 72]
    labels = ["0-1yr", "1-2yr", "2-4yr", "4+yr"]
    df["tenure_group"] = pd.cut(df["tenure"], bins=bins, labels=labels, right=True)
    logger.info("Created 'tenure_group' feature")
    return df


def create_charges_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Create monthly-to-total charges ratio."""
    df = df.copy()
    df["charges_ratio"] = df["MonthlyCharges"] / (df["TotalCharges"] + 1e-6)
    logger.info("Created 'charges_ratio' feature")
    return df


def create_service_count(df: pd.DataFrame) -> pd.DataFrame:
    """Count number of services subscribed by customer."""
    df = df.copy()
    service_cols = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    available_cols = [c for c in service_cols if c in df.columns]
    # Count "Yes" entries
    df["service_count"] = df[available_cols].apply(
        lambda row: sum(1 for v in row if str(v).lower() in ["yes", "dsl", "fiber optic"]), axis=1
    )
    logger.info(f"Created 'service_count' feature from {len(available_cols)} service columns")
    return df


def create_is_senior_partner(df: pd.DataFrame) -> pd.DataFrame:
    """Create a flag for senior citizen with no partner."""
    df = df.copy()
    if "SeniorCitizen" in df.columns and "Partner" in df.columns:
        df["senior_no_partner"] = (
            (df["SeniorCitizen"] == 1) & (df["Partner"].str.lower() == "no")
        ).astype(int)
        logger.info("Created 'senior_no_partner' feature")
    return df


def run_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Run all feature engineering steps."""
    df = create_tenure_groups(df)
    df = create_charges_ratio(df)
    df = create_service_count(df)
    df = create_is_senior_partner(df)
    logger.info(f"Feature engineering complete. Shape: {df.shape}")
    return df
