"""
src/data/preprocess.py
Data cleaning and preprocessing pipeline.
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import logging

logger = logging.getLogger(__name__)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw dataframe:
    - Handle missing values
    - Fix data types
    - Remove duplicates

    Args:
        df: Raw DataFrame

    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    df = df.copy()

    # Fix TotalCharges column (may have spaces instead of NaN)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop duplicates
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    dropped = initial_rows - len(df)
    if dropped:
        logger.warning(f"Dropped {dropped} duplicate rows")

    # Fill missing numerics with median
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns
    for col in num_cols:
        missing = df[col].isna().sum()
        if missing:
            df[col].fillna(df[col].median(), inplace=True)
            logger.info(f"Filled {missing} missing values in '{col}' with median")

    logger.info(f"Data cleaned: {df.shape[0]} rows remaining")
    return df


def encode_target(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """Encode binary target column to 0/1."""
    df = df.copy()
    if df[target_col].dtype == object:
        df[target_col] = df[target_col].map({"Yes": 1, "No": 0})
    logger.info(f"Target '{target_col}' encoded. Churn rate: {df[target_col].mean():.2%}")
    return df


def build_preprocessor(categorical_cols: list, numerical_cols: list) -> ColumnTransformer:
    """
    Build a sklearn ColumnTransformer for numerical and categorical features.

    Args:
        categorical_cols: List of categorical column names
        numerical_cols: List of numerical column names

    Returns:
        ColumnTransformer: Preprocessing pipeline
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", num_pipeline, numerical_cols),
        ("cat", cat_pipeline, categorical_cols),
    ])

    logger.info("Preprocessor built successfully")
    return preprocessor
