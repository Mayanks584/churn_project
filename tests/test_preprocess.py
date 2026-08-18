"""
tests/test_preprocess.py
Unit tests for data preprocessing functions.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocess import clean_data, encode_target


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "customerID": ["001", "002", "003", "004"],
        "tenure": [5, 24, 0, 60],
        "MonthlyCharges": [29.9, 70.5, 45.0, 99.9],
        "TotalCharges": ["149.5", "1692.0", " ", "5994.0"],
        "Contract": ["Month-to-month", "One year", "Month-to-month", "Two year"],
        "Churn": ["Yes", "No", "Yes", "No"],
    })


def test_clean_data_fixes_total_charges(sample_df):
    df = clean_data(sample_df)
    assert df["TotalCharges"].dtype in [float, np.float64]
    assert df["TotalCharges"].isna().sum() == 0


def test_clean_data_removes_duplicates():
    df = pd.DataFrame({
        "tenure": [5, 5],
        "MonthlyCharges": [29.9, 29.9],
        "TotalCharges": ["149.5", "149.5"],
        "Churn": ["Yes", "Yes"],
    })
    cleaned = clean_data(df)
    assert len(cleaned) == 1


def test_encode_target(sample_df):
    df = clean_data(sample_df)
    df = encode_target(df, "Churn")
    assert set(df["Churn"].unique()).issubset({0, 1})
    assert df["Churn"].dtype in [int, np.int64]


def test_clean_data_preserves_rows(sample_df):
    df = clean_data(sample_df)
    assert len(df) <= len(sample_df)
    assert len(df) > 0
