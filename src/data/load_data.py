"""
src/data/load_data.py
Utilities for loading raw and processed data.
"""

import pandas as pd
import yaml
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Config loaded from {config_path}")
    return config


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Load raw CSV dataset.

    Args:
        filepath: Path to the raw CSV file.

    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_csv(filepath)
    logger.info(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def load_processed_data(filepath: str) -> pd.DataFrame:
    """Load preprocessed data from parquet or CSV."""
    path = Path(filepath)
    if path.suffix == ".parquet":
        df = pd.read_parquet(filepath)
    else:
        df = pd.read_csv(filepath)
    logger.info(f"Loaded processed data: {df.shape}")
    return df


def save_processed_data(df: pd.DataFrame, filepath: str) -> None:
    """Save processed dataframe to parquet."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(filepath, index=False)
    logger.info(f"Processed data saved to {filepath}")
