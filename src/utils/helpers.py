"""
src/utils/helpers.py
General utility functions.
"""

import yaml
import json
import logging
import logging.config
import time
from pathlib import Path
from functools import wraps


def load_yaml(path: str) -> dict:
    """Load a YAML file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_json(data: dict, path: str) -> None:
    """Save a dictionary to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def setup_logging(level: str = "INFO", log_file: str = "logs/run.log") -> None:
    """Configure logging to console and file."""
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode="a"),
        ]
    )


def timer(func):
    """Decorator to measure and log function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logging.getLogger(__name__).info(
            f"{func.__name__} completed in {elapsed:.2f}s"
        )
        return result
    return wrapper


def ensure_dir(path: str) -> Path:
    """Create directory if it doesn't exist and return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
