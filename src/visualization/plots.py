"""
src/visualization/plots.py
Visualization utilities for EDA and model evaluation.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
)
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# --- Plot Style ---
plt.rcParams.update({
    "figure.dpi": 150,
    "figure.figsize": (10, 6),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})
PALETTE = "viridis"


def save_fig(fig, filename: str, save_dir: str = "reports/figures/") -> None:
    """Save matplotlib figure to reports directory."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    filepath = Path(save_dir) / filename
    fig.savefig(filepath, bbox_inches="tight")
    logger.info(f"Figure saved: {filepath}")


def plot_churn_distribution(df: pd.DataFrame, target: str = "Churn") -> plt.Figure:
    """Plot target class distribution."""
    fig, ax = plt.subplots()
    counts = df[target].value_counts()
    bars = ax.bar(["No Churn", "Churn"], counts.values, color=["#4CAF50", "#F44336"], edgecolor="white")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{val}\n({val/counts.sum():.1%})", ha="center", fontsize=11)
    ax.set_title("Churn Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count")
    return fig


def plot_feature_correlation(df: pd.DataFrame) -> plt.Figure:
    """Plot heatmap of feature correlations."""
    fig, ax = plt.subplots(figsize=(12, 8))
    num_df = df.select_dtypes(include="number")
    sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    return fig


def plot_confusion_matrix(model, X_test, y_test) -> plt.Figure:
    """Plot confusion matrix."""
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test,
        display_labels=["No Churn", "Churn"],
        cmap="Blues", ax=ax
    )
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    return fig


def plot_roc_curve(model, X_test, y_test) -> plt.Figure:
    """Plot ROC curve."""
    fig, ax = plt.subplots()
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax.legend()
    return fig


def plot_feature_importance(model, feature_names: list, top_n: int = 20) -> plt.Figure:
    """Plot top N feature importances (for tree-based models)."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        [feature_names[i] for i in indices][::-1],
        importances[indices][::-1],
        color="#5C6BC0"
    )
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance Score")
    return fig


def plot_model_comparison(comparison_df: pd.DataFrame) -> plt.Figure:
    """Bar chart comparing model ROC-AUC scores."""
    fig, ax = plt.subplots()
    bars = ax.bar(comparison_df.index, comparison_df["roc_auc"],
                  color=sns.color_palette(PALETTE, len(comparison_df)))
    for bar, val in zip(bars, comparison_df["roc_auc"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", fontsize=10)
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Model Comparison — ROC-AUC", fontsize=14, fontweight="bold")
    ax.set_ylabel("ROC-AUC Score")
    ax.set_xlabel("Model")
    return fig
