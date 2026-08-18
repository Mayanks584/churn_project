"""
scripts/train_pipeline.py
End-to-end training pipeline.

Usage:
    python scripts/train_pipeline.py
    python scripts/train_pipeline.py --config configs/config.yaml
"""

import argparse
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.helpers import setup_logging, load_yaml, timer
from src.data.load_data import load_raw_data, save_processed_data
from src.data.preprocess import clean_data, encode_target, build_preprocessor
from src.features.build_features import run_feature_engineering
from src.models.train import get_models, apply_smote, train_model, save_model
from src.models.evaluate import evaluate_model, compare_models, save_metrics
from src.visualization.plots import (
    plot_churn_distribution, plot_confusion_matrix, plot_roc_curve,
    plot_model_comparison, plot_feature_importance, save_fig
)
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


@timer
def main(config_path: str = "configs/config.yaml"):
    setup_logging()
    config = load_yaml(config_path)
    logger.info("=" * 60)
    logger.info("  CHURN PREDICTION — TRAINING PIPELINE")
    logger.info("=" * 60)

    # ── 1. Load & Clean Data ──────────────────────────────────────
    logger.info("Step 1: Loading and cleaning data...")
    df = load_raw_data(config["paths"]["raw_data"])
    df = clean_data(df)
    df = encode_target(df, config["data"]["target_column"])

    # EDA plot
    fig = plot_churn_distribution(df)
    save_fig(fig, "churn_distribution.png")

    # ── 2. Feature Engineering ────────────────────────────────────
    logger.info("Step 2: Feature engineering...")
    df = run_feature_engineering(df)

    # ── 3. Split ──────────────────────────────────────────────────
    logger.info("Step 3: Splitting data...")
    drop_cols = config["features"]["drop_columns"] + [config["data"]["target_column"]]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols]
    y = df[config["data"]["target_column"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=y,
    )

    # ── 4. Preprocess ─────────────────────────────────────────────
    logger.info("Step 4: Preprocessing features...")
    cat_cols = [c for c in config["features"]["categorical_columns"] if c in X.columns]
    num_cols = [c for c in config["features"]["numerical_columns"] if c in X.columns]
    preprocessor = build_preprocessor(cat_cols, num_cols)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # SMOTE
    if config["training"]["handle_imbalance"]:
        X_train_proc, y_train = apply_smote(X_train_proc, y_train)

    # ── 5. Train Models ───────────────────────────────────────────
    logger.info("Step 5: Training models...")
    models = get_models(config)
    all_results = {}
    trained_models = {}

    for name, model in models.items():
        logger.info(f"  Training: {name}")
        trained = train_model(model, X_train_proc, y_train, name, config)
        metrics = evaluate_model(trained, X_test_proc, y_test)
        all_results[name] = metrics
        trained_models[name] = trained
        save_model(trained, name)

        # Plots
        save_fig(plot_confusion_matrix(trained, X_test_proc, y_test), f"cm_{name}.png")
        save_fig(plot_roc_curve(trained, X_test_proc, y_test), f"roc_{name}.png")

    # ── 6. Compare & Save ─────────────────────────────────────────
    logger.info("Step 6: Comparing models...")
    comparison = compare_models(all_results)
    print("\n" + comparison.to_string())
    save_fig(plot_model_comparison(comparison), "model_comparison.png")

    best_model_name = comparison.index[0]
    save_metrics(all_results, f"reports/metrics/all_metrics.json")
    logger.info(f"\n✅ Best model: {best_model_name} (ROC-AUC: {comparison.iloc[0]['roc_auc']:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Churn Prediction Training Pipeline")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML")
    args = parser.parse_args()
    main(args.config)
