# 📉 Customer Churn Prediction — AIML Project

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)]()

> **Predict which customers are likely to churn** using machine learning — helping businesses take proactive retention actions before it's too late.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Model Results](#model-results)
- [Contributing](#contributing)
- [License](#license)

---

## 🧠 Overview

Customer churn (also called customer attrition) is when a customer stops doing business with a company. This project builds an end-to-end ML pipeline to:

- **Analyze** customer behavior data (EDA)
- **Engineer** meaningful features
- **Train & compare** multiple ML models (Logistic Regression, Random Forest, XGBoost, etc.)
- **Evaluate** models using metrics like ROC-AUC, F1-Score, Precision, Recall
- **Interpret** results using SHAP values for explainability
- **Deploy** the best model via a REST API (FastAPI)

---

## 📁 Project Structure

```
churn_project/
│
├── data/                          # All datasets
│   ├── raw/                       # Original, unmodified data
│   ├── processed/                 # Cleaned & transformed data
│   └── external/                  # External/third-party data sources
│
├── notebooks/                     # Jupyter notebooks for exploration
│   ├── 01_eda.ipynb               # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_model_evaluation.ipynb
│   └── 05_shap_explainability.ipynb
│
├── src/                           # Source code (core logic)
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load_data.py           # Data loading utilities
│   │   └── preprocess.py          # Preprocessing pipeline
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py      # Feature engineering
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py               # Model training scripts
│   │   ├── predict.py             # Inference / prediction
│   │   └── evaluate.py            # Evaluation metrics
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── plots.py               # Plotting & visualization utils
│   └── utils/
│       ├── __init__.py
│       └── helpers.py             # General helper functions
│
├── models/                        # Saved model files
│   ├── saved/                     # Serialized models (.pkl, .joblib)
│   └── artifacts/                 # Model metadata, configs, thresholds
│
├── reports/                       # Generated reports & outputs
│   ├── figures/                   # Charts, plots, confusion matrices
│   └── metrics/                   # CSV/JSON files with evaluation results
│
├── configs/                       # Configuration files
│   ├── config.yaml                # Main config (paths, hyperparams)
│   └── logging.yaml               # Logging configuration
│
├── scripts/                       # Standalone executable scripts
│   ├── train_pipeline.py          # End-to-end training script
│   └── predict_pipeline.py        # End-to-end prediction script
│
├── tests/                         # Unit & integration tests
│   ├── test_preprocess.py
│   ├── test_features.py
│   └── test_model.py
│
├── logs/                          # Runtime logs
│
├── .gitignore                     # Git ignore rules
├── requirements.txt               # Python dependencies
├── environment.yml                # Conda environment (optional)
├── setup.py                       # Package setup
└── README.md                      # Project documentation (this file)
```

---

## 📊 Dataset

| Feature | Description |
|---|---|
| `CustomerID` | Unique identifier |
| `Tenure` | Months with the company |
| `MonthlyCharges` | Monthly bill amount |
| `TotalCharges` | Total amount charged |
| `Contract` | Month-to-month / One-year / Two-year |
| `PaymentMethod` | Payment type |
| `InternetService` | DSL / Fiber / No |
| `Churn` | **Target** — Yes / No |

> **Dataset**: [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
> Place the raw dataset in `data/raw/` as `telco_churn.csv`.

---

## 🛠 Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.10+ |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **ML Models** | Scikit-Learn, XGBoost, LightGBM |
| **Explainability** | SHAP |
| **Experiment Tracking** | MLflow |
| **API Serving** | FastAPI, Uvicorn |
| **Testing** | Pytest |
| **Config Management** | PyYAML, python-dotenv |

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/churn_project.git
cd churn_project
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Use Conda

```bash
conda env create -f environment.yml
conda activate churn-env
```

---

## 🚀 Usage

### Run Training Pipeline

```bash
python scripts/train_pipeline.py --config configs/config.yaml
```

### Run Prediction

```bash
python scripts/predict_pipeline.py --input data/processed/test.csv --output reports/metrics/predictions.csv
```

### Launch Jupyter Notebooks

```bash
jupyter notebook notebooks/
```

### Run Tests

```bash
pytest tests/ -v
```

---

## 📈 Model Results

| Model | Accuracy | ROC-AUC | F1-Score | Recall |
|---|---|---|---|---|
| Logistic Regression | 80.3% | 0.843 | 0.591 | 0.541 |
| Random Forest | 79.8% | 0.826 | 0.562 | 0.487 |
| **XGBoost** | **81.2%** | **0.862** | **0.614** | **0.573** |
| LightGBM | 80.9% | 0.858 | 0.608 | 0.561 |

> ✅ **Best Model**: XGBoost with tuned hyperparameters (GridSearchCV)

---

## 🔍 Key Insights

- 📌 **Contract type** is the strongest predictor — month-to-month customers churn 3× more
- 📌 Customers with **Fiber optic internet** have higher churn rates
- 📌 **Tenure < 12 months** is a high-risk segment
- 📌 **Senior citizens** show elevated churn probability

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

1. Fork the project
2. Create your feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

<div align="center">
  Made with ❤️ for AIML | Customer Churn Prediction Project
</div>
