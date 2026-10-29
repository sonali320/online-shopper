"""
Configuration Module for Online Shoppers Purchasing Intention Project.
Handles directory resolution, environment variables, features definitions,
and target benchmark metrics from Results.pdf.
"""

from pathlib import Path
import os

# Project Root resolution
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

# Attempt loading from .env if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

# Feature definitions from dataset
CATEGORICAL_FEATURES = [
    'Month',
    'OperatingSystems',
    'Browser',
    'Region',
    'TrafficType',
    'VisitorType',
    'Weekend'
]

NUMERICAL_FEATURES = [
    'Administrative',
    'Administrative_Duration',
    'Informational',
    'Informational_Duration',
    'ProductRelated',
    'ProductRelated_Duration',
    'BounceRates',
    'ExitRates',
    'PageValues',
    'SpecialDay'
]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = 'Revenue'

# Data Paths
def resolve_data_path() -> Path:
    env_path = os.getenv("DATA_PATH", "data/online_shoppers_intention.csv")
    candidates = [
        PROJECT_ROOT / env_path,
        PROJECT_ROOT / "Predicting-Online-Shoppers-Purchasing-Intention-main" / "data" / "online_shoppers_intention.csv",
        PROJECT_ROOT / "data" / "online_shoppers_intention.csv",
        Path.cwd() / "data" / "online_shoppers_intention.csv",
        Path.cwd() / "Predicting-Online-Shoppers-Purchasing-Intention-main" / "data" / "online_shoppers_intention.csv",
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()
    # Default fallback
    return (PROJECT_ROOT / "data" / "online_shoppers_intention.csv").resolve()

DATA_PATH = resolve_data_path()
MODELS_DIR = (PROJECT_ROOT / os.getenv("MODELS_DIR", "models")).resolve()
REPORTS_DIR = (PROJECT_ROOT / os.getenv("REPORTS_DIR", "reports")).resolve()

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset Split and Cross Validation
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.3"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "0"))
CV_FOLDS = int(os.getenv("CV_FOLDS", "5"))
RUN_FULL_GRID_SEARCH = os.getenv("RUN_FULL_GRID_SEARCH", "false").strip().lower() in ("true", "1", "yes")

# Web Server
WEB_HOST = os.getenv("WEB_HOST", "127.0.0.1")
WEB_PORT = int(os.getenv("WEB_PORT", "5000"))

# Optimal Hyperparameters as determined and recorded in Results.pdf
BEST_PARAMS_LOGREG = {
    'C': 0.01,
    'class_weight': 'balanced',
    'max_iter': 10000,
    'random_state': RANDOM_STATE
}

BEST_PARAMS_RF = {
    'class_weight': 'balanced',
    'max_depth': None,
    'min_samples_split': 2,
    'n_estimators': 100,
    'random_state': RANDOM_STATE
}

BEST_PARAMS_XGB = {
    'eval_metric': 'logloss',
    'colsample_bytree': 0.9,
    'learning_rate': 0.01,
    'max_depth': 5,
    'n_estimators': 300,
    'reg_alpha': 0.5,
    'reg_lambda': 1.5,
    'subsample': 0.7,
    'random_state': RANDOM_STATE
}

# Ground Truth Benchmark Targets from Results.pdf
PDF_BENCHMARKS = {
    "Logistic Regression": {
        "model_name": "Logistic Regression (Top Features Tuned)",
        "accuracy": 87.08,
        "true_positives": 493,
        "true_negatives": 2728,
        "false_positives": 349,
        "false_negatives": 129,
        "sensitivity": 79.26,
        "specificity": 88.66,
        "cv_mean_accuracy": 87.60,
        "cv_std_accuracy": 0.52
    },
    "Random Forest": {
        "model_name": "Random Forest (Top Features Tuned)",
        "accuracy": 89.19,
        "true_positives": 342,
        "true_negatives": 2957,
        "false_positives": 120,
        "false_negatives": 280,
        "sensitivity": 55.00,
        "specificity": 96.10,
        "cv_mean_accuracy": 90.56,
        "cv_std_accuracy": 0.34
    },
    "XGBoost": {
        "model_name": "XGBoost (Top Features Tuned)",
        "accuracy": 87.00,
        "true_positives": 499,
        "true_negatives": 2719,
        "false_positives": 358,
        "false_negatives": 123,
        "sensitivity": 80.23,
        "specificity": 88.37,
        "cv_mean_accuracy": 87.70,
        "cv_std_accuracy": 0.58
    }
}
