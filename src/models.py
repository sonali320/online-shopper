"""
Models Module for Online Shoppers Purchasing Intention.
Builds base and tuned estimators for Logistic Regression, Random Forest, and XGBoost,
configured exactly as specified in the research paper / Results.pdf benchmark.
"""

from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.config import (
    BEST_PARAMS_LOGREG,
    BEST_PARAMS_RF,
    BEST_PARAMS_XGB,
    RANDOM_STATE
)

def create_base_logistic_regression() -> LogisticRegression:
    """Creates base balanced Logistic Regression model for feature selection."""
    return LogisticRegression(
        max_iter=10000,
        class_weight='balanced',
        random_state=RANDOM_STATE
    )

def create_tuned_logistic_regression(custom_params: Dict[str, Any] = None) -> LogisticRegression:
    """Creates final tuned Logistic Regression model (C=0.01)."""
    params = dict(BEST_PARAMS_LOGREG)
    if custom_params:
        params.update(custom_params)
    return LogisticRegression(**params)

def create_base_random_forest() -> RandomForestClassifier:
    """Creates base balanced Random Forest model for feature selection."""
    return RandomForestClassifier(
        class_weight='balanced',
        random_state=RANDOM_STATE
    )

def create_tuned_random_forest(custom_params: Dict[str, Any] = None) -> RandomForestClassifier:
    """Creates final tuned Random Forest model (n_estimators=100, min_samples_split=2)."""
    params = dict(BEST_PARAMS_RF)
    if custom_params:
        params.update(custom_params)
    return RandomForestClassifier(**params)

def calculate_scale_pos_weight(y_train) -> float:
    """Calculates negative-to-positive ratio for class balancing in XGBoost."""
    counts = y_train.value_counts()
    neg_count = counts.get(0, counts.get(False, 1))
    pos_count = counts.get(1, counts.get(True, 1))
    return float(neg_count / pos_count)

def create_base_xgboost(scale_pos_weight: float) -> XGBClassifier:
    """Creates base balanced XGBoost model for feature selection."""
    return XGBClassifier(
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE
    )

def create_tuned_xgboost(scale_pos_weight: float, custom_params: Dict[str, Any] = None) -> XGBClassifier:
    """Creates final tuned XGBoost model matching parameters from Results.pdf."""
    params = dict(BEST_PARAMS_XGB)
    params['scale_pos_weight'] = scale_pos_weight
    if custom_params:
        params.update(custom_params)
    return XGBClassifier(**params)

def create_decision_tree(custom_params: Dict[str, Any] = None) -> DecisionTreeClassifier:
    """Creates Decision Tree classifier using identical class weighting and random_state."""
    params = {
        'class_weight': 'balanced',
        'random_state': RANDOM_STATE
    }
    if custom_params:
        params.update(custom_params)
    return DecisionTreeClassifier(**params)
