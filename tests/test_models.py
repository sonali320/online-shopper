"""
Tests for Model Builders and Predictions.
"""

import pytest
from src.models import (
    create_tuned_logistic_regression,
    create_tuned_random_forest,
    create_tuned_xgboost
)

def test_model_initialization():
    lr = create_tuned_logistic_regression()
    assert lr.C == 0.01
    assert lr.class_weight == 'balanced'

    rf = create_tuned_random_forest()
    assert rf.n_estimators == 100
    assert rf.class_weight == 'balanced'

    xgb = create_tuned_xgboost(scale_pos_weight=5.4)
    assert xgb.n_estimators == 300
    assert xgb.learning_rate == 0.01
    assert xgb.max_depth == 5
