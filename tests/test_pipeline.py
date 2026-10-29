"""
Tests for Data Preprocessing and Feature Selection Pipelines.
"""

import pytest
import pandas as pd
from src.data_loader import load_data, get_feature_target_split
from src.preprocessor import prepare_logistic_regression_data, prepare_tree_data
from src.models import create_base_logistic_regression, create_base_random_forest
from src.feature_selection import select_top_features

@pytest.fixture(scope="module")
def sample_dataset():
    df = load_data()
    return get_feature_target_split(df)

def test_logistic_regression_preprocessor(sample_dataset):
    X, y = sample_dataset
    X_train, X_test, y_train, y_test, preprocessor = prepare_logistic_regression_data(X, y)
    assert len(X_train) == 8631
    assert len(X_test) == 3699
    assert X_train.shape[1] > 20, "Expected one-hot encoded columns"
    assert not X_train.isnull().values.any()

def test_tree_preprocessor(sample_dataset):
    X, y = sample_dataset
    X_train, X_test, y_train, y_test, preprocessor = prepare_tree_data(X, y)
    assert len(X_train) == 8631
    assert len(X_test) == 3699
    assert X_train.shape[1] > 20
    assert not X_train.isnull().values.any()

def test_select_top_features(sample_dataset):
    X, y = sample_dataset
    X_train, X_test, y_train, y_test, _ = prepare_tree_data(X, y)
    base_rf = create_base_random_forest()
    base_rf.fit(X_train, y_train)

    X_train_top, X_test_top, selector, top_features = select_top_features(base_rf, X_train, X_test)
    assert len(top_features) > 0
    assert len(top_features) < X_train.shape[1], "Feature selection should reduce feature count"
    assert X_train_top.shape[1] == len(top_features)
