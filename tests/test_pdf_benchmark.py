"""
Tests Asserting Alignment with Results.pdf Benchmarks.
"""

import pytest
from src.data_loader import load_data
from src.train import train_logistic_regression, train_random_forest, train_xgboost
from src.config import PDF_BENCHMARKS

@pytest.fixture(scope="module")
def dataset():
    return load_data()

def test_random_forest_matches_pdf_benchmark(dataset):
    res = train_random_forest(dataset)
    metrics = res["metrics"]
    target = PDF_BENCHMARKS["Random Forest"]

    # Target: Accuracy 89.19%
    assert abs(metrics["accuracy"] - target["accuracy"]) < 1.5
    # Target: Specificity 96.10%
    assert abs(metrics["specificity"] - target["specificity"]) < 1.5
    # Target: Sensitivity 55.00%
    assert abs(metrics["sensitivity"] - target["sensitivity"]) < 2.0

def test_logistic_regression_matches_pdf_benchmark(dataset):
    res = train_logistic_regression(dataset)
    metrics = res["metrics"]
    target = PDF_BENCHMARKS["Logistic Regression"]

    # Target: Accuracy 87.08%
    assert abs(metrics["accuracy"] - target["accuracy"]) < 1.5
    # Target: Sensitivity 79.26%
    assert abs(metrics["sensitivity"] - target["sensitivity"]) < 1.5
    # Target: Specificity 88.66%
    assert abs(metrics["specificity"] - target["specificity"]) < 1.5

def test_xgboost_matches_pdf_benchmark(dataset):
    res = train_xgboost(dataset)
    metrics = res["metrics"]
    target = PDF_BENCHMARKS["XGBoost"]

    # Target: Accuracy 87.00%
    assert abs(metrics["accuracy"] - target["accuracy"]) < 1.5
    # Target: Sensitivity 80.23%
    assert abs(metrics["sensitivity"] - target["sensitivity"]) < 1.5
    # Target: Specificity 88.37%
    assert abs(metrics["specificity"] - target["specificity"]) < 1.5
