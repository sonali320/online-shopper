"""
Tests for Data Loading and Schema Integrity.
"""

import pytest
import pandas as pd
from src.data_loader import ensure_dataset_available, load_data, get_feature_target_split
from src.config import ALL_FEATURES, TARGET_COLUMN

def test_dataset_file_exists():
    path = ensure_dataset_available()
    assert path.exists(), f"Dataset file does not exist at {path}"

def test_load_data_schema():
    df = load_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 12330, f"Expected 12,330 rows, got {len(df)}"

    for col in ALL_FEATURES + [TARGET_COLUMN]:
        assert col in df.columns, f"Missing expected column: {col}"

def test_missing_values_absent():
    df = load_data()
    null_counts = df.isnull().sum()
    assert null_counts.sum() == 0, f"Dataset contains unexpected nulls: {null_counts[null_counts > 0]}"

def test_feature_target_split():
    df = load_data()
    X, y = get_feature_target_split(df)
    assert TARGET_COLUMN not in X.columns
    assert len(X) == len(y) == 12330
    assert y.nunique() == 2
