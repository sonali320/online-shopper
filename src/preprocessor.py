"""
Preprocessing Module for Online Shoppers Purchasing Intention.
Builds feature transformers for Logistic Regression (PowerTransformer + OneHotEncoder)
and Tree-Based Models (Raw Numerical + OneHotEncoder).
"""

from typing import Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from sklearn.model_selection import train_test_split

from src.config import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TEST_SIZE,
    RANDOM_STATE
)

def build_logistic_regression_preprocessor() -> ColumnTransformer:
    """
    Builds the ColumnTransformer for Logistic Regression:
    - Numerical: PowerTransformer (Yeo-Johnson to resolve skewness)
    - Categorical: OneHotEncoder
    """
    return ColumnTransformer(
        transformers=[
            ('num', PowerTransformer(), NUMERICAL_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES)
        ]
    )

def prepare_logistic_regression_data(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: ColumnTransformer = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer]:
    """
    Fits preprocessor on X, splits into train/test, and returns named DataFrames.
    """
    if preprocessor is None:
        preprocessor = build_logistic_regression_preprocessor()

    X_processed = preprocessor.fit_transform(X)
    X_train_p, X_test_p, y_train, y_test = train_test_split(
        X_processed, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    feature_names = preprocessor.get_feature_names_out()
    # Handle sparse or dense array
    X_train_arr = X_train_p.toarray() if hasattr(X_train_p, "toarray") else X_train_p
    X_test_arr = X_test_p.toarray() if hasattr(X_test_p, "toarray") else X_test_p

    X_train = pd.DataFrame(X_train_arr, columns=feature_names)
    X_test = pd.DataFrame(X_test_arr, columns=feature_names)

    return X_train, X_test, y_train, y_test, preprocessor

def build_tree_preprocessor() -> ColumnTransformer:
    """
    Builds the ColumnTransformer for Tree models (Random Forest / XGBoost):
    - Categorical: OneHotEncoder
    """
    return ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES)
        ]
    )

def prepare_tree_data(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: ColumnTransformer = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer]:
    """
    Encodes categorical features and concatenates with raw numerical features.
    """
    if preprocessor is None:
        preprocessor = build_tree_preprocessor()

    X_cat_processed = preprocessor.fit_transform(X[CATEGORICAL_FEATURES])
    feature_names_cat = preprocessor.get_feature_names_out()

    X_cat_arr = X_cat_processed.toarray() if hasattr(X_cat_processed, "toarray") else X_cat_processed
    X_cat_df = pd.DataFrame(X_cat_arr, columns=feature_names_cat)

    X_num = X[NUMERICAL_FEATURES].reset_index(drop=True)
    X_processed = pd.concat([X_num, X_cat_df.reset_index(drop=True)], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    return X_train, X_test, y_train, y_test, preprocessor
