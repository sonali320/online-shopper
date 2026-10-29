"""
Feature Selection Module for Online Shoppers Purchasing Intention.
Uses SelectFromModel on pre-fitted balanced base estimators to select top features.
"""

from typing import Tuple, List
import pandas as pd
from sklearn.feature_selection import SelectFromModel

def select_top_features(
    estimator,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, SelectFromModel, List[str]]:
    """
    Applies SelectFromModel on prefit estimator.
    Returns transformed train and test sets, the selector instance, and selected feature names.
    """
    selector = SelectFromModel(estimator, prefit=True)
    if hasattr(selector, "feature_names_in_"):
        selector.feature_names_in_ = X_train.columns.tolist()

    X_train_top = selector.transform(X_train)
    X_test_top = selector.transform(X_test)

    # Determine selected feature names
    support_mask = selector.get_support()
    selected_feature_names = [
        col for col, selected in zip(X_train.columns, support_mask) if selected
    ]

    # Convert back to DataFrame preserving column names for clarity
    X_train_top_df = pd.DataFrame(X_train_top, columns=selected_feature_names)
    X_test_top_df = pd.DataFrame(X_test_top, columns=selected_feature_names)

    return X_train_top_df, X_test_top_df, selector, selected_feature_names
