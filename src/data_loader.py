"""
Data Loader Module for Online Shoppers Purchasing Intention.
Handles dataset loading, validation, directory sync, and exploratory summaries.
"""

from pathlib import Path
import shutil
import pandas as pd
from typing import Tuple, Dict, Any

from src.config import (
    DATA_PATH,
    PROJECT_ROOT,
    ALL_FEATURES,
    TARGET_COLUMN,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES
)

def ensure_dataset_available() -> Path:
    """
    Ensures that online_shoppers_intention.csv is located in the root data/ folder.
    Copies from subfolder if missing.
    """
    target_data_dir = PROJECT_ROOT / "data"
    target_data_file = target_data_dir / "online_shoppers_intention.csv"

    if target_data_file.exists():
        return target_data_file

    target_data_dir.mkdir(parents=True, exist_ok=True)

    # Check fallback subfolder
    subfolder_file = (
        PROJECT_ROOT
        / "Predicting-Online-Shoppers-Purchasing-Intention-main"
        / "data"
        / "online_shoppers_intention.csv"
    )
    if subfolder_file.exists():
        shutil.copyfile(subfolder_file, target_data_file)
        return target_data_file

    if DATA_PATH.exists():
        shutil.copyfile(DATA_PATH, target_data_file)
        return target_data_file

    raise FileNotFoundError(
        f"Could not locate 'online_shoppers_intention.csv' at {target_data_file} "
        f"or at {subfolder_file}."
    )

def load_data(path: Path = None) -> pd.DataFrame:
    """
    Loads and validates the Online Shoppers Purchasing Intention dataset.
    """
    if path is None:
        path = ensure_dataset_available()

    df = pd.read_csv(path)

    # Validate required columns
    missing_cols = [c for c in ALL_FEATURES + [TARGET_COLUMN] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    # Standardize Revenue target to boolean or 0/1 if needed
    if df[TARGET_COLUMN].dtype == object:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str).str.upper() == 'TRUE'

    return df

def get_feature_target_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Splits dataset into feature matrix X and target series y.
    """
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y

def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Returns statistical summaries matching the original exploratory data analysis.
    """
    dataset_num = df[NUMERICAL_FEATURES]
    return {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "class_distribution": df[TARGET_COLUMN].value_counts(normalize=True).to_dict(),
        "numerical_skewness": dataset_num.skew().to_dict(),
        "numerical_correlation": dataset_num.corr().to_dict()
    }
