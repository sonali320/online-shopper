"""
Inference Module for Online Shoppers Purchasing Intention.
Supports single-session interactive inference and batch inference on DataFrames.
"""

from typing import Dict, Any, Union
import joblib
import pandas as pd
from pathlib import Path

from src.config import MODELS_DIR, PROJECT_ROOT, ALL_FEATURES, CATEGORICAL_FEATURES, NUMERICAL_FEATURES

_MODEL_CACHE: Dict[str, Any] = {}

def get_pipeline_path(model_name: str) -> Path:
    normalized = model_name.lower().replace(" ", "_").replace("-", "_")
    if "logistic" in normalized:
        return MODELS_DIR / "logistic_regression_pipeline.joblib"
    elif "xgb" in normalized:
        return MODELS_DIR / "xgboost_pipeline.joblib"
    else:
        # Default to Random Forest - check multiple standard locations
        candidates = [
            MODELS_DIR / "random_forest_pipeline.joblib",
            PROJECT_ROOT / "model.joblib",
            MODELS_DIR / "model.joblib",
            PROJECT_ROOT / "model.pkl",
            MODELS_DIR / "model.pkl"
        ]
        for c in candidates:
            if c.exists():
                return c
        return MODELS_DIR / "random_forest_pipeline.joblib"

def load_pipeline(model_name: str = "Random Forest") -> Dict[str, Any]:
    """Loads and caches the model artifact pipeline."""
    path = get_pipeline_path(model_name)
    if str(path) not in _MODEL_CACHE:
        if not path.exists():
            raise FileNotFoundError(
                f"Model pipeline not found at {path}. Please run 'python train_model.py' first."
            )
        _MODEL_CACHE[str(path)] = joblib.load(path)
    return _MODEL_CACHE[str(path)]

def predict_single(
    session_data: Dict[str, Any],
    model_name: str = "Random Forest"
) -> Dict[str, Any]:
    """
    Predicts purchasing intention for a single shopper session.
    """
    pipeline = load_pipeline(model_name)
    preprocessor = pipeline["preprocessor"]
    selector = pipeline["selector"]
    model = pipeline["model"]
    selected_features = pipeline["selected_features"]

    # Convert dictionary to single-row DataFrame
    df_raw = pd.DataFrame([session_data])

    # Preprocess according to model architecture
    if "logistic" in model_name.lower():
        processed_arr = preprocessor.transform(df_raw)
        feature_names = preprocessor.get_feature_names_out()
        dense_arr = processed_arr.toarray() if hasattr(processed_arr, "toarray") else processed_arr
        df_processed = pd.DataFrame(dense_arr, columns=feature_names)
    else:
        cat_arr = preprocessor.transform(df_raw[CATEGORICAL_FEATURES])
        cat_names = preprocessor.get_feature_names_out()
        cat_dense = cat_arr.toarray() if hasattr(cat_arr, "toarray") else cat_arr
        df_cat = pd.DataFrame(cat_dense, columns=cat_names)
        df_num = df_raw[NUMERICAL_FEATURES].reset_index(drop=True)
        df_processed = pd.concat([df_num, df_cat], axis=1)

    # Feature selection
    top_arr = selector.transform(df_processed)
    top_df = pd.DataFrame(top_arr, columns=selected_features)

    # Prediction & probability
    pred = bool(model.predict(top_df)[0])
    prob = 0.5
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(top_df)[0][1])

    return {
        "model_name": pipeline["model_name"],
        "purchase_prediction": pred,
        "purchase_probability": prob,
        "prediction_label": "Purchase (Revenue = TRUE)" if pred else "No Purchase (Revenue = FALSE)",
        "confidence_score": round(prob * 100.0, 2) if pred else round((1.0 - prob) * 100.0, 2)
    }

def predict_batch(
    df: pd.DataFrame,
    model_name: str = "Random Forest"
) -> pd.DataFrame:
    """
    Adds prediction columns to a batch DataFrame of user sessions.
    """
    pipeline = load_pipeline(model_name)
    preprocessor = pipeline["preprocessor"]
    selector = pipeline["selector"]
    model = pipeline["model"]
    selected_features = pipeline["selected_features"]

    df_features = df[[c for c in ALL_FEATURES if c in df.columns]].copy()

    if "logistic" in model_name.lower():
        processed_arr = preprocessor.transform(df_features)
        feature_names = preprocessor.get_feature_names_out()
        dense_arr = processed_arr.toarray() if hasattr(processed_arr, "toarray") else processed_arr
        df_processed = pd.DataFrame(dense_arr, columns=feature_names)
    else:
        cat_arr = preprocessor.transform(df_features[CATEGORICAL_FEATURES])
        cat_names = preprocessor.get_feature_names_out()
        cat_dense = cat_arr.toarray() if hasattr(cat_arr, "toarray") else cat_arr
        df_cat = pd.DataFrame(cat_dense, columns=cat_names)
        df_num = df_features[NUMERICAL_FEATURES].reset_index(drop=True)
        df_processed = pd.concat([df_num, df_cat], axis=1)

    top_arr = selector.transform(df_processed)
    top_df = pd.DataFrame(top_arr, columns=selected_features)

    preds = model.predict(top_df)
    probs = model.predict_proba(top_df)[:, 1] if hasattr(model, "predict_proba") else [0.5] * len(preds)

    df_out = df.copy()
    df_out["Predicted_Revenue"] = preds.astype(bool)
    df_out["Purchase_Probability"] = probs
    return df_out
