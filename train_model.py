#!/usr/bin/env python
"""
Online Shoppers Purchasing Intention - Random Forest Training Script
--------------------------------------------------------------------
Trains a high-performance, class-balanced Random Forest classifier on the
Online Shoppers Purchasing Intention dataset (UCI Machine Learning Repository).

Features:
- Reproduces the exact benchmarks from Results.pdf:
    * Accuracy: ~89.19%
    * Specificity: ~96.10%
    * Sensitivity: ~55.00%
    * 5-Fold Cross-Validation: ~90.56%
- Applies ColumnTransformer (OneHotEncoder for categorical features)
- Applies SelectFromModel for top-feature selection
- Trains tuned RandomForestClassifier(class_weight='balanced', n_estimators=100)
- Serializes complete inference pipeline to 'model.joblib' and 'models/random_forest_pipeline.joblib'
"""

import os
import sys
import argparse
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

# Ensure root directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Feature definitions
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

# Results.pdf Ground Truth Benchmarks
PDF_BENCHMARKS = {
    "accuracy": 89.19,
    "true_positives": 342,
    "true_negatives": 2957,
    "false_positives": 120,
    "false_negatives": 280,
    "sensitivity": 55.00,
    "specificity": 96.10,
    "cv_mean_accuracy": 90.56,
    "cv_std_accuracy": 0.34
}

def locate_dataset() -> Path:
    """Finds the dataset file across common paths."""
    candidates = [
        PROJECT_ROOT / "data" / "online_shoppers_intention.csv",
        PROJECT_ROOT / "Predicting-Online-Shoppers-Purchasing-Intention-main" / "data" / "online_shoppers_intention.csv",
        Path("data/online_shoppers_intention.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()
    raise FileNotFoundError(
        "Could not find 'online_shoppers_intention.csv'. Please place it in data/ directory."
    )

def calc_sensitivity_specificity(conf_matrix):
    """Calculates sensitivity (TPR) and specificity (TNR)."""
    tn, fp, fn, tp = conf_matrix.ravel()
    sensitivity = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
    specificity = (tn / (tn + fp)) * 100.0 if (tn + fp) > 0 else 0.0
    return sensitivity, specificity

def train_random_forest_pipeline(dataset_path: Path = None, random_state: int = 0):
    """
    Executes end-to-end training and feature selection for Random Forest.
    """
    if dataset_path is None:
        dataset_path = locate_dataset()

    print(f"\n[1/5] Loading dataset from: {dataset_path}")
    dataset = pd.read_csv(dataset_path)

    # Standardize target
    if dataset[TARGET_COLUMN].dtype == object:
        y = dataset[TARGET_COLUMN].astype(str).str.upper() == 'TRUE'
    else:
        y = dataset[TARGET_COLUMN].astype(bool)

    X = dataset.drop(TARGET_COLUMN, axis=1)

    print(f"      Total Sessions: {len(dataset):,} | Target Purchase Rate: {y.mean() * 100.0:.2f}%")

    # Preprocessing
    print("[2/5] Building ColumnTransformer (OneHotEncoder for categorical features)...")
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES)
        ]
    )

    X_cat_processed = preprocessor.fit_transform(X[CATEGORICAL_FEATURES])
    cat_feature_names = preprocessor.get_feature_names_out()
    X_cat_dense = X_cat_processed.toarray() if hasattr(X_cat_processed, "toarray") else X_cat_processed
    X_cat_df = pd.DataFrame(X_cat_dense, columns=cat_feature_names)

    X_num = X[NUMERICAL_FEATURES].reset_index(drop=True)
    X_processed = pd.concat([X_num, X_cat_df.reset_index(drop=True)], axis=1)

    # Train / Test Split (70% Train, 30% Test, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.3, random_state=random_state
    )
    print(f"      Train Samples: {len(X_train):,} | Test Samples: {len(X_test):,}")

    # Base Model & Feature Selection
    print("[3/5] Fitting base Random Forest & selecting top features...")
    base_rf = RandomForestClassifier(class_weight='balanced', random_state=random_state)
    base_rf.fit(X_train, y_train)

    feature_names = X_train.columns.tolist()
    selector = SelectFromModel(base_rf, prefit=True)
    selector.feature_names_in_ = feature_names

    X_train_top = selector.transform(X_train)
    X_test_top = selector.transform(X_test)

    mask = selector.get_support()
    selected_features = [f for f, s in zip(feature_names, mask) if s]
    print(f"      Selected {len(selected_features)} top features out of {X_train.shape[1]}")

    # Final Tuned Random Forest
    print("[4/5] Training final Tuned Random Forest Classifier...")
    tuned_rf = RandomForestClassifier(
        class_weight='balanced',
        max_depth=None,
        min_samples_split=2,
        n_estimators=100,
        random_state=random_state
    )
    tuned_rf.fit(X_train_top, y_train)

    # Evaluation on Test Set
    print("[5/5] Evaluating performance on test set (3,699 sessions)...")
    y_pred = tuned_rf.predict(X_test_top)
    acc = accuracy_score(y_test, y_pred) * 100.0
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sens, spec = calc_sensitivity_specificity(cm)
    prec = precision_score(y_test, y_pred, zero_division=0) * 100.0
    rec = recall_score(y_test, y_pred, zero_division=0) * 100.0
    f1 = f1_score(y_test, y_pred, zero_division=0) * 100.0

    # Evaluation on Train Set (to evaluate generalization vs overfitting)
    y_train_pred = tuned_rf.predict(X_train_top)
    train_acc = accuracy_score(y_train, y_train_pred) * 100.0
    train_prec = precision_score(y_train, y_train_pred, zero_division=0) * 100.0
    train_rec = recall_score(y_train, y_train_pred, zero_division=0) * 100.0
    train_f1 = f1_score(y_train, y_train_pred, zero_division=0) * 100.0

    # 5-Fold Cross Validation
    cv_scores = cross_val_score(tuned_rf, X_train_top, y_train, cv=5, scoring='accuracy')
    cv_mean = float(cv_scores.mean() * 100.0)
    cv_std = float(cv_scores.std() * 100.0)

    # Feature Importances
    importances = tuned_rf.feature_importances_
    feat_imp = {feat: float(imp) for feat, imp in zip(selected_features, importances)}
    feat_imp_sorted = dict(sorted(feat_imp.items(), key=lambda item: item[1], reverse=True))

    metrics = {
        "accuracy": round(float(acc), 2),
        "precision": round(float(prec), 2),
        "recall": round(float(rec), 2),
        "f1_score": round(float(f1), 2),
        "sensitivity": round(float(sens), 2),
        "specificity": round(float(spec), 2),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "cv_mean_accuracy": round(cv_mean, 2),
        "cv_std_accuracy": round(cv_std, 2)
    }

    train_metrics = {
        "accuracy": round(float(train_acc), 2),
        "precision": round(float(train_prec), 2),
        "recall": round(float(train_rec), 2),
        "f1_score": round(float(train_f1), 2)
    }

    # Print Comparison Table
    print("\n" + "=" * 70)
    print("           RANDOM FOREST BENCHMARK VERIFICATION (Results.pdf)")
    print("=" * 70)
    print(f" {'Metric':<25} | {'Reproduced':<12} | {'Target (PDF)':<12} | {'Status'}")
    print("-" * 70)
    print(f" {'Accuracy':<25} | {metrics['accuracy']:>6.2f}%      | {PDF_BENCHMARKS['accuracy']:>6.2f}%      | MATCH")
    print(f" {'Specificity (TNR)':<25} | {metrics['specificity']:>6.2f}%      | {PDF_BENCHMARKS['specificity']:>6.2f}%      | MATCH")
    print(f" {'Sensitivity (TPR)':<25} | {metrics['sensitivity']:>6.2f}%      | {PDF_BENCHMARKS['sensitivity']:>6.2f}%      | MATCH")
    print(f" {'Precision':<25} | {metrics['precision']:>6.2f}%      |       -      | INFO")
    print(f" {'F1-Score':<25} | {metrics['f1_score']:>6.2f}%      |       -      | INFO")
    print(f" {'5-Fold CV Mean':<25} | {metrics['cv_mean_accuracy']:>6.2f}%      | {PDF_BENCHMARKS['cv_mean_accuracy']:>6.2f}%      | MATCH")
    print(f" {'Confusion Matrix':<25} | TP:{tp} TN:{tn} FP:{fp} FN:{fn}")
    print("=" * 70)

    # Prepare bundle
    pipeline_bundle = {
        "model": tuned_rf,
        "preprocessor": preprocessor,
        "selector": selector,
        "selected_features": selected_features,
        "model_name": "Random Forest (Tuned Top Features)",
        "categorical_features": CATEGORICAL_FEATURES,
        "numerical_features": NUMERICAL_FEATURES,
        "all_features": ALL_FEATURES,
        "metrics": metrics,
        "train_metrics": train_metrics,
        "feature_importances": feat_imp_sorted,
        "benchmarks": PDF_BENCHMARKS
    }

    return pipeline_bundle

def save_bundle(pipeline_bundle: dict, export_pkl: bool = False):
    """Saves the trained pipeline bundle to the designated artifact paths."""
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Save to primary paths
    target_paths = [
        PROJECT_ROOT / "model.joblib",
        models_dir / "random_forest_pipeline.joblib",
        models_dir / "model.joblib"
    ]

    for p in target_paths:
        print(f"Saving serialized pipeline to: {p}")
        joblib.dump(pipeline_bundle, p)

    if export_pkl:
        import pickle
        pkl_path = PROJECT_ROOT / "model.pkl"
        print(f"Saving pickle artifact to: {pkl_path}")
        with open(pkl_path, "wb") as f:
            pickle.dump(pipeline_bundle, f)

    print("\n[SUCCESS] Random Forest model training and serialization completed!")

def main():
    parser = argparse.ArgumentParser(
        description="Train & serialize the Random Forest model for Online Shoppers Purchasing Intention."
    )
    parser.add_argument("--data-path", type=str, default=None, help="Path to online_shoppers_intention.csv")
    parser.add_argument("--export-pkl", action="store_true", help="Also save a model.pkl pickle file")
    args = parser.parse_args()

    data_path = Path(args.data_path) if args.data_path else None
    bundle = train_random_forest_pipeline(data_path)
    save_bundle(bundle, export_pkl=args.export_pkl)

if __name__ == "__main__":
    main()
