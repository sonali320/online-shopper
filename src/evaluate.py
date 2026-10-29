"""
Evaluation Module for Online Shoppers Purchasing Intention.
Calculates Accuracy, Sensitivity, Specificity, Confusion Matrix, Cross-Validation,
and formats comparison reports against Results.pdf benchmark.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score

from src.config import CV_FOLDS

def calculate_sensitivity_specificity(conf_matrix: np.ndarray) -> Tuple[float, float]:
    """
    Computes Sensitivity (TPR) and Specificity (TNR) from 2x2 confusion matrix:
    conf_matrix = [[TN, FP],
                   [FN, TP]]
    """
    tn, fp, fn, tp = conf_matrix.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return float(sensitivity), float(specificity)

def evaluate_model_performance(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    perform_cv: bool = True
) -> Dict[str, Any]:
    """
    Evaluates model test performance and 5-fold cross-validation accuracy.
    """
    y_pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    sens, spec = calculate_sensitivity_specificity(cm)
    prec = float(precision_score(y_test, y_pred, zero_division=0)) * 100.0
    rec = float(recall_score(y_test, y_pred, zero_division=0)) * 100.0
    f1 = float(f1_score(y_test, y_pred, zero_division=0)) * 100.0

    cv_mean = None
    cv_std = None
    if perform_cv:
        cv_scores = cross_val_score(model, X_train, y_train, cv=CV_FOLDS, scoring='accuracy')
        cv_mean = float(cv_scores.mean())
        cv_std = float(cv_scores.std())

    return {
        "accuracy": acc * 100.0,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "sensitivity": sens * 100.0,
        "specificity": spec * 100.0,
        "cv_mean_accuracy": (cv_mean * 100.0) if cv_mean is not None else None,
        "cv_std_accuracy": (cv_std * 100.0) if cv_std is not None else None,
        "confusion_matrix": cm.tolist()
    }

def compare_with_pdf_benchmark(
    reproduced: Dict[str, Any],
    benchmark: Dict[str, Any],
    tolerance_pct: float = 1.0
) -> Dict[str, Any]:
    """
    Compares live reproduced metrics against the ground-truth benchmark from Results.pdf.
    """
    comparison = {}
    metrics_to_check = [
        ("accuracy", "Accuracy (%)"),
        ("true_positives", "True Positives"),
        ("true_negatives", "True Negatives"),
        ("false_positives", "False Positives"),
        ("false_negatives", "False Negatives"),
        ("sensitivity", "Sensitivity (%)"),
        ("specificity", "Specificity (%)"),
        ("cv_mean_accuracy", "CV Mean Accuracy (%)"),
        ("cv_std_accuracy", "CV Std Dev (%)")
    ]

    for key, label in metrics_to_check:
        rep_val = reproduced.get(key)
        bm_val = benchmark.get(key)
        diff = (rep_val - bm_val) if (rep_val is not None and bm_val is not None) else None
        
        # Consider a match if within tolerance or exact integer match
        if isinstance(bm_val, int):
            match = (rep_val == bm_val)
        elif bm_val is not None and rep_val is not None:
            match = abs(diff) <= tolerance_pct
        else:
            match = False

        comparison[key] = {
            "label": label,
            "reproduced": rep_val,
            "pdf_benchmark": bm_val,
            "difference": diff,
            "status": "PASS" if match else "DIFF"
        }

    return comparison
