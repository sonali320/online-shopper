#!/usr/bin/env python
"""
Reproduce Results.pdf Benchmark Script
---------------------------------------
Runs all three tuned machine learning models (Logistic Regression, Random Forest, XGBoost)
and reproduces the exact benchmarks documented in Results.pdf:
  - Accuracy
  - Confusion Matrix (TP, TN, FP, FN)
  - Sensitivity (True Positive Rate)
  - Specificity (True Negative Rate)
  - 5-Fold Cross-Validation Mean Accuracy & Standard Deviation
"""

import sys
import os
from pathlib import Path

# Ensure workspace root and src/ are in sys.path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from src.train import train_and_evaluate_all
from src.config import PDF_BENCHMARKS

def print_separator(char="=", length=85):
    print(char * length)

def main():
    print_separator("=")
    print(" PREDICTING ONLINE SHOPPERS PURCHASING INTENTION - REPRODUCING RESULTS.PDF")
    print_separator("=")
    print("Dataset: UCI Online Shoppers Purchasing Intention (12,330 sessions, 18 columns)")
    print("Split: 70% Train, 30% Test (Test Size: 3,699 samples, Random State = 0)")
    print_separator("-")

    # Run full training and benchmark comparison
    results = train_and_evaluate_all(save_models=True, generate_report=True)

    print("\n" + "=" * 85)
    print("                        BENCHMARK COMPARISON TABLE")
    print("=" * 85)
    print(f"{'Model & Metric':<35} | {'Reproduced':<12} | {'Target (PDF)':<12} | {'Diff':<8} | {'Status'}")
    print("-" * 85)

    for model_name, res in results.items():
        print(f"\n--- {model_name} (Top Features Tuned) ---")
        comp = res["comparison"]
        for key, item in comp.items():
            label = item["label"]
            rep = item["reproduced"]
            tgt = item["pdf_benchmark"]
            diff = item["difference"]
            status = item["status"]

            if "accuracy" in key or "sensitivity" in key or "specificity" in key:
                rep_str = f"{rep:6.2f}%"
                tgt_str = f"{tgt:6.2f}%"
                diff_str = f"{diff:+5.2f}%" if diff is not None else "  -  "
            else:
                rep_str = f"{rep:>6}"
                tgt_str = f"{tgt:>6}"
                diff_str = f"{diff:>+5}" if diff is not None else "  -  "

            badge = "[MATCH]" if status == "PASS" else "[DIFF]"
            print(f"  {label:<33} | {rep_str:<12} | {tgt_str:<12} | {diff_str:<8} | {badge}")

    print_separator("=")
    print(" KEY TAKEAWAYS & BUSINESS SUMMARY (FROM RESULTS.PDF):")
    print_separator("-")
    print(" 1. Accuracy: Balanced Random Forest has highest overall accuracy (89.19%),")
    print("    closely followed by Logistic Regression (87.08%) and XGBoost (87.00%).")
    print(" 2. Sensitivity: XGBoost leads in capturing actual positive purchases (80.23%),")
    print("    followed by Logistic Regression (79.26%) and Random Forest (55.00%).")
    print(" 3. Specificity: Random Forest excels at correctly identifying non-purchases (96.10%),")
    print("    minimizing false alarms (only 120 false positives vs 349 for LR and 358 for XGB).")
    print(" 4. Cross-Validation Stability: Random Forest shows highest mean CV accuracy (90.56%)")
    print("    with lowest standard deviation (0.34%), proving superior generalizability.")
    print(" 5. Conclusion: Use Random Forest for overall accuracy and specificity. Use XGBoost")
    print("    or Logistic Regression when maximizing captured purchasers is top priority.")
    print_separator("=")
    print("\nReports successfully written to:")
    print(" - reports/pdf_benchmark_comparison.md")
    print(" - reports/pdf_benchmark_comparison.json")
    print("Serialized models saved to:")
    print(" - models/logistic_regression_pipeline.joblib")
    print(" - models/random_forest_pipeline.joblib")
    print(" - models/xgboost_pipeline.joblib")
    print_separator("=")

if __name__ == "__main__":
    main()
