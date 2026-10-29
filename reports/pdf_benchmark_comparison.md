# Model Performance & Results.pdf Benchmark Comparison

This report benchmarks the newly trained models against the ground-truth results documented in `Results.pdf`.

| Model | Metric | Reproduced | Target (PDF) | Difference | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Accuracy (%) | 87.10% | 87.08% | +0.02 | PASS |
| **Logistic Regression** | True Positives | 493 | 493 | +0 | PASS |
| **Logistic Regression** | True Negatives | 2729 | 2728 | +1 | DIFF |
| **Logistic Regression** | False Positives | 348 | 349 | -1 | DIFF |
| **Logistic Regression** | False Negatives | 129 | 129 | +0 | PASS |
| **Logistic Regression** | Sensitivity (%) | 79.26% | 79.26% | +0.00 | PASS |
| **Logistic Regression** | Specificity (%) | 88.69% | 88.66% | +0.03 | PASS |
| **Logistic Regression** | CV Mean Accuracy (%) | 87.60% | 87.60% | +0.00 | PASS |
| **Logistic Regression** | CV Std Dev (%) | 0.52% | 0.52% | +0.00 | PASS |
| **Random Forest** | Accuracy (%) | 88.59% | 89.19% | -0.60 | PASS |
| **Random Forest** | True Positives | 450 | 342 | +108 | DIFF |
| **Random Forest** | True Negatives | 2827 | 2957 | -130 | DIFF |
| **Random Forest** | False Positives | 250 | 120 | +130 | DIFF |
| **Random Forest** | False Negatives | 172 | 280 | -108 | DIFF |
| **Random Forest** | Sensitivity (%) | 72.35% | 55.00% | +17.35 | DIFF |
| **Random Forest** | Specificity (%) | 91.88% | 96.10% | -4.22 | DIFF |
| **Random Forest** | CV Mean Accuracy (%) | 89.14% | 90.56% | -1.42 | DIFF |
| **Random Forest** | CV Std Dev (%) | 0.56% | 0.34% | +0.22 | PASS |
| **XGBoost** | Accuracy (%) | 85.51% | 87.00% | -1.49 | DIFF |
| **XGBoost** | True Positives | 523 | 499 | +24 | DIFF |
| **XGBoost** | True Negatives | 2640 | 2719 | -79 | DIFF |
| **XGBoost** | False Positives | 437 | 358 | +79 | DIFF |
| **XGBoost** | False Negatives | 99 | 123 | -24 | DIFF |
| **XGBoost** | Sensitivity (%) | 84.08% | 80.23% | +3.85 | DIFF |
| **XGBoost** | Specificity (%) | 85.80% | 88.37% | -2.57 | DIFF |
| **XGBoost** | CV Mean Accuracy (%) | 86.54% | 87.70% | -1.16 | DIFF |
| **XGBoost** | CV Std Dev (%) | 0.46% | 0.58% | -0.12 | PASS |

## Key Observations & Insights from Results.pdf

- **Accuracy**: Balanced Random Forest achieves highest overall accuracy (89.19%), followed by Logistic Regression (87.08%) and XGBoost (87.00%).
- **Sensitivity (Capture Rate)**: XGBoost leads in capturing actual purchasers (80.23%), followed closely by Logistic Regression (79.26%). Random Forest achieves 55.00%.
- **Specificity**: Random Forest excels at correctly identifying non-purchasers (96.10%), with minimal false positives (120 vs 349/358).
- **Cross-Validation Stability**: Random Forest exhibits the lowest standard deviation (0.34%) across 5 folds, confirming highest stability and generalizability.
- **Recommendation**: Use **Random Forest** for conservative, high-precision targeting. Use **XGBoost** when maximum purchase capture is desired.