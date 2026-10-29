"""
Training Pipeline for Online Shoppers Purchasing Intention.
Executes preprocessing, feature selection, model training, cross-validation,
benchmark verification against Results.pdf, and artifact serialization.
"""

from typing import Dict, Any
import json
import joblib
import pandas as pd

from src.config import (
    MODELS_DIR,
    REPORTS_DIR,
    PDF_BENCHMARKS,
    PROJECT_ROOT
)
from src.data_loader import load_data, get_feature_target_split
from src.preprocessor import prepare_logistic_regression_data, prepare_tree_data
from src.feature_selection import select_top_features
from src.models import (
    create_base_logistic_regression,
    create_tuned_logistic_regression,
    create_base_random_forest,
    create_tuned_random_forest,
    create_decision_tree,
    calculate_scale_pos_weight,
    create_base_xgboost,
    create_tuned_xgboost
)
from src.evaluate import evaluate_model_performance, compare_with_pdf_benchmark

def train_logistic_regression(df: pd.DataFrame) -> Dict[str, Any]:
    """Trains and evaluates tuned Logistic Regression pipeline."""
    X, y = get_feature_target_split(df)
    X_train, X_test, y_train, y_test, preprocessor = prepare_logistic_regression_data(X, y)

    # 1. Base model for feature selection
    base_model = create_base_logistic_regression()
    base_model.fit(X_train, y_train)

    # 2. Select top features
    X_train_top, X_test_top, selector, top_features = select_top_features(base_model, X_train, X_test)

    # 3. Train tuned model
    tuned_model = create_tuned_logistic_regression()
    tuned_model.fit(X_train_top, y_train)

    # 4. Evaluate metrics & cross validation
    metrics = evaluate_model_performance(tuned_model, X_train_top, y_train, X_test_top, y_test)
    comparison = compare_with_pdf_benchmark(metrics, PDF_BENCHMARKS["Logistic Regression"])

    return {
        "model_name": "Logistic Regression (Top Features Tuned)",
        "preprocessor": preprocessor,
        "selector": selector,
        "model": tuned_model,
        "selected_features": top_features,
        "metrics": metrics,
        "comparison": comparison
    }

def train_random_forest(df: pd.DataFrame) -> Dict[str, Any]:
    """Trains and evaluates tuned Random Forest pipeline."""
    X, y = get_feature_target_split(df)
    X_train, X_test, y_train, y_test, preprocessor = prepare_tree_data(X, y)

    # 1. Base model for feature selection
    base_model = create_base_random_forest()
    base_model.fit(X_train, y_train)

    # 2. Select top features
    X_train_top, X_test_top, selector, top_features = select_top_features(base_model, X_train, X_test)

    # 3. Train tuned model
    tuned_model = create_tuned_random_forest()
    tuned_model.fit(X_train_top, y_train)

    # 4. Evaluate metrics & cross validation
    metrics = evaluate_model_performance(tuned_model, X_train_top, y_train, X_test_top, y_test)
    comparison = compare_with_pdf_benchmark(metrics, PDF_BENCHMARKS["Random Forest"])

    return {
        "model_name": "Random Forest (Top Features Tuned)",
        "preprocessor": preprocessor,
        "selector": selector,
        "model": tuned_model,
        "selected_features": top_features,
        "metrics": metrics,
        "comparison": comparison
    }

def train_decision_tree(
    df: pd.DataFrame,
    preprocessor=None,
    selector=None
) -> Dict[str, Any]:
    """
    Trains and evaluates Decision Tree on the exact same dataset,
    preprocessing steps, and train/test split as Random Forest.
    """
    X, y = get_feature_target_split(df)
    X_train, X_test, y_train, y_test, preproc = prepare_tree_data(X, y, preprocessor=preprocessor)

    if selector is None:
        base_rf = create_base_random_forest()
        base_rf.fit(X_train, y_train)
        X_train_top, X_test_top, sel, top_features = select_top_features(base_rf, X_train, X_test)
    else:
        sel = selector
        X_train_top = sel.transform(X_train)
        X_test_top = sel.transform(X_test)
        top_features = [f for f, s in zip(X_train.columns, sel.get_support()) if s]

    dt_model = create_decision_tree()
    dt_model.fit(X_train_top, y_train)

    metrics = evaluate_model_performance(dt_model, X_train_top, y_train, X_test_top, y_test, perform_cv=False)

    return {
        "model_name": "Decision Tree (Balanced)",
        "preprocessor": preproc,
        "selector": sel,
        "model": dt_model,
        "selected_features": top_features,
        "metrics": metrics
    }

def compare_rf_and_dt(
    df: pd.DataFrame,
    existing_rf_bundle: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Evaluates Random Forest and Decision Tree on the exact same dataset,
    preprocessing steps, and test split, producing authentic side-by-side performance metrics.
    """
    X, y = get_feature_target_split(df)
    X_train, X_test, y_train, y_test, preprocessor = prepare_tree_data(X, y)

    # Use existing selector and RF model if available in bundle to preserve exact benchmark weights
    if existing_rf_bundle and "selector" in existing_rf_bundle and "model" in existing_rf_bundle:
        selector = existing_rf_bundle["selector"]
        rf_model = existing_rf_bundle["model"]
        X_train_top = selector.transform(X_train)
        X_test_top = selector.transform(X_test)
        rf_metrics = evaluate_model_performance(rf_model, X_train_top, y_train, X_test_top, y_test, perform_cv=False)
    else:
        base_model = create_base_random_forest()
        base_model.fit(X_train, y_train)
        X_train_top, X_test_top, selector, _ = select_top_features(base_model, X_train, X_test)
        rf_model = create_tuned_random_forest()
        rf_model.fit(X_train_top, y_train)
        rf_metrics = evaluate_model_performance(rf_model, X_train_top, y_train, X_test_top, y_test, perform_cv=False)

    # Train Decision Tree on the identical training set and feature subset
    dt_model = create_decision_tree()
    dt_model.fit(X_train_top, y_train)
    dt_metrics = evaluate_model_performance(dt_model, X_train_top, y_train, X_test_top, y_test, perform_cv=False)

    return {
        "random_forest": {
            "model_name": "Random Forest",
            "model": rf_model,
            "metrics": rf_metrics
        },
        "decision_tree": {
            "model_name": "Decision Tree",
            "model": dt_model,
            "metrics": dt_metrics
        }
    }

def train_xgboost(df: pd.DataFrame) -> Dict[str, Any]:
    """Trains and evaluates tuned XGBoost pipeline."""
    X, y = get_feature_target_split(df)
    X_train, X_test, y_train, y_test, preprocessor = prepare_tree_data(X, y)

    scale_pos_weight = calculate_scale_pos_weight(y_train)

    # 1. Base model for feature selection
    base_model = create_base_xgboost(scale_pos_weight)
    base_model.fit(X_train, y_train)

    # 2. Select top features
    X_train_top, X_test_top, selector, top_features = select_top_features(base_model, X_train, X_test)

    # 3. Train tuned model
    tuned_model = create_tuned_xgboost(scale_pos_weight)
    tuned_model.fit(X_train_top, y_train)

    # 4. Evaluate metrics & cross validation
    metrics = evaluate_model_performance(tuned_model, X_train_top, y_train, X_test_top, y_test)
    comparison = compare_with_pdf_benchmark(metrics, PDF_BENCHMARKS["XGBoost"])

    return {
        "model_name": "XGBoost (Top Features Tuned)",
        "preprocessor": preprocessor,
        "selector": selector,
        "model": tuned_model,
        "selected_features": top_features,
        "metrics": metrics,
        "comparison": comparison
    }

def generate_markdown_report(results: Dict[str, Any]) -> str:
    """Generates an aesthetic Markdown report comparing reproduced results with Results.pdf."""
    lines = [
        "# Model Performance & Results.pdf Benchmark Comparison",
        "",
        "This report benchmarks the newly trained models against the ground-truth results documented in `Results.pdf`.",
        "",
        "| Model | Metric | Reproduced | Target (PDF) | Difference | Status |",
        "| :--- | :--- | :---: | :---: | :---: | :---: |"
    ]

    for model_key, res in results.items():
        comp = res["comparison"]
        for metric_key, item in comp.items():
            rep = f"{item['reproduced']:.2f}%" if "accuracy" in metric_key or "sensitivity" in metric_key or "specificity" in metric_key else str(item['reproduced'])
            tgt = f"{item['pdf_benchmark']:.2f}%" if "accuracy" in metric_key or "sensitivity" in metric_key or "specificity" in metric_key else str(item['pdf_benchmark'])
            diff = f"{item['difference']:+.2f}" if isinstance(item['difference'], float) else (f"{item['difference']:+d}" if item['difference'] is not None else "-")
            status = item['status']
            lines.append(f"| **{model_key}** | {item['label']} | {rep} | {tgt} | {diff} | {status} |")

    lines.extend([
        "",
        "## Key Observations & Insights from Results.pdf",
        "",
        "- **Accuracy**: Balanced Random Forest achieves highest overall accuracy (89.19%), followed by Logistic Regression (87.08%) and XGBoost (87.00%).",
        "- **Sensitivity (Capture Rate)**: XGBoost leads in capturing actual purchasers (80.23%), followed closely by Logistic Regression (79.26%). Random Forest achieves 55.00%.",
        "- **Specificity**: Random Forest excels at correctly identifying non-purchasers (96.10%), with minimal false positives (120 vs 349/358).",
        "- **Cross-Validation Stability**: Random Forest exhibits the lowest standard deviation (0.34%) across 5 folds, confirming highest stability and generalizability.",
        "- **Recommendation**: Use **Random Forest** for conservative, high-precision targeting. Use **XGBoost** when maximum purchase capture is desired."
    ])

    return "\n".join(lines)

def train_and_evaluate_all(save_models: bool = True, generate_report: bool = True) -> Dict[str, Any]:
    """Runs end-to-end training, benchmarking, and artifact generation."""
    print("Loading dataset...")
    df = load_data()

    print("Training Logistic Regression (Top Features Tuned)...")
    res_lr = train_logistic_regression(df)

    print("Training Random Forest (Top Features Tuned)...")
    res_rf = train_random_forest(df)

    print("Training XGBoost (Top Features Tuned)...")
    res_xgb = train_xgboost(df)

    results = {
        "Logistic Regression": res_lr,
        "Random Forest": res_rf,
        "XGBoost": res_xgb
    }

    if save_models:
        print(f"Saving serialized model pipelines to {MODELS_DIR}...")
        for name, data in results.items():
            slug = name.lower().replace(" ", "_")
            artifact = {
                "model_name": data["model_name"],
                "preprocessor": data["preprocessor"],
                "selector": data["selector"],
                "model": data["model"],
                "selected_features": data["selected_features"],
                "metrics": data["metrics"]
            }
            joblib.dump(artifact, MODELS_DIR / f"{slug}_pipeline.joblib")
            if "forest" in slug:
                joblib.dump(artifact, PROJECT_ROOT / "model.joblib")
                joblib.dump(artifact, MODELS_DIR / "model.joblib")

    if generate_report:
        print(f"Writing benchmark comparison reports to {REPORTS_DIR}...")
        report_md = generate_markdown_report(results)
        (REPORTS_DIR / "pdf_benchmark_comparison.md").write_text(report_md, encoding="utf-8")

        summary_json = {
            model_key: {
                "metrics": res["metrics"],
                "comparison": res["comparison"],
                "selected_features_count": len(res["selected_features"])
            }
            for model_key, res in results.items()
        }
        (REPORTS_DIR / "pdf_benchmark_comparison.json").write_text(
            json.dumps(summary_json, indent=2), encoding="utf-8"
        )

    print("Training and evaluation completed successfully!")
    return results
