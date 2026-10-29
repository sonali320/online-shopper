#!/usr/bin/env python
"""
Main Entry Point - Predicting Online Shoppers Purchasing Intention
-----------------------------------------------------------------
Commands:
  python main.py benchmark    - Reproduce all metrics from Results.pdf
  python main.py train        - Train and serialize all 3 model pipelines
  python main.py launch-web   - Launch the interactive glassmorphic web dashboard
  python main.py predict      - Run sample prediction from CLI
  python main.py test         - Execute the automated test suite
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

def cmd_benchmark(args):
    from reproduce_pdf_results import main as run_benchmark
    run_benchmark()

def cmd_train(args):
    from src.train import train_and_evaluate_all
    save_models = not args.no_save
    train_and_evaluate_all(save_models=save_models, generate_report=True)

def cmd_launch_web(args):
    from web.app import start_server
    start_server(host=args.host, port=args.port)

def cmd_streamlit(args):
    import subprocess
    shadow = ROOT_DIR / "streamlit.py"
    if shadow.exists():
        try:
            shadow.unlink()
        except Exception:
            pass
    cmd = [sys.executable, "-m", "streamlit", "run", str(ROOT_DIR / "app.py")]
    subprocess.run(cmd)

def cmd_predict(args):
    from src.predict import predict_single
    sample = {
        "Administrative": 3,
        "Administrative_Duration": 72.0,
        "Informational": 1,
        "Informational_Duration": 18.5,
        "ProductRelated": 28,
        "ProductRelated_Duration": 1420.5,
        "BounceRates": 0.005,
        "ExitRates": 0.012,
        "PageValues": 53.8,
        "SpecialDay": 0.0,
        "Month": "Nov",
        "OperatingSystems": 2,
        "Browser": 2,
        "Region": 1,
        "TrafficType": 2,
        "VisitorType": "Returning_Visitor",
        "Weekend": False
    }
    model_name = args.model
    print(f"\n--- Running Inference with {model_name} ---")
    print(f"Sample Input (High-Engagement Session):")
    for k, v in sample.items():
        print(f"  {k}: {v}")
    print("-" * 50)
    result = predict_single(sample, model_name=model_name)
    print(f"Prediction: {result['prediction_label']}")
    print(f"Purchase Probability: {result['purchase_probability'] * 100.0:.2f}%")
    print(f"Confidence: {result['confidence_score']:.2f}%")
    print("-" * 50)

def cmd_test(args):
    import pytest
    exit_code = pytest.main(["tests", "-v"])
    sys.exit(exit_code)

def main():
    parser = argparse.ArgumentParser(
        description="Online Shoppers Purchasing Intention ML System (Results.pdf)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python main.py benchmark        # Reproduce Results.pdf comparison table
  python main.py train            # Train and save all models
  python main.py launch-web       # Start web application on http://127.0.0.1:5000
  python main.py predict          # Test CLI prediction on sample shopper
  python main.py test             # Run pytest validation suite
"""
    )

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Benchmark command
    subparsers.add_parser("benchmark", help="Reproduce results directly matching Results.pdf")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train all models and save serialized pipelines")
    train_parser.add_argument("--no-save", action="store_true", help="Do not save model artifacts to disk")

    # Web command
    web_parser = subparsers.add_parser("launch-web", help="Launch interactive web dashboard")
    web_parser.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    web_parser.add_argument("--port", type=int, default=5000, help="Port (default: 5000)")

    # Streamlit command
    subparsers.add_parser("streamlit", help="Launch Streamlit Antigravity frontend (streamlit.py)")

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Run single session prediction")
    predict_parser.add_argument(
        "--model",
        choices=["Random Forest", "XGBoost", "Logistic Regression"],
        default="Random Forest",
        help="Model engine to use"
    )

    # Test command
    subparsers.add_parser("test", help="Run automated test suite")

    args = parser.parse_args()

    if args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "launch-web":
        cmd_launch_web(args)
    elif args.command == "streamlit":
        cmd_streamlit(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "test":
        cmd_test(args)
    else:
        # Default action if no subcommand is given: run benchmark
        cmd_benchmark(args)

if __name__ == "__main__":
    main()
