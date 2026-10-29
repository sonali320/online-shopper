# Predicting Online Shoppers Purchasing Intention with Machine Learning

A production-grade Machine Learning solution for predicting whether an online browsing session will result in a purchase (`Revenue = TRUE`). This project implements, evaluates, and benchmarks **Balanced Logistic Regression**, **Balanced Random Forest**, and **Balanced XGBoost** models, reproducing the experimental results documented in **`Results.pdf`**.

---

## 📊 Benchmark Summary (`Results.pdf`)

All models were evaluated on the **UCI Online Shoppers Purchasing Intention Dataset** (12,330 sessions, 18 features) with a 70/30 train-test split (3,699 test samples, random seed 0).

| Model | Accuracy | Specificity (TNR) | Sensitivity (TPR) | 5-Fold CV Accuracy | CV Std Dev | Primary Advantage |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Random Forest (Tuned)** | **89.19%** | **96.10%** | 55.00% | **90.56%** | **±0.34%** | Highest overall accuracy, fewest false alarms (only 120 false positives). |
| **XGBoost (Tuned)** | 87.00% | 88.37% | **80.23%** | 87.70% | ±0.58% | **Highest capture rate** (identifies 80.23% of all purchasers, only 123 false negatives). |
| **Logistic Regression (Tuned)** | 87.08% | 88.66% | 79.26% | 87.60% | ±0.52% | Linear interpretability with Yeo-Johnson PowerTransformer normalization. |

### 🎯 Key Business Decision Takeaways
1. **For Conversion-Maximizing Campaigns (Retargeting, Checkout Incentives, Live Chat):**
   Use **XGBoost** or **Logistic Regression**. With **~80% Sensitivity**, they capture the vast majority of potential revenue opportunities.
2. **For High-Cost / High-Risk Interventions (Heavy Discount Codes, Expensive VIP Perks):**
   Use **Random Forest**. With **96.10% Specificity** and **89.19% Accuracy**, it prevents wasting discounts on shoppers who were never going to buy.

---

## 🛠 Project Architecture

```
Predicting-Online-Shoppers-Purchasing-Intention-main/
│
├── .env                              # Environment variables (configured)
├── .env.example                      # Configuration template
├── .gitignore                        # Git exclusion rules
├── requirements.txt                  # Production dependencies
├── requirements-dev.txt              # Testing & dev dependencies
├── pyproject.toml                    # Modern project configuration
├── README.md                         # Project documentation (this file)
│
├── .vscode/                          # VS Code Workspace Configuration
│   ├── settings.json                 # Python virtualenv, formatting, PowerShell defaults
│   └── launch.json                   # F5 Debug configurations
│
├── data/
│   └── online_shoppers_intention.csv # Dataset (UCI Machine Learning Repository)
│
├── models/                           # Serialized model pipelines (.joblib)
│   ├── logistic_regression_pipeline.joblib
│   ├── random_forest_pipeline.joblib
│   └── xgboost_pipeline.joblib
│
├── reports/                          # Generated benchmark comparison reports
│   ├── pdf_benchmark_comparison.md
│   └── pdf_benchmark_comparison.json
│
├── src/                              # Modular Python Package
│   ├── config.py                     # Central configuration & PDF benchmarks
│   ├── data_loader.py                # Data loading, validation, and schema checks
│   ├── preprocessor.py               # PowerTransformer + OneHotEncoder pipelines
│   ├── feature_selection.py          # SelectFromModel feature reduction
│   ├── models.py                     # Base & tuned model architectures
│   ├── evaluate.py                   # Metrics & Results.pdf comparison engine
│   ├── train.py                      # End-to-end training and artifact saving
│   └── predict.py                    # Real-time single & batch inference engine
│
├── web/                              # Interactive Web Dashboard
│   ├── app.py                        # HTTP / REST API Server
│   ├── templates/index.html          # Modern glassmorphism UI
│   └── static/                       # CSS design system & client JS
│
├── tests/                            # Automated Pytest Suite
│   ├── test_data_loader.py
│   ├── test_pipeline.py
│   ├── test_models.py
│   └── test_pdf_benchmark.py
│
├── main.py                           # Unified CLI Entry Point
├── reproduce_pdf_results.py          # Benchmark verification script
└── Predicting-Online-Shoppers-Purchasing-Intention-main/  # Standalone original scripts (fixed & enhanced)
    ├── LogisticRegression.py
    ├── RandomForest.py
    ├── XGBoost.py
    └── Results.pdf
```

---

## 💻 Step-by-Step PowerShell Terminal Commands

Run these exact commands in **Windows PowerShell** inside your VS Code terminal.

### Step 1: Sync Dataset to Workspace Root
```powershell
if (!(Test-Path "data")) { Copy-Item -Path "Predicting-Online-Shoppers-Purchasing-Intention-main/data" -Destination "." -Recurse }
```

### Step 2: Activate the Virtual Environment
```powershell
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell restricts script execution, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` then re-run the activate command).*

### Step 3: Install / Verify Dependencies
The virtual environment is already pre-configured with core packages. To ensure test and dev tooling is ready:
```powershell
pip install -r requirements-dev.txt
```

### Step 4: Reproduce Results.pdf Benchmark Table
Runs all 3 tuned models and outputs the exact side-by-side benchmark comparison against `Results.pdf`:
```powershell
python reproduce_pdf_results.py
```
*(Or via CLI: `python main.py benchmark`)*

### Step 5: Run Automated Tests (Pytest)
Validates data integrity, pipelines, models, and verifies that accuracy is within tolerance of `Results.pdf`:
```powershell
pytest tests -v
```

### Step 6: Launch the Interactive Web Dashboard
Launches the modern browser dashboard with live session simulation, probability gauge, and benchmark viewer:
```powershell
python main.py launch-web
```
Open your browser at: **`http://127.0.0.1:5000`**

### Step 7: Test Real-time Prediction via CLI
```powershell
python main.py predict --model "Random Forest"
```
Or test with XGBoost for high sensitivity:
```powershell
python main.py predict --model "XGBoost"
```

---

## 🔬 VS Code One-Click Run & Debug

You can also use VS Code's native **Run & Debug** panel (`Ctrl+Shift+D`):
- **Reproduce Results.pdf Benchmark**: Runs `reproduce_pdf_results.py` in the integrated terminal.
- **Train & Evaluate All Models**: Trains all pipelines and updates `models/` and `reports/`.
- **Launch Interactive Web Dashboard**: Starts the local server on port 5000.
- **Run Automated Tests**: Executes pytest with detailed assertions.
- **Run Standalone Scripts**: Run `LogisticRegression.py`, `RandomForest.py`, or `XGBoost.py` directly.
