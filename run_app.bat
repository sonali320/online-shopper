@echo off
setlocal
title Online Shopper Behaviour Segmentation and Purchase Intention Prediction

REM 1. Navigate to the project directory
cd /d "%~dp0"

REM Prevent Python namespace collision: remove local streamlit.py so it does not shadow the real streamlit library
if exist "%~dp0streamlit.py" (
    echo [INFO] Removing local streamlit.py to prevent library shadowing...
    del /f /q "%~dp0streamlit.py"
)
if exist "%~dp0__pycache__\streamlit*.pyc" (
    del /f /q "%~dp0__pycache__\streamlit*.pyc"
)

echo ===============================================================================
echo   Online Shopper Behaviour Segmentation and Purchase Intention Prediction
echo ===============================================================================
echo.

REM 2. Check whether Python is installed
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python is not installed or not available in the system PATH.
    echo Please install Python from https://www.python.org/
    echo Make sure to check the option: "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

REM 3. Create a virtual environment if needed
if not exist "%~dp0.venv" (
    if not exist "%~dp0venv" (
        echo [INFO] Virtual environment not found. Creating .venv...
        python -m venv "%~dp0.venv"
    )
)

REM 4. Activate the environment
if exist "%~dp0.venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment .venv...
    call "%~dp0.venv\Scripts\activate.bat"
) else if exist "%~dp0venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment venv...
    call "%~dp0venv\Scripts\activate.bat"
) else (
    echo [INFO] Running with active Python interpreter.
)

REM 5. Install missing dependencies from requirements.txt
echo [INFO] Checking dependencies from requirements.txt...
python -m pip install -r "%~dp0requirements.txt"

python -c "import streamlit" >nul 2>nul
if errorlevel 1 (
    echo [INFO] Installing Streamlit...
    python -m pip install streamlit
)

REM 6. Verify required dataset exists
if not exist "%~dp0data\online_shoppers_intention.csv" (
    if exist "%~dp0Predicting-Online-Shoppers-Purchasing-Intention-main\data\online_shoppers_intention.csv" (
        echo [INFO] Setting up data directory...
        if not exist "%~dp0data" mkdir "%~dp0data"
        copy "%~dp0Predicting-Online-Shoppers-Purchasing-Intention-main\data\online_shoppers_intention.csv" "%~dp0data\online_shoppers_intention.csv" >nul
    ) else (
        echo [ERROR] Dataset file 'data/online_shoppers_intention.csv' was not found!
        pause
        exit /b 1
    )
)

REM 7. Train the model only when necessary
if not exist "%~dp0model.joblib" (
    if not exist "%~dp0models\random_forest_pipeline.joblib" (
        echo [INFO] Pre-trained model not found. Training Random Forest model now...
        python "%~dp0train_model.py"
        if errorlevel 1 (
            echo [ERROR] Model training encountered an error.
            pause
            exit /b 1
        )
    )
)

REM 8. Start the app using python -m streamlit run app.py
echo.
echo ===============================================================================
echo   Starting Streamlit Dashboard...
echo   URL: http://localhost:8501
echo   Press Ctrl+C in this window to stop the server.
echo ===============================================================================
echo.

python -m streamlit run "%~dp0app.py"

REM 10. Display helpful errors and keep console open if startup fails
if errorlevel 1 (
    echo.
    echo [ERROR] The application exited with an error code.
    pause
)

endlocal