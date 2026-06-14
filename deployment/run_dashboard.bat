@echo off
echo ===================================================
echo   VENDOR INVOICE INTELLIGENCE DASHBOARD LAUNCHER
echo ===================================================
echo Installing/Checking dependencies...
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Dependency installation failed.
    pause
    exit /b %ERRORLEVEL%
)

echo Starting Streamlit App...
streamlit run app.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to start Streamlit app.
    pause
    exit /b %ERRORLEVEL%
)
