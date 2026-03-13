@echo off
echo ===================================
echo  Data Analysis Assistant v3.0
echo  Install Script
echo ===================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.9+
    echo Download: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [1/3] Creating virtual environment...
python -m venv data_assistant_env
call data_assistant_env\Scripts\activate.bat

echo [2/3] Upgrading pip...
python -m pip install --upgrade pip

echo [3/3] Installing dependencies (may take a few minutes)...
pip install -r requirements.txt

echo.
echo ===================================
echo  Install complete!
echo ===================================
echo.
echo Next steps:
echo   1. Install Ollama: https://ollama.ai/
echo   2. Pull model: ollama pull qwen2.5:14b
echo   3. Run: double-click start.bat
echo.
pause
