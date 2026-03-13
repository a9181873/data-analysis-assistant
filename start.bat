@echo off
echo ===================================
echo  Data Analysis Assistant v3.0
echo ===================================
echo.

if exist "data_assistant_env" (
    call data_assistant_env\Scripts\activate.bat
    echo Starting... browser will open at http://localhost:8501
    echo Press Ctrl+C to stop
    echo.
    streamlit run streamlit_app.py
) else if exist ".venv" (
    call .venv\Scripts\activate.bat
    echo Starting... browser will open at http://localhost:8501
    echo Press Ctrl+C to stop
    echo.
    streamlit run streamlit_app.py
) else (
    echo [ERROR] Virtual environment not found. Run install.bat first.
    pause
)
