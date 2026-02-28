@echo off
REM SafeSight ML Service Starter
REM =================================
REM Always uses the virtual environment Python with TensorFlow

echo Starting SafeSight ML Service...
echo.

cd /d "%~dp0"

REM Check if venv exists
if not exist "venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found!
    echo.
    echo Please create it first:
    echo   python -m venv venv
    echo   venv\Scripts\pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

REM Activate venv and start service
echo Using Python: venv\Scripts\python.exe
echo.

venv\Scripts\python.exe main.py

pause
