@echo off

set VENV_DIR=venv
REM Check if virtual environment exists

if not exist %VENV_DIR% (
    python -m venv %VENV_DIR%
    echo Virtual environment created.
)

REM Activate virtual environment

call %VENV_DIR%\Scripts\activate.bat

REM Install dependencies

if exist requirements.txt (
    pip install -r requirements.txt
)

REM Run the Python script
python m8.py
pause