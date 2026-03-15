@echo off
setlocal

echo Starting system setup...

REM 1. Check Python version (approximation for batch, assumes python is available)
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"
if errorlevel 1 (
    echo ERROR: Python 3.11 or higher is required.
    exit /b 1
)
echo Python 3.11+ found.

REM 2. Create virtual environment
if not exist "venv\" (
    echo Creating virtual environment 'venv'...
    python -m venv venv
) else (
    echo Virtual environment 'venv' already exists.
)

REM 3. Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM 4. Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM 5. Install dependencies
if exist "requirements.txt" (
    echo Installing dependencies from requirements.txt...
    python -m pip install -r requirements.txt
) else (
    echo WARNING: requirements.txt not found.
)

REM 6. Copy .env.example to .env
if not exist ".env" (
    if exist ".env.example" (
        echo Creating .env from .env.example...
        copy .env.example .env >nul
    ) else (
        echo WARNING: .env.example not found.
    )
) else (
    echo .env already exists -- skipping to protect your keys.
)

REM 7. Create models directory
if not exist "models\" (
    echo Creating 'models' directory...
    mkdir models
)

echo ==========================================================
echo Setup complete. Next step: open .env and add your API keys.
echo ==========================================================
