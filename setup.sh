#!/usr/bin/env bash

set -e

echo "Starting system setup..."

# 1. Check Python version
if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
    echo "ERROR: Python 3.11 or higher is required."
    exit 1
fi
echo "Python 3.11+ found."

# 2. Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment 'venv'..."
    python3 -m venv venv
else
    echo "Virtual environment 'venv' already exists."
fi

# 3. Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# 4. Upgrade pip
echo "Upgrading pip..."
python3 -m pip install --upgrade pip

# 5. Install dependencies
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    python3 -m pip install -r requirements.txt
else
    echo "WARNING: requirements.txt not found."
fi

# 6. Copy .env.example to .env
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo "Creating .env from .env.example..."
        cp .env.example .env
    else
        echo "WARNING: .env.example not found."
    fi
else
    echo ".env already exists — skipping to protect your keys."
fi

# 7. Create models directory
if [ ! -d "models" ]; then
    echo "Creating 'models' directory..."
    mkdir models
fi

echo "=========================================================="
echo "Setup complete. Next step: open .env and add your API keys."
echo "=========================================================="
