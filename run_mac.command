#!/bin/bash
# MAT launcher for macOS (Intel and Apple Silicon).
# First run creates a virtual environment and installs dependencies.
# Double-click in Finder, or run: ./run_mac.command

cd "$(dirname "$0")"

VENV_DIR=venv

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "Virtual environment created."
fi

source "$VENV_DIR/bin/activate"

if [ -f requirements.txt ]; then
    pip install --quiet -r requirements.txt
fi

python mat_v4.py
