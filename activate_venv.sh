#!/bin/bash

# Script to activate the virtual environment for the AI Parts Recommendation System
# Usage: source activate_venv.sh

echo "Activating virtual environment for AI Parts Recommendation System..."
echo "Project directory: $(pwd)"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found!"
    echo "Please run the setup script first."
    exit 1
fi

# Activate the virtual environment
source venv/bin/activate

echo "Virtual environment activated successfully!"
echo "Python version: $(python --version)"
echo "Pip version: $(pip --version)"

# Show installed packages
echo ""
echo "Installed packages:"
pip list | head -20
echo "... (showing first 20 packages)"

echo ""
echo "To deactivate the virtual environment, run: deactivate"
echo "To run the EMA example, run: python examples/ema_usage_example.py"
