#!/bin/bash

# AI Parts Recommendation System - Conda Environment Setup Script
# This script creates and activates the conda environment for the project

set -e  # Exit on any error

echo "🚀 Setting up AI Parts Recommendation System with Conda..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed. Please install Anaconda or Miniconda first."
    echo "   Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Check if environment.yml exists
if [ ! -f "environment.yml" ]; then
    echo "❌ environment.yml not found. Please run this script from the project root."
    exit 1
fi

# Create conda environment from environment.yml
echo "📦 Creating conda environment from environment.yml..."
conda env create -f environment.yml

# Activate the environment
echo "🔄 Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate ai-parts-recommendation

# Verify environment is active
if [[ "$CONDA_DEFAULT_ENV" == "ai-parts-recommendation" ]]; then
    echo "✅ Environment 'ai-parts-recommendation' is now active!"
    echo ""
    echo "📋 Next steps:"
    echo "   1. Initialize the database: python scripts/init_database.py"
    echo "   2. Run the API: python run_api.py"
    echo "   3. Run tests: pytest"
    echo ""
    echo "💡 To activate this environment in the future, run:"
    echo "   conda activate ai-parts-recommendation"
else
    echo "❌ Failed to activate environment"
    exit 1
fi
