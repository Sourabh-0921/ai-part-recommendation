# Virtual Environment Setup Guide

This guide explains how to set up and use the Python virtual environment for the AI Parts Recommendation System.

## Overview

A Python virtual environment has been created to isolate the project dependencies from your system Python installation. This ensures:

- **Dependency Isolation**: Project packages won't conflict with system packages
- **Version Control**: Specific package versions are maintained
- **Reproducible Environment**: Same environment across different machines
- **Easy Cleanup**: Remove the entire environment by deleting the `venv` folder

## Virtual Environment Details

- **Location**: `./venv/` (in project root)
- **Python Version**: 3.11.9
- **Package Manager**: pip 25.2

## Quick Start

### 1. Activate the Virtual Environment

```bash
# Navigate to project directory
cd "/Users/rahulchaturvedi/Project/Excellon/AI Parts Recommendation System"

# Activate virtual environment
source venv/bin/activate

# Or use the provided script
source activate_venv.sh
```

### 2. Verify Installation

```bash
# Check Python version
python --version

# Check installed packages
pip list

# Test core dependencies
python -c "import pandas, numpy, lightgbm, fastapi; print('All dependencies working!')"
```

### 3. Run Examples

```bash
# Run EMA calculator examples
python examples/ema_usage_example.py

# Run API server (when implemented)
python run_api.py
```

### 4. Deactivate When Done

```bash
deactivate
```

## Installed Packages

### Core Dependencies
- **FastAPI** (0.104.1) - Web framework
- **Uvicorn** (0.24.0) - ASGI server
- **Pydantic** (2.5.0) - Data validation
- **LightGBM** (4.6.0) - Machine learning
- **Pandas** (2.1.4) - Data manipulation
- **NumPy** (1.26.4) - Numerical computing
- **Scikit-learn** (1.3.2) - Machine learning utilities

### Database & Storage
- **SQLAlchemy** (1.4.54) - ORM
- **psycopg2-binary** (2.9.9) - PostgreSQL adapter
- **pymongo** (4.6.0) - MongoDB adapter
- **redis** (5.0.1) - Caching

### Development Tools
- **pytest** (7.4.3) - Testing framework
- **black** (23.11.0) - Code formatting
- **flake8** (6.1.0) - Linting
- **mypy** (1.7.1) - Type checking
- **jupyter** (1.0.0) - Interactive notebooks

### Monitoring & Utilities
- **MLflow** (2.8.1) - Model registry
- **prometheus-client** (0.19.0) - Metrics
- **structlog** (23.2.0) - Structured logging

## Troubleshooting

### Common Issues

1. **Virtual environment not found**
   ```bash
   # Recreate virtual environment
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Permission denied**
   ```bash
   # Make activation script executable
   chmod +x activate_venv.sh
   ```

3. **Package import errors**
   ```bash
   # Reinstall packages
   pip install --force-reinstall -r requirements.txt
   ```

4. **PostgreSQL connection issues**
   - Install PostgreSQL client libraries
   - For macOS: `brew install postgresql`
   - For Ubuntu: `sudo apt-get install libpq-dev`

### Environment Variables

Create a `.env` file in the project root:

```bash
# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/parts_recommendation
REDIS_URL=redis://localhost:6379

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Model Configuration
MODEL_PATH=models/latest
CONFIDENCE_THRESHOLD=80.0

# EMA Configuration
EMA_PERIODS=6
MIN_SERVICES_FOR_EMA=2
```

## Development Workflow

### 1. Daily Development
```bash
# Start your day
source venv/bin/activate

# Make changes to code
# Run tests
python -m pytest tests/unit/ -v

# Run examples
python examples/ema_usage_example.py

# When done
deactivate
```

### 2. Adding New Dependencies
```bash
# Activate environment
source venv/bin/activate

# Install new package
pip install new-package

# Update requirements
pip freeze > requirements.txt
```

### 3. Code Quality Checks
```bash
# Format code
black src/ tests/

# Check linting
flake8 src/ tests/

# Type checking
mypy src/

# Run tests
pytest tests/ -v --cov=src
```

## File Structure

```
project_root/
├── venv/                    # Virtual environment
├── src/                     # Source code
├── tests/                   # Test files
├── examples/                # Example scripts
├── requirements.txt         # Production dependencies
├── requirements-dev.txt     # Development dependencies
├── activate_venv.sh         # Activation script
└── VENV_SETUP.md           # This file
```

## Best Practices

1. **Always activate the virtual environment** before working on the project
2. **Never commit the `venv/` folder** to version control
3. **Use `requirements.txt`** to track dependencies
4. **Test your changes** before committing
5. **Keep dependencies up to date** regularly

## Next Steps

1. **Set up your IDE** to use the virtual environment Python interpreter
2. **Configure your database** connections
3. **Run the example scripts** to verify everything works
4. **Start developing** your features!

## Support

If you encounter issues:

1. Check this guide first
2. Verify all dependencies are installed
3. Check the project logs
4. Review the example scripts for usage patterns

Happy coding! 🚀
