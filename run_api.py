#!/usr/bin/env python3
"""
Startup script for AI Parts Recommendation API.

This script starts the FastAPI application with proper configuration.
"""

import sys
import os
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Set environment variables for development
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("DEBUG", "true")
os.environ.setdefault("LOG_LEVEL", "INFO")

# Import and run the application
if __name__ == "__main__":
    from src.api.main import run_server
    run_server()
