#!/bin/bash

# Database setup script for AI Parts Recommendation System
# This script sets up the database with migrations and sample data

set -e  # Exit on any error

echo "🚀 Starting AI Parts Recommendation System Database Setup"
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "alembic.ini" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is required but not installed"
    exit 1
fi

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ Error: pip3 is required but not installed"
    exit 1
fi

echo "📦 Installing required packages..."
pip3 install -r requirements.txt

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found. Creating from template..."
    if [ -f "env.example" ]; then
        cp env.example .env
        echo "📝 Please edit .env file with your database credentials"
        echo "   Required variables: DATABASE_URL, SECRET_KEY"
        read -p "Press Enter after updating .env file..."
    else
        echo "❌ Error: env.example file not found"
        exit 1
    fi
fi

# Source environment variables
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Check if DATABASE_URL is set
if [ -z "$DATABASE_URL" ]; then
    echo "❌ Error: DATABASE_URL not set in .env file"
    echo "   Example: DATABASE_URL=postgresql://user:password@localhost/ai_parts_recommendation"
    exit 1
fi

echo "🗄️  Running database migrations..."

# Initialize Alembic if not already done
if [ ! -d "migrations/versions" ] || [ ! "$(ls -A migrations/versions)" ]; then
    echo "📋 Initializing Alembic..."
    alembic revision --autogenerate -m "Initial schema"
fi

# Run migrations
echo "🔄 Applying database migrations..."
alembic upgrade head

echo "📊 Inserting sample data..."
python3 scripts/init_database.py

echo "✅ Database setup completed successfully!"
echo ""
echo "🎯 Next steps:"
echo "   1. Verify database connection: python3 -c 'from src.data.database import test_connection; print(\"✅\" if test_connection() else \"❌\")'"
echo "   2. Start the API server: python3 run_api.py"
echo "   3. Test the API: curl http://localhost:8000/health"
echo ""
echo "📚 Database schema includes:"
echo "   - vehicle_master: Vehicle information"
echo "   - service_history: Service records"
echo "   - part_master: Parts catalog"
echo "   - part_recommendations: ML predictions"
echo "   - user_feedback: User interactions"
echo "   - business_rules: Configuration rules"
echo "   - seasonal_config: Seasonal adjustments"
echo "   - terrain_config: Terrain adjustments"
echo "   - model_versions: ML model tracking"
echo "   - prediction_cache: Performance optimization"
echo ""
echo "🔧 Useful commands:"
echo "   - View migration status: alembic current"
echo "   - Create new migration: alembic revision --autogenerate -m 'Description'"
echo "   - Rollback migration: alembic downgrade -1"
echo "   - Reset database: alembic downgrade base && alembic upgrade head"
