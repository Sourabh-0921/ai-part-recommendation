# AI Parts Recommendation System

An AI-powered vehicle parts recommendation system for automotive service centers that uses machine learning (LightGBM) to predict which parts need replacement based on vehicle data, service history, customer complaints, and environmental factors.

## Features

- **Intelligent Recommendations**: Predict parts needing replacement with 80%+ confidence
- **Top 10 Recommendations**: Display the most relevant parts per vehicle
- **Learning System**: Learn from user feedback (accepted/rejected recommendations)
- **Seasonal Adjustments**: Apply seasonal and terrain-based adjustments
- **High Performance**: Handle 500+ concurrent users with <2 second response time
- **Scalable**: Process 10,000+ vehicles efficiently

## Technology Stack

### Backend
- **Language**: Python 3.9+
- **ML Framework**: LightGBM 3.3+ (with native categorical feature support)
- **API Framework**: FastAPI
- **Data Processing**: pandas, numpy, scikit-learn
- **Workflow**: Apache Airflow for data pipelines

### Databases
- **Primary**: PostgreSQL 13+ (vehicle data, service history, parts inventory)
- **Configuration**: MongoDB (seasonal/terrain configs, business rules)
- **Cache**: Redis (predictions, session data)

### Infrastructure
- **Message Queue**: RabbitMQ or Apache Kafka
- **Monitoring**: Prometheus metrics
- **Model Registry**: MLflow

## Project Structure

```
project_root/
├── src/
│   ├── api/                    # FastAPI routes and endpoints
│   ├── models/                 # ML model code
│   ├── services/               # Business logic
│   ├── data/                   # Data access layer
│   ├── config/                 # Configuration management
│   ├── utils/                  # Utility functions
│   └── airflow_dags/          # Airflow DAG definitions
├── tests/
│   ├── unit/
│   ├── integration/
│   └── load/
├── scripts/                    # Deployment and utility scripts
├── docs/                       # Documentation
├── requirements.txt
├── requirements-dev.txt
└── env.example
```

## Quick Start

### Prerequisites

- **Conda**: Anaconda or Miniconda installed
- **Python**: 3.9+ (managed by conda)
- **PostgreSQL**: 13+ (for production)
- **Redis**: For caching (optional for development)

### 1. Clone and Setup

```bash
git clone <repository-url>
cd ai-parts-recommendation-system
```

### 2. Create Conda Environment

```bash
# Option A: Use the automated setup script
./activate_conda.sh

# Option B: Manual setup
conda env create -f environment.yml
conda activate ai-parts-recommendation
```

### 3. Environment Configuration

```bash
# Copy environment template
cp env.example .env

# Edit configuration
nano .env
```

### 4. Database Setup

```bash
# Initialize database
python scripts/init_database.py
```

### 5. Run the Application

```bash
# Start the API server
python run_api.py

# Or with uvicorn directly
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Verify Installation

```bash
# Run tests to verify everything works
pytest

# Check API health
curl http://localhost:8000/api/health
```

## Configuration

The system uses environment variables for configuration. Key settings include:

- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_HOST`: Redis server host
- `MODEL_PATH`: Path to trained models
- `CONFIDENCE_THRESHOLD`: Minimum confidence for recommendations (default: 80.0)
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

See `env.example` for all available configuration options.

## API Endpoints

### Core Endpoints

- `POST /api/recommendations/generate` - Generate parts recommendations
- `GET /api/recommendations/{vehicle_id}` - Get recommendations for a vehicle
- `POST /api/feedback` - Submit user feedback
- `GET /api/health` - Health check endpoint

### Documentation

- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /redoc` - Alternative API documentation (ReDoc)

## Development

### Code Quality

The project follows strict code quality standards:

- **Type Hints**: All functions must have type hints
- **Docstrings**: Google-style docstrings for all classes and functions
- **PEP 8**: Strict adherence to Python style guide
- **Testing**: Comprehensive unit, integration, and load tests

### Running Tests

```bash
# Make sure conda environment is active
conda activate ai-parts-recommendation

# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/load/
```

### Code Formatting

```bash
# Make sure conda environment is active
conda activate ai-parts-recommendation

# Format code with black
black .

# Sort imports with isort
isort .

# Check code style with flake8
flake8 .
```

### Environment Management

```bash
# Activate environment
conda activate ai-parts-recommendation

# Deactivate environment
conda deactivate

# Update environment from environment.yml
conda env update -f environment.yml

# Remove environment (if needed)
conda env remove -n ai-parts-recommendation

# List all environments
conda env list
```

## Performance Requirements

- **Recommendation Generation**: < 2 seconds
- **Model Inference**: < 100 milliseconds
- **Database Queries**: < 200 milliseconds
- **Cache Hits**: < 10 milliseconds

## Monitoring

The system includes comprehensive monitoring:

- **Prometheus Metrics**: Performance and business metrics
- **Structured Logging**: JSON-formatted logs for production
- **Health Checks**: Database and external service monitoring
- **Error Tracking**: Detailed error logging and alerting

## Contributing

1. Follow the code quality standards outlined
2. Write comprehensive tests for new functionality
3. Update documentation for API changes
4. Ensure all tests pass before submitting PRs


