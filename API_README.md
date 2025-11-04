# AI Parts Recommendation API

A FastAPI-based REST API for generating intelligent vehicle parts recommendations using machine learning (LightGBM).

## Features

- **ML-Powered Recommendations**: Uses LightGBM model to predict parts that need replacement
- **Seasonal & Terrain Adjustments**: Applies business rules based on location and season
- **EMA-Based Usage Patterns**: Calculates Exponential Moving Average for usage categorization
- **User Feedback Integration**: Learns from user acceptance/rejection of recommendations
- **High Performance**: Supports 500+ concurrent users with <2 second response time
- **Comprehensive Monitoring**: Health checks, metrics, and status endpoints
- **Production Ready**: Proper error handling, logging, and security

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

Create a `.env` file:

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/parts_recommendation
MONGODB_URL=mongodb://localhost:27017/parts_recommendation

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# Model
MODEL_PATH=models/latest
CONFIDENCE_THRESHOLD=80.0
MODEL_VERSION=1.0.0

# API
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=true
LOG_LEVEL=INFO

# Security
SECRET_KEY=your-secret-key-here
```

### 3. Start the API Server

```bash
python run_api.py
```

The API will be available at `http://localhost:8000`

### 4. View API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### Health & Status

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health/` | GET | Basic health check |
| `/api/health/detailed` | GET | Comprehensive health check |
| `/api/health/model` | GET | ML model status |
| `/api/health/metrics` | GET | System performance metrics |
| `/api/health/readiness` | GET | Kubernetes readiness probe |
| `/api/health/liveness` | GET | Kubernetes liveness probe |

### Recommendations

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/recommendations/generate` | POST | Generate parts recommendations |
| `/api/recommendations/batch` | POST | Batch recommendations for multiple vehicles |
| `/api/recommendations/history/{vehicle_id}` | GET | Get recommendation history |

### Feedback

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/feedback/submit` | POST | Submit user feedback |
| `/api/feedback/stats` | GET | Get recommendation statistics |
| `/api/feedback/history/{vehicle_id}` | GET | Get feedback history |
| `/api/feedback/update/{feedback_id}` | PUT | Update existing feedback |
| `/api/feedback/delete/{feedback_id}` | DELETE | Delete feedback |

## Usage Examples

### Generate Recommendations

```python
import requests

# Generate recommendations for a vehicle
response = requests.post("http://localhost:8000/api/recommendations/generate", json={
    "vehicle_id": "MH12AB1234",
    "current_odometer": 15250.5,
    "customer_complaints": "Brake making noise when stopping",
    "dealer_code": "DLR_MUM_01"
})

recommendations = response.json()
print(f"Found {len(recommendations['recommendations'])} recommendations")
```

### Submit Feedback

```python
# Submit feedback on a recommendation
response = requests.post("http://localhost:8000/api/feedback/submit", json={
    "recommendation_id": 123,
    "feedback_type": "ACCEPTED",
    "feedback_reason": "Customer approved this recommendation",
    "actual_cost": 2500.0
})
```

### Batch Processing

```python
# Generate recommendations for multiple vehicles
response = requests.post("http://localhost:8000/api/recommendations/batch", json={
    "vehicles": [
        {
            "vehicle_id": "MH12AB1234",
            "current_odometer": 15250.5,
            "dealer_code": "DLR_MUM_01"
        },
        {
            "vehicle_id": "DL01CD5678",
            "current_odometer": 25000.0,
            "dealer_code": "DLR_DEL_01"
        }
    ]
})
```

## Request/Response Models

### Recommendation Request

```json
{
    "vehicle_id": "MH12AB1234",
    "current_odometer": 15250.5,
    "customer_complaints": "Brake making noise when stopping",
    "dealer_code": "DLR_MUM_01"
}
```

### Recommendation Response

```json
{
    "status": "success",
    "vehicle_info": {
        "vehicle_id": "MH12AB1234",
        "vehicle_model": "Pulsar 150",
        "current_odometer": 15250.5,
        "dealer_code": "DLR_MUM_01",
        "region_code": "MH",
        "terrain_type": "Urban",
        "season_code": "Summer",
        "ema_value": 520.5,
        "ema_category": "MEDIUM_USAGE"
    },
    "recommendations": [
        {
            "rank": 1,
            "part_code": "BP001",
            "part_name": "Brake Pads - Front",
            "confidence_score": 95.2,
            "category": "Brakes",
            "estimated_cost": 2500.0,
            "reasoning": {
                "primary_factor": "High odometer reading",
                "secondary_factors": ["Customer complaints", "Seasonal wear"]
            },
            "seasonal_adjustment": 5.0,
            "terrain_adjustment": 2.0,
            "final_confidence": 95.2
        }
    ],
    "total_estimated_cost": 15000.0,
    "model_version": "1.0.0",
    "timestamp": "2024-01-15T10:30:00Z",
    "processing_time_ms": 1250.5
}
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | Required |
| `MONGODB_URL` | MongoDB connection string | Required |
| `REDIS_HOST` | Redis host | localhost |
| `REDIS_PORT` | Redis port | 6379 |
| `MODEL_PATH` | Path to ML model | models/latest |
| `CONFIDENCE_THRESHOLD` | Minimum confidence for recommendations | 80.0 |
| `API_HOST` | API server host | 0.0.0.0 |
| `API_PORT` | API server port | 8000 |
| `DEBUG` | Enable debug mode | false |
| `LOG_LEVEL` | Logging level | INFO |

### Model Configuration

The API uses LightGBM models with the following features:

- **Categorical Features**: vehicle_model, dealer_code, region_code, terrain_type, season_code
- **Numerical Features**: current_odometer, ema_value, complaint_length
- **Target**: Binary classification (needs replacement: 1, doesn't need: 0)

## Performance

### Response Time Targets

- **Recommendation Generation**: < 2 seconds
- **Model Inference**: < 100 milliseconds
- **Database Queries**: < 200 milliseconds
- **Cache Hits**: < 10 milliseconds

### Scalability

- **Concurrent Users**: 500+
- **Throughput**: 10,000+ vehicles per hour
- **Batch Processing**: Up to 100 vehicles per request

## Monitoring

### Health Checks

The API provides comprehensive health monitoring:

```bash
# Basic health check
curl http://localhost:8000/api/health/

# Detailed health check
curl http://localhost:8000/api/health/detailed

# Model status
curl http://localhost:8000/api/health/model

# System metrics
curl http://localhost:8000/api/health/metrics
```

### Metrics

The API exposes Prometheus-compatible metrics:

- `recommendations_generated_total` - Total recommendations generated
- `recommendation_latency_seconds` - Recommendation generation time
- `active_models` - Number of active models
- `cache_hits_total` - Cache hit count
- `cache_misses_total` - Cache miss count

## Security

### Authentication

The API supports JWT-based authentication:

```python
# Include JWT token in requests
headers = {
    "Authorization": "Bearer your-jwt-token"
}
```

### Rate Limiting

- **Default**: 100 requests per minute per user
- **Batch Processing**: 10 requests per minute per user
- **Configurable**: Via environment variables

### Input Validation

- **Vehicle ID**: Validates format (2-letter state + 2-digit RTO + 1-2 letters + 4 digits)
- **Odometer**: Must be positive and reasonable (< 1M km)
- **Complaints**: Maximum 1000 characters
- **Dealer Code**: Validates format and existence

## Error Handling

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request (validation error) |
| 401 | Unauthorized |
| 403 | Forbidden |
| 404 | Not Found (vehicle not found) |
| 422 | Unprocessable Entity (validation error) |
| 429 | Too Many Requests (rate limit) |
| 500 | Internal Server Error |
| 503 | Service Unavailable |

### Error Response Format

```json
{
    "status": "error",
    "error_code": "VEHICLE_NOT_FOUND",
    "message": "Vehicle not found: MH12AB1234",
    "details": {
        "vehicle_id": "MH12AB1234"
    },
    "timestamp": "2024-01-15T10:30:00Z"
}
```

## Development

### Running Tests

```bash
# Run unit tests
pytest tests/unit/

# Run integration tests
pytest tests/integration/

# Run all tests
pytest
```

### Code Quality

```bash
# Format code
black src/

# Lint code
flake8 src/

# Type checking
mypy src/
```

### API Testing

Use the provided example client:

```bash
python examples/api_usage.py
```

## Deployment

### Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY run_api.py .

EXPOSE 8000
CMD ["python", "run_api.py"]
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: parts-recommendation-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: parts-recommendation-api
  template:
    metadata:
      labels:
        app: parts-recommendation-api
    spec:
      containers:
      - name: api
        image: parts-recommendation-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
        livenessProbe:
          httpGet:
            path: /api/health/liveness
            port: 8000
        readinessProbe:
          httpGet:
            path: /api/health/readiness
            port: 8000
```

## Support

For issues and questions:

1. Check the API documentation at `/docs`
2. Review the health endpoints for system status
3. Check logs for detailed error information
4. Contact the development team

## License

This project is proprietary software. All rights reserved.
