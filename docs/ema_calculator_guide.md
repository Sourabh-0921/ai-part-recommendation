# EMA Calculator Guide

## Overview

The Exponential Moving Average (EMA) Calculator is a critical component of the AI Parts Recommendation System. It analyzes vehicle usage patterns based on service history data to categorize vehicles into usage categories: HIGH_USAGE, MEDIUM_USAGE, and LOW_USAGE.

## Key Features

- **Exponential Moving Average Calculation**: Uses 6-period EMA by default for smooth trend analysis
- **Usage Categorization**: Automatically categorizes vehicles based on usage patterns
- **Batch Processing**: Efficiently processes multiple vehicles
- **Anomaly Detection**: Identifies unusual usage patterns
- **Trend Analysis**: Analyzes usage trends over time
- **Statistical Analysis**: Provides comprehensive usage statistics

## Usage Categories

| Category | EMA Range (km/month) | Description |
|----------|----------------------|-------------|
| HIGH_USAGE | > 800 | Heavy usage vehicles requiring frequent maintenance |
| MEDIUM_USAGE | 400-800 | Moderate usage vehicles with standard maintenance needs |
| LOW_USAGE | ≤ 400 | Light usage vehicles with minimal maintenance needs |

## Basic Usage

### Simple EMA Calculation

```python
from src.models.ema_calculator import EMACalculator
import pandas as pd
from datetime import datetime

# Initialize calculator
calculator = EMACalculator(n_periods=6, min_services=2)

# Create service history data
service_history = pd.DataFrame({
    'service_date': [
        datetime(2023, 1, 1),
        datetime(2023, 2, 1),
        datetime(2023, 3, 1),
        datetime(2023, 4, 1),
        datetime(2023, 5, 1),
        datetime(2023, 6, 1)
    ],
    'odometer_reading': [1000, 2500, 4000, 5500, 7000, 8500]
})

# Calculate EMA
ema_value, method, metadata = calculator.calculate_ema(service_history, "VEH001")
category = calculator.categorize_ema(ema_value)

print(f"EMA Value: {ema_value:.2f} km/month")
print(f"Category: {category}")
print(f"Method: {method}")
```

### Using Structured Results

```python
# Get structured result object
result = calculator.calculate_ema_with_result(service_history, "VEH001")

print(f"EMA Value: {result.ema_value}")
print(f"Category: {result.ema_category}")
print(f"Method: {result.calculation_method}")
print(f"Calculated At: {result.calculated_at}")

# Convert to dictionary for serialization
result_dict = result.to_dict()
```

## Advanced Usage

### Batch Processing

```python
# Process multiple vehicles
vehicle_histories = [
    ("VEH001", service_history_1),
    ("VEH002", service_history_2),
    ("VEH003", service_history_3)
]

results = calculator.calculate_ema_for_vehicle_list(vehicle_histories)

for vehicle_id, result in results.items():
    print(f"{vehicle_id}: {result.ema_value:.2f} km/month ({result.ema_category})")
```

### Usage Statistics

```python
# Calculate usage statistics for a fleet
ema_values = {
    "VEH001": 1200,  # High usage
    "VEH002": 600,   # Medium usage
    "VEH003": 300,   # Low usage
    "VEH004": 900,   # High usage
    "VEH005": 500    # Medium usage
}

stats = calculator.get_usage_statistics_structured(ema_values)

print(f"Total Vehicles: {stats.total_vehicles}")
print(f"High Usage: {stats.high_usage_count} ({stats.high_usage_percentage:.1f}%)")
print(f"Medium Usage: {stats.medium_usage_count} ({stats.medium_usage_percentage:.1f}%)")
print(f"Low Usage: {stats.low_usage_count} ({stats.low_usage_percentage:.1f}%)")
print(f"Average EMA: {stats.average_ema:.2f} km/month")
```

### Anomaly Detection

```python
# Detect usage anomalies
ema_values = {
    "VEH001": 500,   # Normal
    "VEH002": 550,   # Normal
    "VEH003": 2000,  # Anomaly
    "VEH004": 480,   # Normal
    "VEH005": 520    # Normal
}

anomalies = calculator.detect_usage_anomalies(ema_values, threshold=2.0)

print(f"Total Anomalies: {anomalies['total_anomalies']}")
for anomaly in anomalies['anomalies']:
    print(f"Vehicle {anomaly['vehicle_id']}: {anomaly['ema_value']} km/month (Z-score: {anomaly['z_score']:.2f})")
```

### Trend Analysis

```python
# Analyze usage trends
trend = calculator.get_ema_trend(service_history, window=3)

print(f"Trend: {trend['trend']}")
print(f"Recent EMA: {trend['recent_ema']:.2f} km/month")
print(f"Earlier EMA: {trend['earlier_ema']:.2f} km/month")
print(f"Change: {trend['change_percentage']:.1f}%")
print(f"Strength: {trend['trend_strength']}")
```

## Configuration

### EMA Parameters

```python
# Customize EMA calculation
calculator = EMACalculator(
    n_periods=8,        # Use 8-period EMA instead of 6
    min_services=3      # Require at least 3 services for EMA
)
```

### Settings Integration

```python
from src.config.settings import get_settings

settings = get_settings()

calculator = EMACalculator(
    n_periods=settings.ema_periods,
    min_services=settings.min_services_for_ema
)
```

## Data Requirements

### Service History Format

The service history DataFrame must contain these columns:

- `service_date`: DateTime column with service dates
- `odometer_reading`: Numeric column with odometer readings in kilometers

### Data Validation

```python
# Validate service history before calculation
is_valid, message = calculator.validate_service_history(service_history)

if not is_valid:
    print(f"Invalid data: {message}")
else:
    result = calculator.calculate_ema_with_result(service_history)
```

### Common Data Issues

1. **Missing Columns**: Ensure both `service_date` and `odometer_reading` columns exist
2. **Duplicate Dates**: Remove duplicate service dates
3. **Negative Odometer**: Check for negative odometer readings
4. **Decreasing Odometer**: Ensure odometer readings are non-decreasing
5. **Invalid Dates**: Handle missing or invalid service dates

## Performance Considerations

### Large Datasets

```python
# For large datasets, use batch processing
vehicle_histories = [(vid, history) for vid, history in large_dataset.items()]
results = calculator.calculate_ema_for_vehicle_list(vehicle_histories)
```

### Memory Optimization

```python
# Process in chunks for very large datasets
chunk_size = 1000
for i in range(0, len(vehicle_histories), chunk_size):
    chunk = vehicle_histories[i:i+chunk_size]
    results = calculator.calculate_ema_for_vehicle_list(chunk)
    # Process results...
```

## Error Handling

### Common Exceptions

```python
try:
    result = calculator.calculate_ema_with_result(service_history, "VEH001")
except Exception as e:
    print(f"Error calculating EMA: {e}")
    # Handle error appropriately
```

### Graceful Degradation

```python
# The calculator handles errors gracefully
result = calculator.calculate_ema_with_result(invalid_data, "VEH001")

if result.calculation_method == "ERROR":
    print(f"Calculation failed: {result.metadata.get('error')}")
    # Use fallback logic
```

## Integration with ML Models

### Feature Engineering

```python
# EMA is used as a feature in ML models
def get_vehicle_features(vehicle_id: str, service_history: pd.DataFrame) -> Dict:
    """Extract features for ML model."""
    result = calculator.calculate_ema_with_result(service_history, vehicle_id)
    
    return {
        'ema_value': result.ema_value,
        'ema_category': result.ema_category,
        'usage_intensity': result.ema_value / 1000,  # Normalize
        'is_high_usage': result.ema_category == 'HIGH_USAGE',
        'is_medium_usage': result.ema_category == 'MEDIUM_USAGE',
        'is_low_usage': result.ema_category == 'LOW_USAGE'
    }
```

### Model Training Integration

```python
# Use EMA in model training
def prepare_training_data(vehicles_data: List[Dict]) -> pd.DataFrame:
    """Prepare training data with EMA features."""
    features = []
    
    for vehicle_data in vehicles_data:
        ema_result = calculator.calculate_ema_with_result(
            vehicle_data['service_history'], 
            vehicle_data['vehicle_id']
        )
        
        features.append({
            'vehicle_id': vehicle_data['vehicle_id'],
            'ema_value': ema_result.ema_value,
            'ema_category': ema_result.ema_category,
            'target': vehicle_data['target']
        })
    
    return pd.DataFrame(features)
```

## Monitoring and Logging

### Logging Configuration

```python
import logging

# Configure logging for EMA calculations
logger = logging.getLogger('ema_calculator')
logger.setLevel(logging.INFO)

# The calculator logs important events automatically
```

### Performance Monitoring

```python
import time

# Monitor calculation performance
start_time = time.time()
result = calculator.calculate_ema_with_result(service_history, "VEH001")
calculation_time = time.time() - start_time

logger.info(f"EMA calculation took {calculation_time:.3f} seconds")
```

## Best Practices

### 1. Data Quality

- Always validate service history before calculation
- Handle missing or invalid data gracefully
- Ensure odometer readings are consistent

### 2. Performance

- Use batch processing for multiple vehicles
- Cache results when possible
- Monitor calculation times

### 3. Error Handling

- Implement proper error handling
- Use structured result objects
- Log errors for debugging

### 4. Testing

- Test with various data scenarios
- Include edge cases in tests
- Validate calculation accuracy

## Examples

### Complete Workflow Example

```python
from src.models.ema_calculator import EMACalculator
import pandas as pd
from datetime import datetime, timedelta

def analyze_fleet_usage(vehicles_data: Dict[str, pd.DataFrame]) -> Dict:
    """Analyze usage patterns for an entire fleet."""
    
    # Initialize calculator
    calculator = EMACalculator()
    
    # Calculate EMA for all vehicles
    results = {}
    for vehicle_id, service_history in vehicles_data.items():
        result = calculator.calculate_ema_with_result(service_history, vehicle_id)
        results[vehicle_id] = result
    
    # Extract EMA values for statistics
    ema_values = {vid: result.ema_value for vid, result in results.items()}
    
    # Calculate fleet statistics
    stats = calculator.get_usage_statistics_structured(ema_values)
    
    # Detect anomalies
    anomalies = calculator.detect_usage_anomalies(ema_values)
    
    # Calculate percentiles
    percentiles = calculator.get_ema_percentiles(ema_values)
    
    return {
        'individual_results': {vid: result.to_dict() for vid, result in results.items()},
        'fleet_statistics': stats.to_dict(),
        'anomalies': anomalies,
        'percentiles': percentiles
    }

# Usage
fleet_data = {
    'VEH001': service_history_1,
    'VEH002': service_history_2,
    'VEH003': service_history_3
}

analysis = analyze_fleet_usage(fleet_data)
print(f"Fleet Analysis: {analysis['fleet_statistics']}")
```

This guide provides comprehensive documentation for using the EMA Calculator in the AI Parts Recommendation System. The calculator is designed to be robust, efficient, and easy to integrate with other system components.
