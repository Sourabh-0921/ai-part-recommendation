"""
EMA Calculator Usage Examples

This script demonstrates various ways to use the EMA Calculator
for vehicle usage pattern analysis in the AI Parts Recommendation System.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.models.ema_calculator import EMACalculator, EMAResult, UsageStatistics


def create_sample_service_history(vehicle_id: str, usage_pattern: str = "normal") -> pd.DataFrame:
    """
    Create sample service history data for testing.
    
    Args:
        vehicle_id: Vehicle identifier
        usage_pattern: Usage pattern ("normal", "high", "low", "increasing", "decreasing")
        
    Returns:
        DataFrame with service history
    """
    base_date = datetime(2023, 1, 1)
    service_data = []
    odometer = 1000
    
    if usage_pattern == "normal":
        # Normal usage: ~1500 km/month
        for i in range(6):
            service_data.append({
                'service_date': base_date + timedelta(days=i*30),
                'odometer_reading': odometer
            })
            odometer += 1500
    
    elif usage_pattern == "high":
        # High usage: ~3000 km/month
        for i in range(6):
            service_data.append({
                'service_date': base_date + timedelta(days=i*30),
                'odometer_reading': odometer
            })
            odometer += 3000
    
    elif usage_pattern == "low":
        # Low usage: ~500 km/month
        for i in range(6):
            service_data.append({
                'service_date': base_date + timedelta(days=i*30),
                'odometer_reading': odometer
            })
            odometer += 500
    
    elif usage_pattern == "increasing":
        # Increasing usage pattern
        for i in range(6):
            service_data.append({
                'service_date': base_date + timedelta(days=i*30),
                'odometer_reading': odometer
            })
            odometer += 1000 + (i * 500)  # Increasing usage
    
    elif usage_pattern == "decreasing":
        # Decreasing usage pattern
        for i in range(6):
            service_data.append({
                'service_date': base_date + timedelta(days=i*30),
                'odometer_reading': odometer
            })
            odometer += 3000 - (i * 400)  # Decreasing usage
    
    return pd.DataFrame(service_data)


def example_basic_ema_calculation():
    """Example 1: Basic EMA calculation for a single vehicle."""
    print("=" * 60)
    print("EXAMPLE 1: Basic EMA Calculation")
    print("=" * 60)
    
    # Create calculator
    calculator = EMACalculator(n_periods=6, min_services=2)
    
    # Create sample data
    service_history = create_sample_service_history("VEH001", "normal")
    
    print(f"Service History for VEH001:")
    print(service_history)
    print()
    
    # Calculate EMA
    result = calculator.calculate_ema_with_result(service_history, "VEH001")
    
    print(f"EMA Calculation Results:")
    print(f"  Vehicle ID: VEH001")
    print(f"  EMA Value: {result.ema_value:.2f} km/month")
    print(f"  Category: {result.ema_category}")
    print(f"  Method: {result.calculation_method}")
    print(f"  Valid Records: {result.metadata['valid_records']}")
    print(f"  Total Services: {result.metadata['total_services']}")
    print()


def example_usage_categories():
    """Example 2: Demonstrate different usage categories."""
    print("=" * 60)
    print("EXAMPLE 2: Usage Categories")
    print("=" * 60)
    
    calculator = EMACalculator()
    
    # Create vehicles with different usage patterns
    vehicles = {
        "HIGH_USAGE_VEH": create_sample_service_history("HIGH_USAGE_VEH", "high"),
        "MEDIUM_USAGE_VEH": create_sample_service_history("MEDIUM_USAGE_VEH", "normal"),
        "LOW_USAGE_VEH": create_sample_service_history("LOW_USAGE_VEH", "low")
    }
    
    print("Usage Category Analysis:")
    print("-" * 40)
    
    for vehicle_id, service_history in vehicles.items():
        result = calculator.calculate_ema_with_result(service_history, vehicle_id)
        
        print(f"{vehicle_id}:")
        print(f"  EMA: {result.ema_value:.2f} km/month")
        print(f"  Category: {result.ema_category}")
        print(f"  Method: {result.calculation_method}")
        print()


def example_batch_processing():
    """Example 3: Batch processing for multiple vehicles."""
    print("=" * 60)
    print("EXAMPLE 3: Batch Processing")
    print("=" * 60)
    
    calculator = EMACalculator()
    
    # Create multiple vehicles with different patterns
    vehicle_histories = [
        ("VEH001", create_sample_service_history("VEH001", "normal")),
        ("VEH002", create_sample_service_history("VEH002", "high")),
        ("VEH003", create_sample_service_history("VEH003", "low")),
        ("VEH004", create_sample_service_history("VEH004", "increasing")),
        ("VEH005", create_sample_service_history("VEH005", "decreasing"))
    ]
    
    # Process all vehicles
    results = calculator.calculate_ema_for_vehicle_list(vehicle_histories)
    
    print("Batch Processing Results:")
    print("-" * 40)
    
    for vehicle_id, result in results.items():
        print(f"{vehicle_id}: {result.ema_value:.2f} km/month ({result.ema_category})")
    
    print()


def example_usage_statistics():
    """Example 4: Fleet usage statistics."""
    print("=" * 60)
    print("EXAMPLE 4: Fleet Usage Statistics")
    print("=" * 60)
    
    calculator = EMACalculator()
    
    # Create a fleet with various usage patterns
    fleet_data = {
        "VEH001": 1200,  # High usage
        "VEH002": 600,   # Medium usage
        "VEH003": 300,   # Low usage
        "VEH004": 900,   # High usage
        "VEH005": 500,   # Medium usage
        "VEH006": 800,   # High usage
        "VEH007": 400,   # Medium usage
        "VEH008": 200,   # Low usage
        "VEH009": 1100,  # High usage
        "VEH010": 550    # Medium usage
    }
    
    # Calculate statistics
    stats = calculator.get_usage_statistics_structured(fleet_data)
    
    print("Fleet Usage Statistics:")
    print("-" * 40)
    print(f"Total Vehicles: {stats.total_vehicles}")
    print(f"Valid Vehicles: {stats.valid_vehicles}")
    print()
    print("Usage Distribution:")
    print(f"  High Usage: {stats.high_usage_count} ({stats.high_usage_percentage:.1f}%)")
    print(f"  Medium Usage: {stats.medium_usage_count} ({stats.medium_usage_percentage:.1f}%)")
    print(f"  Low Usage: {stats.low_usage_count} ({stats.low_usage_percentage:.1f}%)")
    print()
    print("Statistical Measures:")
    print(f"  Average EMA: {stats.average_ema:.2f} km/month")
    print(f"  Median EMA: {stats.median_ema:.2f} km/month")
    print(f"  Standard Deviation: {stats.std_ema:.2f} km/month")
    print(f"  Min EMA: {stats.min_ema:.2f} km/month")
    print(f"  Max EMA: {stats.max_ema:.2f} km/month")
    print()


def example_anomaly_detection():
    """Example 5: Anomaly detection in usage patterns."""
    print("=" * 60)
    print("EXAMPLE 5: Anomaly Detection")
    print("=" * 60)
    
    calculator = EMACalculator()
    
    # Create data with some anomalies
    fleet_data = {
        "VEH001": 500,   # Normal
        "VEH002": 550,   # Normal
        "VEH003": 480,   # Normal
        "VEH004": 520,   # Normal
        "VEH005": 2000,  # Anomaly (very high usage)
        "VEH006": 510,   # Normal
        "VEH007": 490,   # Normal
        "VEH008": 50,    # Anomaly (very low usage)
        "VEH009": 530,   # Normal
        "VEH010": 540    # Normal
    }
    
    # Detect anomalies
    anomalies = calculator.detect_usage_anomalies(fleet_data, threshold=2.0)
    
    print("Anomaly Detection Results:")
    print("-" * 40)
    print(f"Total Anomalies Detected: {anomalies['total_anomalies']}")
    print(f"Detection Threshold: {anomalies['threshold']}")
    print(f"Fleet Mean: {anomalies['mean']:.2f} km/month")
    print(f"Fleet Std Dev: {anomalies['std']:.2f} km/month")
    print()
    
    if anomalies['anomalies']:
        print("Detected Anomalies:")
        for anomaly in anomalies['anomalies']:
            print(f"  {anomaly['vehicle_id']}: {anomaly['ema_value']} km/month")
            print(f"    Z-score: {anomaly['z_score']:.2f}")
            print(f"    Severity: {anomaly['severity']}")
    else:
        print("No anomalies detected.")
    print()


def example_trend_analysis():
    """Example 6: Trend analysis for usage patterns."""
    print("=" * 60)
    print("EXAMPLE 6: Trend Analysis")
    print("=" * 60)
    
    calculator = EMACalculator()
    
    # Analyze different trend patterns
    patterns = ["increasing", "decreasing", "normal"]
    
    for pattern in patterns:
        service_history = create_sample_service_history(f"TREND_{pattern.upper()}", pattern)
        trend = calculator.get_ema_trend(service_history, window=3)
        
        print(f"Trend Analysis for {pattern.upper()} pattern:")
        print(f"  Trend Direction: {trend['trend']}")
        print(f"  Recent EMA: {trend['recent_ema']:.2f} km/month")
        print(f"  Earlier EMA: {trend['earlier_ema']:.2f} km/month")
        print(f"  Change: {trend['change_percentage']:.1f}%")
        print(f"  Trend Strength: {trend['trend_strength']}")
        print()


def example_percentiles():
    """Example 7: Percentile analysis."""
    print("=" * 60)
    print("EXAMPLE 7: Percentile Analysis")
    print("=" * 60)
    
    calculator = EMACalculator()
    
    # Create a large dataset for percentile analysis
    np.random.seed(42)  # For reproducible results
    fleet_data = {}
    
    # Generate realistic usage data
    for i in range(100):
        # Generate EMA values with realistic distribution
        ema_value = np.random.lognormal(mean=6.0, sigma=0.5)  # Log-normal distribution
        fleet_data[f"VEH{i:03d}"] = ema_value
    
    # Calculate percentiles
    percentiles = calculator.get_ema_percentiles(fleet_data)
    
    print("Fleet Usage Percentiles:")
    print("-" * 40)
    for percentile, value in percentiles.items():
        print(f"{percentile}: {value:.2f} km/month")
    print()


def example_data_validation():
    """Example 8: Data validation examples."""
    print("=" * 60)
    print("EXAMPLE 8: Data Validation")
    print("=" * 60)
    
    calculator = EMACalculator()
    
    # Test cases for data validation
    test_cases = [
        {
            "name": "Valid Data",
            "data": pd.DataFrame({
                'service_date': [datetime(2023, 1, 1), datetime(2023, 2, 1)],
                'odometer_reading': [1000, 2000]
            })
        },
        {
            "name": "Empty Data",
            "data": pd.DataFrame(columns=['service_date', 'odometer_reading'])
        },
        {
            "name": "Missing Columns",
            "data": pd.DataFrame({
                'service_date': [datetime(2023, 1, 1)]
            })
        },
        {
            "name": "Duplicate Dates",
            "data": pd.DataFrame({
                'service_date': [datetime(2023, 1, 1), datetime(2023, 1, 1)],
                'odometer_reading': [1000, 2000]
            })
        },
        {
            "name": "Negative Odometer",
            "data": pd.DataFrame({
                'service_date': [datetime(2023, 1, 1)],
                'odometer_reading': [-1000]
            })
        },
        {
            "name": "Decreasing Odometer",
            "data": pd.DataFrame({
                'service_date': [datetime(2023, 1, 1), datetime(2023, 2, 1)],
                'odometer_reading': [2000, 1000]
            })
        }
    ]
    
    print("Data Validation Test Results:")
    print("-" * 40)
    
    for test_case in test_cases:
        is_valid, message = calculator.validate_service_history(test_case["data"])
        status = "✓ VALID" if is_valid else "✗ INVALID"
        print(f"{test_case['name']}: {status}")
        if not is_valid:
            print(f"  Reason: {message}")
    print()


def main():
    """Run all examples."""
    print("EMA Calculator Usage Examples")
    print("=" * 60)
    print("This script demonstrates various features of the EMA Calculator")
    print("for vehicle usage pattern analysis.")
    print()
    
    # Run all examples
    example_basic_ema_calculation()
    example_usage_categories()
    example_batch_processing()
    example_usage_statistics()
    example_anomaly_detection()
    example_trend_analysis()
    example_percentiles()
    example_data_validation()
    
    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
