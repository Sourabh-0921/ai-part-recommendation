"""
Unit tests for EMA Calculator.

Tests the Exponential Moving Average calculation functionality
as par requirement.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.models.ema_calculator import EMACalculator


class TestEMACalculator:
    """Test suite for EMACalculator class."""
    
    @pytest.fixture
    def calculator(self):
        """Create EMA calculator instance."""
        return EMACalculator(n_periods=6, min_services=2)
    
    @pytest.fixture
    def sample_service_history(self):
        """Create sample service history data."""
        dates = [
            datetime(2023, 1, 1),
            datetime(2023, 2, 1),
            datetime(2023, 3, 1),
            datetime(2023, 4, 1),
            datetime(2023, 5, 1),
            datetime(2023, 6, 1)
        ]
        odometer_readings = [1000, 2500, 4000, 5500, 7000, 8500]
        
        return pd.DataFrame({
            'service_date': dates,
            'odometer_reading': odometer_readings
        })
    
    @pytest.fixture
    def high_usage_history(self):
        """Create high usage service history."""
        dates = [
            datetime(2023, 1, 1),
            datetime(2023, 2, 1),
            datetime(2023, 3, 1),
            datetime(2023, 4, 1)
        ]
        odometer_readings = [1000, 4000, 7000, 10000]  # 3000 km per month
        
        return pd.DataFrame({
            'service_date': dates,
            'odometer_reading': odometer_readings
        })
    
    @pytest.fixture
    def low_usage_history(self):
        """Create low usage service history."""
        dates = [
            datetime(2023, 1, 1),
            datetime(2023, 4, 1),
            datetime(2023, 7, 1),
            datetime(2023, 10, 1)
        ]
        odometer_readings = [1000, 2000, 3000, 4000]  # 1000 km per 3 months = 333 km/month
        
        return pd.DataFrame({
            'service_date': dates,
            'odometer_reading': odometer_readings
        })
    
    def test_initialization(self, calculator):
        """Test EMA calculator initialization."""
        assert calculator.n_periods == 6
        assert calculator.min_services == 2
    
    def test_calculate_ema_success(self, calculator, sample_service_history):
        """Test successful EMA calculation."""
        ema_value, method, metadata = calculator.calculate_ema(
            sample_service_history, 
            vehicle_id="TEST123"
        )
        
        assert ema_value > 0
        assert method == "EMA"
        assert "valid_records" in metadata
        assert "total_services" in metadata
        assert metadata["valid_records"] == 5  # 6 services - 1 (first has no diff)
        assert metadata["total_services"] == 6
    
    def test_calculate_ema_insufficient_data(self, calculator):
        """Test EMA calculation with insufficient data."""
        # Empty DataFrame
        empty_df = pd.DataFrame(columns=['service_date', 'odometer_reading'])
        ema_value, method, metadata = calculator.calculate_ema(empty_df)
        
        assert ema_value == 0.0
        assert method == "INSUFFICIENT_DATA"
        assert "reason" in metadata
        
        # Single service
        single_service = pd.DataFrame({
            'service_date': [datetime(2023, 1, 1)],
            'odometer_reading': [1000]
        })
        ema_value, method, metadata = calculator.calculate_ema(single_service)
        
        assert ema_value == 0.0
        assert method == "INSUFFICIENT_DATA"
    
    def test_calculate_ema_simple_average_fallback(self, calculator):
        """Test fallback to simple average when insufficient data for EMA."""
        # Two services - should use simple average
        two_services = pd.DataFrame({
            'service_date': [datetime(2023, 1, 1), datetime(2023, 2, 1)],
            'odometer_reading': [1000, 2500]
        })
        
        ema_value, method, metadata = calculator.calculate_ema(two_services)
        
        assert ema_value > 0
        assert method == "SIMPLE_AVERAGE"
        assert "simple_average" in metadata
    
    def test_categorize_ema(self, calculator):
        """Test EMA categorization."""
        # High usage
        assert calculator.categorize_ema(900) == "HIGH_USAGE"
        assert calculator.categorize_ema(800.1) == "HIGH_USAGE"
        
        # Medium usage
        assert calculator.categorize_ema(600) == "MEDIUM_USAGE"
        assert calculator.categorize_ema(400.1) == "MEDIUM_USAGE"
        assert calculator.categorize_ema(800) == "MEDIUM_USAGE"
        
        # Low usage
        assert calculator.categorize_ema(300) == "LOW_USAGE"
        assert calculator.categorize_ema(400) == "LOW_USAGE"
        assert calculator.categorize_ema(0) == "LOW_USAGE"
    
    def test_calculate_ema_batch(self, calculator, sample_service_history):
        """Test batch EMA calculation."""
        service_histories = {
            "VEH001": sample_service_history,
            "VEH002": sample_service_history.copy()
        }
        
        results = calculator.calculate_ema_batch(service_histories)
        
        assert len(results) == 2
        assert "VEH001" in results
        assert "VEH002" in results
        
        for vehicle_id, result in results.items():
            assert "ema_value" in result
            assert "ema_category" in result
            assert "calculation_method" in result
            assert "metadata" in result
            assert "calculated_at" in result
    
    def test_get_usage_statistics(self, calculator):
        """Test usage statistics calculation."""
        ema_values = {
            "VEH001": 900,  # High usage
            "VEH002": 600,  # Medium usage
            "VEH003": 300,  # Low usage
            "VEH004": 0,    # Invalid
            "VEH005": 1200  # High usage
        }
        
        stats = calculator.get_usage_statistics(ema_values)
        
        assert stats["total_vehicles"] == 5
        assert stats["valid_vehicles"] == 4
        assert stats["high_usage_count"] == 2
        assert stats["medium_usage_count"] == 1
        assert stats["low_usage_count"] == 1
        assert stats["high_usage_percentage"] == 50.0
        assert stats["medium_usage_percentage"] == 25.0
        assert stats["low_usage_percentage"] == 25.0
    
    def test_validate_service_history(self, calculator):
        """Test service history validation."""
        # Valid history
        valid_history = pd.DataFrame({
            'service_date': [datetime(2023, 1, 1), datetime(2023, 2, 1)],
            'odometer_reading': [1000, 2000]
        })
        
        is_valid, message = calculator.validate_service_history(valid_history)
        assert is_valid
        assert message == "Valid service history"
        
        # Empty history
        empty_history = pd.DataFrame(columns=['service_date', 'odometer_reading'])
        is_valid, message = calculator.validate_service_history(empty_history)
        assert not is_valid
        assert "empty" in message.lower()
        
        # Missing columns
        invalid_history = pd.DataFrame({
            'service_date': [datetime(2023, 1, 1)]
        })
        is_valid, message = calculator.validate_service_history(invalid_history)
        assert not is_valid
        assert "missing" in message.lower()
        
        # Duplicate dates
        duplicate_history = pd.DataFrame({
            'service_date': [datetime(2023, 1, 1), datetime(2023, 1, 1)],
            'odometer_reading': [1000, 2000]
        })
        is_valid, message = calculator.validate_service_history(duplicate_history)
        assert not is_valid
        assert "duplicate" in message.lower()
        
        # Negative odometer
        negative_history = pd.DataFrame({
            'service_date': [datetime(2023, 1, 1)],
            'odometer_reading': [-1000]
        })
        is_valid, message = calculator.validate_service_history(negative_history)
        assert not is_valid
        assert "negative" in message.lower()
        
        # Decreasing odometer
        decreasing_history = pd.DataFrame({
            'service_date': [datetime(2023, 1, 1), datetime(2023, 2, 1)],
            'odometer_reading': [2000, 1000]
        })
        is_valid, message = calculator.validate_service_history(decreasing_history)
        assert not is_valid
        assert "decreasing" in message.lower()
    
    def test_get_ema_trend(self, calculator, sample_service_history):
        """Test EMA trend analysis."""
        trend = calculator.get_ema_trend(sample_service_history, window=3)
        
        assert "trend" in trend
        assert "recent_ema" in trend
        assert "earlier_ema" in trend
        assert "change_percentage" in trend
        assert "trend_strength" in trend
        
        # Test with insufficient data
        single_service = pd.DataFrame({
            'service_date': [datetime(2023, 1, 1)],
            'odometer_reading': [1000]
        })
        
        trend = calculator.get_ema_trend(single_service, window=3)
        assert trend["trend"] == "INSUFFICIENT_DATA"
    
    def test_high_usage_scenario(self, calculator, high_usage_history):
        """Test EMA calculation for high usage vehicle."""
        ema_value, method, metadata = calculator.calculate_ema(high_usage_history)
        category = calculator.categorize_ema(ema_value)
        
        assert ema_value > 800  # Should be high usage
        assert category == "HIGH_USAGE"
        assert method in ["EMA", "SIMPLE_AVERAGE"]
    
    def test_low_usage_scenario(self, calculator, low_usage_history):
        """Test EMA calculation for low usage vehicle."""
        ema_value, method, metadata = calculator.calculate_ema(low_usage_history)
        category = calculator.categorize_ema(ema_value)
        
        assert ema_value < 400  # Should be low usage
        assert category == "LOW_USAGE"
        assert method in ["EMA", "SIMPLE_AVERAGE"]
    
    def test_error_handling(self, calculator):
        """Test error handling in EMA calculation."""
        # Invalid DataFrame structure
        invalid_df = pd.DataFrame({
            'wrong_column': [1, 2, 3]
        })
        
        ema_value, method, metadata = calculator.calculate_ema(invalid_df)
        
        assert ema_value == 0.0
        assert method == "ERROR"
        assert "error" in metadata
    
    def test_ema_calculation_accuracy(self, calculator):
        """Test EMA calculation mathematical accuracy."""
        # Create predictable data
        dates = [
            datetime(2023, 1, 1),
            datetime(2023, 2, 1),
            datetime(2023, 3, 1),
            datetime(2023, 4, 1),
            datetime(2023, 5, 1),
            datetime(2023, 6, 1),
            datetime(2023, 7, 1)
        ]
        odometer_readings = [1000, 2000, 3000, 4000, 5000, 6000, 7000]  # 1000 km/month
        
        service_history = pd.DataFrame({
            'service_date': dates,
            'odometer_reading': odometer_readings
        })
        
        ema_value, method, metadata = calculator.calculate_ema(service_history)
        
        # Should be close to 1000 km/month
        assert 900 <= ema_value <= 1100
        assert method == "EMA"
    
    def test_metadata_completeness(self, calculator, sample_service_history):
        """Test that metadata contains all expected fields."""
        ema_value, method, metadata = calculator.calculate_ema(sample_service_history)
        
        expected_fields = [
            "valid_records", "total_services", "smoothing_factor",
            "ema_series_length", "simple_average", "std_deviation",
            "min_usage", "max_usage", "last_service_date", "first_service_date"
        ]
        
        for field in expected_fields:
            assert field in metadata, f"Missing field: {field}"
    
    def test_edge_cases(self, calculator):
        """Test edge cases in EMA calculation."""
        # Same odometer readings (no usage)
        no_usage = pd.DataFrame({
            'service_date': [datetime(2023, 1, 1), datetime(2023, 2, 1)],
            'odometer_reading': [1000, 1000]
        })
        
        ema_value, method, metadata = calculator.calculate_ema(no_usage)
        assert ema_value == 0.0
        assert method == "INSUFFICIENT_DATA"
        
        # Very large time gaps
        large_gaps = pd.DataFrame({
            'service_date': [
                datetime(2023, 1, 1),
                datetime(2023, 1, 2),  # 1 day gap
                datetime(2024, 1, 1)   # 1 year gap
            ],
            'odometer_reading': [1000, 1100, 2000]
        })
        
        ema_value, method, metadata = calculator.calculate_ema(large_gaps)
        # Should handle large gaps gracefully
        assert ema_value >= 0
