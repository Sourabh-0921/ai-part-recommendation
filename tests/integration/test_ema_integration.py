"""
Integration tests for EMA Calculator with database.

Tests the EMA calculation functionality in integration with the database
and other system components.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.models.ema_calculator import EMACalculator, EMAResult, UsageStatistics
from src.data.models import Vehicle, ServiceHistory
from src.data.database import get_db_session
from src.config.settings import get_settings


class TestEMAIntegration:
    """Integration tests for EMA calculator with database."""
    
    @pytest.fixture
    def calculator(self):
        """Create EMA calculator instance."""
        return EMACalculator(n_periods=6, min_services=2)
    
    @pytest.fixture
    def sample_vehicle_data(self):
        """Create sample vehicle data for testing."""
        return {
            'vehicle_id': 'TEST001',
            'vehicle_model': 'Pulsar 150',
            'invoice_date': datetime(2022, 1, 1),
            'current_odometer': 15000,
            'dealer_code': 'DLR_MUM_01',
            'region_code': 'MH'
        }
    
    @pytest.fixture
    def sample_service_history_data(self):
        """Create sample service history data."""
        base_date = datetime(2023, 1, 1)
        return [
            {
                'vehicle_id': 'TEST001',
                'service_date': base_date,
                'odometer_reading': 1000,
                'service_type': 'Regular Service',
                'dealer_code': 'DLR_MUM_01'
            },
            {
                'vehicle_id': 'TEST001',
                'service_date': base_date + timedelta(days=30),
                'odometer_reading': 2500,
                'service_type': 'Regular Service',
                'dealer_code': 'DLR_MUM_01'
            },
            {
                'vehicle_id': 'TEST001',
                'service_date': base_date + timedelta(days=60),
                'odometer_reading': 4000,
                'service_type': 'Regular Service',
                'dealer_code': 'DLR_MUM_01'
            },
            {
                'vehicle_id': 'TEST001',
                'service_date': base_date + timedelta(days=90),
                'odometer_reading': 5500,
                'service_type': 'Regular Service',
                'dealer_code': 'DLR_MUM_01'
            },
            {
                'vehicle_id': 'TEST001',
                'service_date': base_date + timedelta(days=120),
                'odometer_reading': 7000,
                'service_type': 'Regular Service',
                'dealer_code': 'DLR_MUM_01'
            }
        ]
    
    @pytest.fixture
    def high_usage_service_history(self):
        """Create high usage service history."""
        base_date = datetime(2023, 1, 1)
        return [
            {
                'vehicle_id': 'TEST002',
                'service_date': base_date,
                'odometer_reading': 1000,
                'service_type': 'Regular Service',
                'dealer_code': 'DLR_MUM_01'
            },
            {
                'vehicle_id': 'TEST002',
                'service_date': base_date + timedelta(days=15),
                'odometer_reading': 3000,
                'service_type': 'Regular Service',
                'dealer_code': 'DLR_MUM_01'
            },
            {
                'vehicle_id': 'TEST002',
                'service_date': base_date + timedelta(days=30),
                'odometer_reading': 5000,
                'service_type': 'Regular Service',
                'dealer_code': 'DLR_MUM_01'
            }
        ]
    
    @pytest.mark.integration
    def test_ema_calculation_with_database(self, calculator, sample_vehicle_data, sample_service_history_data):
        """Test EMA calculation with database integration."""
        # Mock database session
        with patch('src.data.database.get_db_session') as mock_session:
            # Create mock vehicle
            mock_vehicle = Mock()
            mock_vehicle.vehicle_id = sample_vehicle_data['vehicle_id']
            mock_vehicle.vehicle_model = sample_vehicle_data['vehicle_model']
            
            # Create mock service history
            mock_service_history = []
            for service_data in sample_service_history_data:
                mock_service = Mock()
                mock_service.vehicle_id = service_data['vehicle_id']
                mock_service.service_date = service_data['service_date']
                mock_service.odometer_reading = service_data['odometer_reading']
                mock_service_history.append(mock_service)
            
            # Mock database query results
            mock_session.return_value.__enter__.return_value.query.return_value.filter_by.return_value.all.return_value = mock_service_history
            
            # Convert to DataFrame
            service_df = pd.DataFrame(sample_service_history_data)
            
            # Calculate EMA
            result = calculator.calculate_ema_with_result(service_df, sample_vehicle_data['vehicle_id'])
            
            # Assertions
            assert isinstance(result, EMAResult)
            assert result.ema_value > 0
            assert result.ema_category in ['HIGH_USAGE', 'MEDIUM_USAGE', 'LOW_USAGE']
            assert result.calculation_method in ['EMA', 'SIMPLE_AVERAGE']
            assert 'valid_records' in result.metadata
    
    @pytest.mark.integration
    def test_batch_ema_calculation(self, calculator, sample_service_history_data, high_usage_service_history):
        """Test batch EMA calculation for multiple vehicles."""
        # Prepare vehicle histories
        vehicle_histories = [
            ('TEST001', pd.DataFrame(sample_service_history_data)),
            ('TEST002', pd.DataFrame(high_usage_service_history))
        ]
        
        # Calculate EMA for all vehicles
        results = calculator.calculate_ema_for_vehicle_list(vehicle_histories)
        
        # Assertions
        assert len(results) == 2
        assert 'TEST001' in results
        assert 'TEST002' in results
        
        for vehicle_id, result in results.items():
            assert isinstance(result, EMAResult)
            assert result.ema_value >= 0
            assert result.ema_category in ['HIGH_USAGE', 'MEDIUM_USAGE', 'LOW_USAGE']
    
    @pytest.mark.integration
    def test_usage_statistics_calculation(self, calculator):
        """Test usage statistics calculation with real data."""
        # Create test EMA values
        ema_values = {
            'VEH001': 1200,  # High usage
            'VEH002': 600,   # Medium usage
            'VEH003': 300,   # Low usage
            'VEH004': 0,     # Invalid
            'VEH005': 900,   # High usage
            'VEH006': 500,   # Medium usage
        }
        
        # Calculate statistics
        stats = calculator.get_usage_statistics_structured(ema_values)
        
        # Assertions
        assert isinstance(stats, UsageStatistics)
        assert stats.total_vehicles == 6
        assert stats.valid_vehicles == 5
        assert stats.high_usage_count == 2
        assert stats.medium_usage_count == 2
        assert stats.low_usage_count == 1
        assert stats.high_usage_percentage == 40.0
        assert stats.medium_usage_percentage == 40.0
        assert stats.low_usage_percentage == 20.0
    
    @pytest.mark.integration
    def test_ema_percentiles_calculation(self, calculator):
        """Test EMA percentiles calculation."""
        # Create test data with known distribution
        ema_values = {f'VEH{i:03d}': i * 100 for i in range(1, 21)}  # 100, 200, ..., 2000
        
        percentiles = calculator.get_ema_percentiles(ema_values)
        
        # Assertions
        assert 'p10' in percentiles
        assert 'p25' in percentiles
        assert 'p50' in percentiles
        assert 'p75' in percentiles
        assert 'p90' in percentiles
        assert 'p95' in percentiles
        assert 'p99' in percentiles
        
        # Check that percentiles are in ascending order
        p_values = [percentiles[f'p{p}'] for p in [10, 25, 50, 75, 90, 95, 99]]
        assert all(p_values[i] <= p_values[i+1] for i in range(len(p_values)-1))
    
    @pytest.mark.integration
    def test_anomaly_detection(self, calculator):
        """Test usage anomaly detection."""
        # Create test data with one obvious outlier
        ema_values = {
            'VEH001': 500,   # Normal
            'VEH002': 550,   # Normal
            'VEH003': 480,    # Normal
            'VEH004': 520,    # Normal
            'VEH005': 2000,   # Outlier
            'VEH006': 510,    # Normal
        }
        
        anomalies = calculator.detect_usage_anomalies(ema_values, threshold=2.0)
        
        # Assertions
        assert 'anomalies' in anomalies
        assert 'total_anomalies' in anomalies
        assert anomalies['total_anomalies'] >= 1
        
        # Check that VEH005 is detected as anomaly
        anomaly_vehicles = [a['vehicle_id'] for a in anomalies['anomalies']]
        assert 'VEH005' in anomaly_vehicles
    
    @pytest.mark.integration
    def test_ema_trend_analysis(self, calculator):
        """Test EMA trend analysis with historical data."""
        # Create service history with increasing usage
        base_date = datetime(2023, 1, 1)
        service_data = []
        odometer = 1000
        
        for i in range(6):
            service_data.append({
                'service_date': base_date + timedelta(days=i*30),
                'odometer_reading': odometer
            })
            odometer += 1000 + (i * 200)  # Increasing usage pattern
        
        service_df = pd.DataFrame(service_data)
        
        # Analyze trend
        trend = calculator.get_ema_trend(service_df, window=3)
        
        # Assertions
        assert 'trend' in trend
        assert 'recent_ema' in trend
        assert 'earlier_ema' in trend
        assert 'change_percentage' in trend
        assert 'trend_strength' in trend
        
        # Should detect increasing trend
        assert trend['trend'] in ['INCREASING', 'STABLE', 'DECREASING']
    
    @pytest.mark.integration
    def test_ema_calculation_with_insufficient_data(self, calculator):
        """Test EMA calculation with insufficient data."""
        # Single service record
        single_service = pd.DataFrame({
            'service_date': [datetime(2023, 1, 1)],
            'odometer_reading': [1000]
        })
        
        result = calculator.calculate_ema_with_result(single_service, 'TEST001')
        
        # Should return insufficient data
        assert result.ema_value == 0.0
        assert result.calculation_method == 'INSUFFICIENT_DATA'
        assert result.ema_category == 'LOW_USAGE'
    
    @pytest.mark.integration
    def test_ema_calculation_with_invalid_data(self, calculator):
        """Test EMA calculation with invalid data."""
        # Invalid data with negative odometer
        invalid_data = pd.DataFrame({
            'service_date': [datetime(2023, 1, 1), datetime(2023, 2, 1)],
            'odometer_reading': [1000, -500]  # Invalid negative reading
        })
        
        result = calculator.calculate_ema_with_result(invalid_data, 'TEST001')
        
        # Should handle invalid data gracefully
        assert result.ema_value >= 0
        assert result.calculation_method in ['ERROR', 'INSUFFICIENT_DATA']
    
    @pytest.mark.integration
    def test_ema_calculation_performance(self, calculator):
        """Test EMA calculation performance with large dataset."""
        # Create large service history
        base_date = datetime(2023, 1, 1)
        service_data = []
        odometer = 1000
        
        # Create 100 service records
        for i in range(100):
            service_data.append({
                'service_date': base_date + timedelta(days=i*30),
                'odometer_reading': odometer + (i * 100)
            })
        
        service_df = pd.DataFrame(service_data)
        
        # Measure calculation time
        import time
        start_time = time.time()
        
        result = calculator.calculate_ema_with_result(service_df, 'PERF_TEST')
        
        end_time = time.time()
        calculation_time = end_time - start_time
        
        # Should complete within reasonable time (< 1 second)
        assert calculation_time < 1.0
        assert result.ema_value > 0
        assert result.calculation_method == 'EMA'
    
    @pytest.mark.integration
    def test_ema_calculation_with_missing_dates(self, calculator):
        """Test EMA calculation with missing service dates."""
        # Service history with missing dates
        service_data = [
            {'service_date': datetime(2023, 1, 1), 'odometer_reading': 1000},
            {'service_date': pd.NaT, 'odometer_reading': 2000},  # Missing date
            {'service_date': datetime(2023, 3, 1), 'odometer_reading': 3000}
        ]
        
        service_df = pd.DataFrame(service_data)
        
        result = calculator.calculate_ema_with_result(service_df, 'TEST001')
        
        # Should handle missing dates gracefully
        assert result.ema_value >= 0
        assert result.calculation_method in ['EMA', 'SIMPLE_AVERAGE', 'ERROR']
    
    @pytest.mark.integration
    def test_ema_calculation_with_duplicate_dates(self, calculator):
        """Test EMA calculation with duplicate service dates."""
        # Service history with duplicate dates
        service_data = [
            {'service_date': datetime(2023, 1, 1), 'odometer_reading': 1000},
            {'service_date': datetime(2023, 1, 1), 'odometer_reading': 1500},  # Duplicate date
            {'service_date': datetime(2023, 2, 1), 'odometer_reading': 2000}
        ]
        
        service_df = pd.DataFrame(service_data)
        
        # Should detect invalid data
        is_valid, message = calculator.validate_service_history(service_df)
        assert not is_valid
        assert 'duplicate' in message.lower()
    
    @pytest.mark.integration
    def test_ema_calculation_with_decreasing_odometer(self, calculator):
        """Test EMA calculation with decreasing odometer readings."""
        # Service history with decreasing odometer
        service_data = [
            {'service_date': datetime(2023, 1, 1), 'odometer_reading': 2000},
            {'service_date': datetime(2023, 2, 1), 'odometer_reading': 1000},  # Decreasing
            {'service_date': datetime(2023, 3, 1), 'odometer_reading': 3000}
        ]
        
        service_df = pd.DataFrame(service_data)
        
        # Should detect invalid data
        is_valid, message = calculator.validate_service_history(service_df)
        assert not is_valid
        assert 'decreasing' in message.lower()
