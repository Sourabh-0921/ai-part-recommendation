"""
Exponential Moving Average (EMA) calculator for vehicle usage patterns.

This module implements EMA calculation as specified in the requirement document file
to analyze vehicle usage patterns and categorize them into HIGH_USAGE,
MEDIUM_USAGE, and LOW_USAGE categories.

The EMA calculation is a key feature for the ML models as it provides
insight into vehicle usage patterns that affect parts replacement timing.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EMAResult:
    """Result of EMA calculation with metadata."""
    ema_value: float
    ema_category: str
    calculation_method: str
    metadata: Dict[str, Any]
    calculated_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'ema_value': self.ema_value,
            'ema_category': self.ema_category,
            'calculation_method': self.calculation_method,
            'metadata': self.metadata,
            'calculated_at': self.calculated_at.isoformat()
        }


@dataclass
class UsageStatistics:
    """Usage statistics for a collection of vehicles."""
    total_vehicles: int
    valid_vehicles: int
    high_usage_count: int
    medium_usage_count: int
    low_usage_count: int
    high_usage_percentage: float
    medium_usage_percentage: float
    low_usage_percentage: float
    average_ema: float
    median_ema: float
    std_ema: float
    min_ema: float
    max_ema: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'total_vehicles': self.total_vehicles,
            'valid_vehicles': self.valid_vehicles,
            'high_usage_count': self.high_usage_count,
            'medium_usage_count': self.medium_usage_count,
            'low_usage_count': self.low_usage_count,
            'high_usage_percentage': self.high_usage_percentage,
            'medium_usage_percentage': self.medium_usage_percentage,
            'low_usage_percentage': self.low_usage_percentage,
            'average_ema': self.average_ema,
            'median_ema': self.median_ema,
            'std_ema': self.std_ema,
            'min_ema': self.min_ema,
            'max_ema': self.max_ema
        }


class EMACalculator:
    """
    Calculator for Exponential Moving Average of vehicle usage.
    
    This class calculates EMA values for vehicle usage patterns based on
    service history data, which is used as a key feature in the ML models.
    """
    
    def __init__(self, n_periods: int = 6, min_services: int = 2):
        """
        Initialize the EMA calculator.
        
        Args:
            n_periods: Number of periods for EMA calculation (default: 6)
            min_services: Minimum services required for EMA (default: 2)
        """
        self.n_periods = n_periods
        self.min_services = min_services
        
        logger.info(f"Initialized EMA calculator with {n_periods} periods, min {min_services} services")
    
    def calculate_ema(
        self,
        service_history: pd.DataFrame,
        vehicle_id: str = None
    ) -> Tuple[float, str, Dict[str, Any]]:
        """
        Calculate Exponential Moving Average for vehicle usage.
        
        Args:
            service_history: DataFrame with columns [service_date, odometer_reading]
            vehicle_id: Vehicle identifier for logging
            
        Returns:
            Tuple of (ema_value in km/month, calculation_method, metadata)
        """
        try:
            if vehicle_id:
                logger.debug(f"Calculating EMA for vehicle {vehicle_id}")
            
            # Validate input data
            if service_history.empty:
                return 0.0, "INSUFFICIENT_DATA", {"reason": "No service history"}
            
            required_columns = ['service_date', 'odometer_reading']
            missing_columns = [col for col in required_columns if col not in service_history.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Sort by date
            service_history = service_history.sort_values('service_date').copy()
            
            # Calculate distances and time gaps
            service_history['distance_km'] = service_history['odometer_reading'].diff()
            service_history['days_gap'] = service_history['service_date'].diff().dt.days
            service_history['months_gap'] = service_history['days_gap'] / 30.44  # Average days per month
            service_history['km_per_month'] = service_history['distance_km'] / service_history['months_gap']
            
            # Remove first row (NaN values) and invalid entries
            valid_data = service_history.dropna(subset=['km_per_month'])
            valid_data = valid_data[valid_data['km_per_month'] > 0]
            
            if len(valid_data) < self.min_services:
                # Use simple average if insufficient data for EMA
                if len(valid_data) == 0:
                    return 0.0, "INSUFFICIENT_DATA", {
                        "reason": "No valid service intervals",
                        "valid_records": 0
                    }
                
                ema_value = valid_data['km_per_month'].mean()
                return float(ema_value), "SIMPLE_AVERAGE", {
                    "reason": f"Insufficient data for EMA (need {self.min_services}, have {len(valid_data)})",
                    "valid_records": len(valid_data),
                    "simple_average": ema_value
                }
            
            # Calculate EMA
            smoothing_factor = 2 / (self.n_periods + 1)
            ema_series = valid_data['km_per_month'].ewm(
                alpha=smoothing_factor,
                adjust=False
            ).mean()
            
            ema_value = ema_series.iloc[-1]
            
            # Calculate additional statistics
            metadata = {
                "valid_records": len(valid_data),
                "total_services": len(service_history),
                "smoothing_factor": smoothing_factor,
                "ema_series_length": len(ema_series),
                "simple_average": valid_data['km_per_month'].mean(),
                "std_deviation": valid_data['km_per_month'].std(),
                "min_usage": valid_data['km_per_month'].min(),
                "max_usage": valid_data['km_per_month'].max(),
                "last_service_date": service_history['service_date'].iloc[-1].isoformat(),
                "first_service_date": service_history['service_date'].iloc[0].isoformat()
            }
            
            logger.debug(f"EMA calculated: {ema_value:.2f} km/month for vehicle {vehicle_id}")
            return float(ema_value), "EMA", metadata
            
        except Exception as e:
            logger.error(f"Error calculating EMA for vehicle {vehicle_id}: {e}")
            return 0.0, "ERROR", {"error": str(e)}
    
    def categorize_ema(self, ema_value: float) -> str:
        """
        Categorize EMA value into usage patterns.
        
        Args:
            ema_value: EMA value in km/month
            
        Returns:
            Category string: HIGH_USAGE, MEDIUM_USAGE, or LOW_USAGE
        """
        if ema_value > 800:
            return "HIGH_USAGE"
        elif ema_value > 400:
            return "MEDIUM_USAGE"
        else:
            return "LOW_USAGE"
    
    def calculate_ema_batch(
        self,
        service_histories: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate EMA for multiple vehicles in batch.
        
        Args:
            service_histories: Dictionary mapping vehicle_id to service history DataFrame
            
        Returns:
            Dictionary mapping vehicle_id to EMA results
        """
        results = {}
        
        for vehicle_id, service_history in service_histories.items():
            try:
                ema_value, method, metadata = self.calculate_ema(service_history, vehicle_id)
                category = self.categorize_ema(ema_value)
                
                results[vehicle_id] = {
                    'ema_value': ema_value,
                    'ema_category': category,
                    'calculation_method': method,
                    'metadata': metadata,
                    'calculated_at': datetime.utcnow().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error calculating EMA for vehicle {vehicle_id}: {e}")
                results[vehicle_id] = {
                    'ema_value': 0.0,
                    'ema_category': 'LOW_USAGE',
                    'calculation_method': 'ERROR',
                    'metadata': {'error': str(e)},
                    'calculated_at': datetime.utcnow().isoformat()
                }
        
        logger.info(f"Calculated EMA for {len(results)} vehicles")
        return results
    
    def get_usage_statistics(self, ema_values: Dict[str, float]) -> Dict[str, Any]:
        """
        Get usage statistics for a collection of EMA values.
        
        Args:
            ema_values: Dictionary mapping vehicle_id to EMA value
            
        Returns:
            Dictionary containing usage statistics
        """
        if not ema_values:
            return {}
        
        values = list(ema_values.values())
        valid_values = [v for v in values if v > 0]
        
        if not valid_values:
            return {
                'total_vehicles': len(values),
                'valid_vehicles': 0,
                'high_usage_count': 0,
                'medium_usage_count': 0,
                'low_usage_count': 0
            }
        
        # Categorize usage
        high_usage = sum(1 for v in valid_values if v > 800)
        medium_usage = sum(1 for v in valid_values if 400 < v <= 800)
        low_usage = sum(1 for v in valid_values if v <= 400)
        
        return {
            'total_vehicles': len(values),
            'valid_vehicles': len(valid_values),
            'high_usage_count': high_usage,
            'medium_usage_count': medium_usage,
            'low_usage_count': low_usage,
            'high_usage_percentage': (high_usage / len(valid_values)) * 100,
            'medium_usage_percentage': (medium_usage / len(valid_values)) * 100,
            'low_usage_percentage': (low_usage / len(valid_values)) * 100,
            'average_ema': np.mean(valid_values),
            'median_ema': np.median(valid_values),
            'std_ema': np.std(valid_values),
            'min_ema': np.min(valid_values),
            'max_ema': np.max(valid_values)
        }
    
    def validate_service_history(self, service_history: pd.DataFrame) -> Tuple[bool, str]:
        """
        Validate service history data for EMA calculation.
        
        Args:
            service_history: Service history DataFrame
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if service_history.empty:
            return False, "Service history is empty"
        
        required_columns = ['service_date', 'odometer_reading']
        missing_columns = [col for col in required_columns if col not in service_history.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
        
        # Check for duplicate dates
        if service_history['service_date'].duplicated().any():
            return False, "Duplicate service dates found"
        
        # Check for negative odometer readings
        if (service_history['odometer_reading'] < 0).any():
            return False, "Negative odometer readings found"
        
        # Check for decreasing odometer readings
        sorted_history = service_history.sort_values('service_date')
        if (sorted_history['odometer_reading'].diff() < 0).any():
            return False, "Decreasing odometer readings found"
        
        return True, "Valid service history"
    
    def get_ema_trend(self, service_history: pd.DataFrame, window: int = 3) -> Dict[str, Any]:
        """
        Analyze EMA trend over time.
        
        Args:
            service_history: Service history DataFrame
            window: Window size for trend analysis
            
        Returns:
            Dictionary containing trend analysis
        """
        try:
            if len(service_history) < window:
                return {"trend": "INSUFFICIENT_DATA", "reason": f"Need at least {window} records"}
            
            # Calculate rolling EMA
            service_history = service_history.sort_values('service_date').copy()
            service_history['distance_km'] = service_history['odometer_reading'].diff()
            service_history['days_gap'] = service_history['service_date'].diff().dt.days
            service_history['months_gap'] = service_history['days_gap'] / 30.44
            service_history['km_per_month'] = service_history['distance_km'] / service_history['months_gap']
            
            valid_data = service_history.dropna(subset=['km_per_month'])
            valid_data = valid_data[valid_data['km_per_month'] > 0]
            
            if len(valid_data) < window:
                return {"trend": "INSUFFICIENT_DATA", "reason": f"Need at least {window} valid records"}
            
            # Calculate rolling EMA
            smoothing_factor = 2 / (self.n_periods + 1)
            rolling_ema = valid_data['km_per_month'].ewm(
                alpha=smoothing_factor,
                adjust=False
            ).mean()
            
            # Analyze trend
            recent_ema = rolling_ema.iloc[-window:].mean()
            earlier_ema = rolling_ema.iloc[:-window].mean() if len(rolling_ema) > window else rolling_ema.iloc[0]
            
            trend_direction = "STABLE"
            if recent_ema > earlier_ema * 1.1:  # 10% increase
                trend_direction = "INCREASING"
            elif recent_ema < earlier_ema * 0.9:  # 10% decrease
                trend_direction = "DECREASING"
            
            return {
                "trend": trend_direction,
                "recent_ema": float(recent_ema),
                "earlier_ema": float(earlier_ema),
                "change_percentage": float((recent_ema - earlier_ema) / earlier_ema * 100),
                "trend_strength": "STRONG" if abs(recent_ema - earlier_ema) / earlier_ema > 0.2 else "WEAK"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing EMA trend: {e}")
            return {"trend": "ERROR", "error": str(e)}
    
    def calculate_ema_with_result(self, service_history: pd.DataFrame, vehicle_id: str = None) -> EMAResult:
        """
        Calculate EMA and return structured result object.
        
        Args:
            service_history: DataFrame with columns [service_date, odometer_reading]
            vehicle_id: Vehicle identifier for logging
            
        Returns:
            EMAResult object with all calculation details
        """
        ema_value, method, metadata = self.calculate_ema(service_history, vehicle_id)
        category = self.categorize_ema(ema_value)
        
        return EMAResult(
            ema_value=ema_value,
            ema_category=category,
            calculation_method=method,
            metadata=metadata,
            calculated_at=datetime.utcnow()
        )
    
    def get_usage_statistics_structured(self, ema_values: Dict[str, float]) -> UsageStatistics:
        """
        Get usage statistics as structured object.
        
        Args:
            ema_values: Dictionary mapping vehicle_id to EMA value
            
        Returns:
            UsageStatistics object
        """
        if not ema_values:
            return UsageStatistics(
                total_vehicles=0, valid_vehicles=0, high_usage_count=0,
                medium_usage_count=0, low_usage_count=0,
                high_usage_percentage=0.0, medium_usage_percentage=0.0,
                low_usage_percentage=0.0, average_ema=0.0, median_ema=0.0,
                std_ema=0.0, min_ema=0.0, max_ema=0.0
            )
        
        values = list(ema_values.values())
        valid_values = [v for v in values if v > 0]
        
        if not valid_values:
            return UsageStatistics(
                total_vehicles=len(values), valid_vehicles=0, high_usage_count=0,
                medium_usage_count=0, low_usage_count=0,
                high_usage_percentage=0.0, medium_usage_percentage=0.0,
                low_usage_percentage=0.0, average_ema=0.0, median_ema=0.0,
                std_ema=0.0, min_ema=0.0, max_ema=0.0
            )
        
        # Categorize usage
        high_usage = sum(1 for v in valid_values if v > 800)
        medium_usage = sum(1 for v in valid_values if 400 < v <= 800)
        low_usage = sum(1 for v in valid_values if v <= 400)
        
        return UsageStatistics(
            total_vehicles=len(values),
            valid_vehicles=len(valid_values),
            high_usage_count=high_usage,
            medium_usage_count=medium_usage,
            low_usage_count=low_usage,
            high_usage_percentage=(high_usage / len(valid_values)) * 100,
            medium_usage_percentage=(medium_usage / len(valid_values)) * 100,
            low_usage_percentage=(low_usage / len(valid_values)) * 100,
            average_ema=float(np.mean(valid_values)),
            median_ema=float(np.median(valid_values)),
            std_ema=float(np.std(valid_values)),
            min_ema=float(np.min(valid_values)),
            max_ema=float(np.max(valid_values))
        )
    
    def calculate_ema_for_vehicle_list(self, vehicle_histories: List[Tuple[str, pd.DataFrame]]) -> Dict[str, EMAResult]:
        """
        Calculate EMA for a list of vehicles efficiently.
        
        Args:
            vehicle_histories: List of (vehicle_id, service_history) tuples
            
        Returns:
            Dictionary mapping vehicle_id to EMAResult
        """
        results = {}
        
        for vehicle_id, service_history in vehicle_histories:
            try:
                result = self.calculate_ema_with_result(service_history, vehicle_id)
                results[vehicle_id] = result
                
            except Exception as e:
                logger.error(f"Error calculating EMA for vehicle {vehicle_id}: {e}")
                results[vehicle_id] = EMAResult(
                    ema_value=0.0,
                    ema_category="LOW_USAGE",
                    calculation_method="ERROR",
                    metadata={"error": str(e)},
                    calculated_at=datetime.utcnow()
                )
        
        logger.info(f"Calculated EMA for {len(results)} vehicles")
        return results
    
    def get_ema_percentiles(self, ema_values: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate percentile distribution of EMA values.
        
        Args:
            ema_values: Dictionary mapping vehicle_id to EMA value
            
        Returns:
            Dictionary with percentile values
        """
        if not ema_values:
            return {}
        
        values = [v for v in ema_values.values() if v > 0]
        if not values:
            return {}
        
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        result = {}
        
        for p in percentiles:
            result[f"p{p}"] = float(np.percentile(values, p))
        
        return result
    
    def detect_usage_anomalies(self, ema_values: Dict[str, float], threshold: float = 2.0) -> Dict[str, Any]:
        """
        Detect usage anomalies using statistical methods.
        
        Args:
            ema_values: Dictionary mapping vehicle_id to EMA value
            threshold: Z-score threshold for anomaly detection
            
        Returns:
            Dictionary with anomaly analysis
        """
        if not ema_values:
            return {"anomalies": [], "total_anomalies": 0}
        
        values = [v for v in ema_values.values() if v > 0]
        if len(values) < 3:  # Need at least 3 values for meaningful statistics
            return {"anomalies": [], "total_anomalies": 0}
        
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if std_val == 0:
            return {"anomalies": [], "total_anomalies": 0}
        
        anomalies = []
        for vehicle_id, ema_value in ema_values.items():
            if ema_value > 0:
                z_score = abs(ema_value - mean_val) / std_val
                if z_score > threshold:
                    anomalies.append({
                        "vehicle_id": vehicle_id,
                        "ema_value": ema_value,
                        "z_score": z_score,
                        "severity": "HIGH" if z_score > 3.0 else "MEDIUM"
                    })
        
        return {
            "anomalies": anomalies,
            "total_anomalies": len(anomalies),
            "threshold": threshold,
            "mean": float(mean_val),
            "std": float(std_val)
        }
