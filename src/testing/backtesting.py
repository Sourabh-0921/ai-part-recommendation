"""
Backtesting framework for historical validation.

This module implements comprehensive backtesting to validate model accuracy
on historical data before deployment.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import ast
import json

from ..models.validation import AccuracyMetrics, ValidationError
from ..utils import (
    parse_date,
    normalize_part_code,
    is_valid_part_code
)

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Container for backtest results."""
    vehicle_id: str
    service_date: datetime
    ml_recommendations: List[str]
    pm_recommendations: List[str]
    actual_replaced: List[str]
    ml_confidences: Dict[str, float]
    job_card_number: Optional[str] = None
    vehicle_model: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'vehicle_id': self.vehicle_id,
            'service_date': self.service_date.isoformat() if self.service_date else None,
            'ml_recommendations': self.ml_recommendations,
            'pm_recommendations': self.pm_recommendations,
            'actual_replaced': self.actual_replaced,
            'ml_confidences': self.ml_confidences,
            'job_card_number': self.job_card_number,
            'vehicle_model': self.vehicle_model,
        }


class PMScheduleEngine:
    """
    PM Schedule rule engine.
    
    Generates recommendations based on OEM preventive maintenance schedules.
    """
    
    def __init__(self, pm_rules: Dict[str, Any]):
        """
        Initialize PM schedule engine.
        
        Args:
            pm_rules: Dictionary mapping vehicle_model to PM schedule rules
        """
        self.pm_rules = pm_rules
        logger.info(f"Initialized PM schedule engine with {len(pm_rules)} vehicle models")
    
    def get_recommendations(
        self,
        vehicle_model: str,
        odometer_reading: float,
        days_since_invoice: int,
        service_type: str = "REGULAR"
    ) -> List[str]:
        """
        Get PM schedule recommendations.
        
        Args:
            vehicle_model: Vehicle model code
            odometer_reading: Current odometer reading
            days_since_invoice: Days since vehicle invoice
            service_type: Type of service (REGULAR, MAJOR, EMERGENCY)
            
        Returns:
            List of recommended part codes
        """
        try:
            if vehicle_model not in self.pm_rules:
                logger.warning(f"No PM rules found for model {vehicle_model}")
                return []
            
            rules = self.pm_rules[vehicle_model]
            recommendations = []
            
            # Get parts based on odometer intervals
            for part_code, part_rules in rules.get('parts', {}).items():
                interval_km = part_rules.get('interval_km')
                interval_months = part_rules.get('interval_months')
                
                should_replace = False
                
                # Check odometer-based replacement
                if interval_km:
                    km_intervals = int(odometer_reading / interval_km)
                    if km_intervals >= 1:
                        should_replace = True
                
                # Check time-based replacement
                if interval_months:
                    months_since_invoice = days_since_invoice / 30.0
                    if months_since_invoice >= interval_months:
                        should_replace = True
                
                # Service type filter
                if should_replace and part_rules.get('service_types'):
                    if service_type not in part_rules['service_types']:
                        should_replace = False
                
                if should_replace:
                    recommendations.append(part_code)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting PM recommendations: {e}")
            return []


class BacktestingFramework:
    """
    Framework for comprehensive historical validation.
    
    This class implements the complete backtesting pipeline:
    1. Generate predictions for test data
    2. Compare with PM schedule
    3. Calculate accuracy metrics
    4. Generate validation report
    """
    
    def __init__(
        self,
        ml_model: Optional[Any] = None,
        pm_schedule_rules: Optional[Dict[str, Any]] = None,
        confidence_threshold: float = 0.80,
        parts_list: Optional[List[str]] = None
    ):
        """
        Initialize backtesting framework.
        
        Args:
            ml_model: Trained ML model (or dict of part_code -> model)
            pm_schedule_rules: PM schedule rules by vehicle model
            confidence_threshold: Minimum confidence for recommendations (0-1)
            parts_list: List of part codes to test
        """
        if not 0 < confidence_threshold < 1:
            raise ValueError(f"Confidence threshold must be between 0 and 1, got {confidence_threshold}")
        
        self.ml_model = ml_model
        self.pm_engine = PMScheduleEngine(pm_rules=pm_schedule_rules or {})
        self.threshold = confidence_threshold
        self.parts_list = parts_list or []
        
        logger.info(
            f"Backtesting framework initialized with threshold {confidence_threshold}"
        )
    
    def run_backtest(
        self,
        test_data: pd.DataFrame,
        feature_columns: List[str],
        actual_parts_column: str = 'parts_replaced',
        vehicle_model_column: str = 'vehicle_model',
        odometer_column: str = 'odometer_reading',
        service_date_column: str = 'service_date'
    ) -> List[BacktestResult]:
        """
        Run complete backtest on test data.
        
        Args:
            test_data: Test dataset with features and actual outcomes
            feature_columns: List of feature column names
            actual_parts_column: Column name for actual parts replaced
            vehicle_model_column: Column name for vehicle model
            odometer_column: Column name for odometer reading
            service_date_column: Column name for service date
            
        Returns:
            List of BacktestResult objects for each test service
            
        Raises:
            ValueError: If test_data is invalid
            ValidationError: If backtesting fails
        """
        try:
            logger.info(f"Starting backtest on {len(test_data)} services")
            results = []
            
            for idx, row in test_data.iterrows():
                try:
                    # Extract features
                    features = row[feature_columns].to_frame().T
                    
                    # Generate ML predictions
                    ml_recs, ml_confs = self._generate_ml_predictions(features, row)
                    
                    # Get PM schedule recommendations
                    vehicle_model = row.get(vehicle_model_column, '')
                    odometer = row.get(odometer_column, 0)
                    service_date = row.get(service_date_column)
                    
                    # Calculate days since invoice using parse_date utility
                    days_since_invoice = 0
                    if 'invoice_date' in row:
                        invoice_date = row['invoice_date']
                        if pd.notna(service_date) and pd.notna(invoice_date):
                            # Use parse_date utility for consistent date parsing
                            parsed_service_date = parse_date(service_date)
                            parsed_invoice_date = parse_date(invoice_date)
                            
                            if parsed_service_date and parsed_invoice_date:
                                service_date = parsed_service_date
                                days_since_invoice = (parsed_service_date - parsed_invoice_date).days
                            elif isinstance(service_date, pd.Timestamp):
                                service_date = service_date.to_pydatetime()
                            else:
                                service_date = pd.to_datetime(service_date).to_pydatetime()
                    
                    pm_recs = self.pm_engine.get_recommendations(
                        vehicle_model=str(vehicle_model),
                        odometer_reading=float(odometer),
                        days_since_invoice=int(days_since_invoice),
                        service_type=str(row.get('service_type', 'REGULAR'))
                    )
                    
                    # Get actual parts replaced
                    actual = self._parse_parts_list(row.get(actual_parts_column))
                    
                    # Store result
                    result = BacktestResult(
                        vehicle_id=str(row.get('vehicle_id', '')),
                        service_date=service_date if isinstance(service_date, datetime) else pd.to_datetime(service_date),
                        ml_recommendations=ml_recs,
                        pm_recommendations=pm_recs,
                        actual_replaced=actual,
                        ml_confidences=ml_confs,
                        job_card_number=str(row.get('job_card_number', '')) if 'job_card_number' in row else None,
                        vehicle_model=str(vehicle_model)
                    )
                    results.append(result)
                    
                    # Progress logging
                    if (idx + 1) % 100 == 0:
                        logger.info(f"Processed {idx + 1}/{len(test_data)} services")
                        
                except Exception as e:
                    logger.error(
                        f"Error processing service {row.get('vehicle_id', 'unknown')}: {e}",
                        exc_info=True
                    )
                    continue
            
            logger.info(f"Backtest complete: {len(results)} results generated")
            return results
            
        except Exception as e:
            logger.error(f"Error in backtest run: {e}")
            raise ValidationError(f"Backtesting failed: {e}") from e
    
    def _generate_ml_predictions(
        self,
        features: pd.DataFrame,
        row: pd.Series
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Generate ML predictions with confidence scores.
        
        Args:
            features: Feature DataFrame (single row)
            row: Complete row with metadata
            
        Returns:
            Tuple of (recommended_parts, confidence_dict)
        """
        all_predictions = {}
        
        try:
            # Handle different model formats
            if isinstance(self.ml_model, dict):
                # Multiple part models
                models = self.ml_model
            elif hasattr(self.ml_model, 'parts_list'):
                # Single model with multiple parts
                models = {
                    part_code: self.ml_model
                    for part_code in (self.parts_list or [])
                }
            else:
                # Single part model
                if self.parts_list:
                    models = {part_code: self.ml_model for part_code in self.parts_list}
                else:
                    logger.warning("No parts list specified, cannot generate predictions")
                    return [], {}
            
            # Predict for each part
            for part_code, model in models.items():
                try:
                    if hasattr(model, 'predict_proba'):
                        prob = float(model.predict_proba(features)[0])
                    elif hasattr(model, 'predict'):
                        # Assume prediction returns probability
                        pred = model.predict(features)
                        if isinstance(pred, np.ndarray):
                            prob = float(pred[0])
                        else:
                            prob = float(pred)
                    else:
                        logger.warning(f"Model for {part_code} doesn't support prediction")
                        continue
                    
                    # Convert to 0-1 scale if needed (might be 0-100)
                    if prob > 1.0:
                        prob = prob / 100.0
                    
                    if prob >= self.threshold:
                        all_predictions[part_code] = prob
                        
                except Exception as e:
                    logger.warning(f"Prediction failed for {part_code}: {e}")
                    continue
            
            # Sort by confidence
            sorted_parts = sorted(
                all_predictions.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Take top 10
            top_10 = sorted_parts[:10]
            
            recommended_parts = [part for part, conf in top_10]
            confidence_dict = dict(top_10)
            
            return recommended_parts, confidence_dict
            
        except Exception as e:
            logger.error(f"Error generating ML predictions: {e}")
            return [], {}
    
    def _parse_parts_list(self, parts_data: Any) -> List[str]:
        """
        Parse parts list from various formats.
        
        Validates and normalizes part codes using utility functions.
        
        Args:
            parts_data: Parts data (list, string, JSON, etc.)
            
        Returns:
            List of validated and normalized part codes
        """
        if parts_data is None or (isinstance(parts_data, float) and pd.isna(parts_data)):
            return []
        
        parsed_parts = []
        
        if isinstance(parts_data, list):
            parsed_parts = [str(p) for p in parts_data if p]
        
        elif isinstance(parts_data, str):
            # Try JSON first
            try:
                parsed = json.loads(parts_data)
                if isinstance(parsed, list):
                    parsed_parts = [str(p) for p in parsed if p]
            except (json.JSONDecodeError, ValueError):
                pass
            
            # Try Python literal evaluation if JSON failed
            if not parsed_parts:
                try:
                    parsed = ast.literal_eval(parts_data)
                    if isinstance(parsed, list):
                        parsed_parts = [str(p) for p in parsed if p]
                except (ValueError, SyntaxError):
                    pass
            
            # Try comma-separated if still not parsed
            if not parsed_parts and ',' in parts_data:
                parsed_parts = [p.strip() for p in parts_data.split(',') if p.strip()]
            
            # Single value as fallback
            if not parsed_parts:
                parsed_parts = [parts_data.strip()] if parts_data.strip() else []
        
        # Validate and normalize all part codes
        validated_parts = []
        for part in parsed_parts:
            try:
                normalized = normalize_part_code(part)
                if is_valid_part_code(normalized):
                    validated_parts.append(normalized)
                else:
                    logger.warning(f"Invalid part code in backtest data: {part}")
            except Exception as e:
                logger.warning(f"Error normalizing part code {part}: {e}")
                continue
        
        return validated_parts
    
    def calculate_metrics(
        self,
        results: List[BacktestResult]
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive metrics from backtest results.
        
        Args:
            results: List of BacktestResult objects
            
        Returns:
            Dictionary containing all validation metrics
        """
        try:
            if not results:
                raise ValueError("No backtest results provided")
            
            # Extract predictions and actuals
            ml_predictions = [r.ml_recommendations for r in results]
            pm_predictions = [r.pm_recommendations for r in results]
            actuals = [r.actual_replaced for r in results]
            
            # Calculate ML metrics
            ml_metrics = AccuracyMetrics.calculate_multilabel_metrics(
                ml_predictions, actuals
            )
            
            # Calculate PM metrics
            pm_metrics = AccuracyMetrics.calculate_multilabel_metrics(
                pm_predictions, actuals
            )
            
            # Calculate advantage
            ml_advantage = {
                'precision_improvement': ml_metrics['precision'] - pm_metrics['precision'],
                'recall_improvement': ml_metrics['recall'] - pm_metrics['recall'],
                'f1_improvement': ml_metrics['f1_score'] - pm_metrics['f1_score']
            }
            
            # Additional analysis
            avg_confidences = []
            for r in results:
                if r.ml_confidences:
                    avg_confidences.append(np.mean(list(r.ml_confidences.values())))
            
            return {
                'ml_metrics': ml_metrics,
                'pm_metrics': pm_metrics,
                'ml_advantage': ml_advantage,
                'sample_size': len(results),
                'avg_confidence': float(np.mean(avg_confidences)) if avg_confidences else 0.0,
                'min_confidence': float(np.min(avg_confidences)) if avg_confidences else 0.0,
                'max_confidence': float(np.max(avg_confidences)) if avg_confidences else 0.0,
            }
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            raise ValidationError(f"Failed to calculate metrics: {e}") from e

    def sweep_thresholds(
        self,
        test_data: pd.DataFrame,
        feature_columns: List[str],
        thresholds: List[float]
    ) -> List[Dict[str, Any]]:
        """Evaluate metrics across multiple confidence thresholds.
        
        Returns list of dicts: {'threshold': t, **metrics}
        """
        results: List[Dict[str, Any]] = []
        original_threshold = self.threshold
        try:
            for t in thresholds:
                if not 0 < t < 1:
                    logger.warning(f"Skipping invalid threshold {t}")
                    continue
                self.threshold = t
                backtest_results = self.run_backtest(test_data, feature_columns)
                metrics = self.calculate_metrics(backtest_results)
                results.append({'threshold': t, **metrics})
            return results
        finally:
            self.threshold = original_threshold

