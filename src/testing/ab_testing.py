"""
A/B testing framework for model retraining validation.

Tests new model against current production model before deployment.
"""

from typing import Dict, Optional, Any, List
from datetime import datetime
import logging
import pandas as pd

from ..models.validation import AccuracyMetrics, ValidationError
from ..utils import (
    get_ab_test_group,
    AB_TEST_SPLIT_RATIO_DEFAULT,
    MIN_PRECISION_THRESHOLD,
    MIN_RECALL_THRESHOLD
)

logger = logging.getLogger(__name__)


class ModelABTester:
    """
    A/B test new model against current production model.
    
    Before deploying new model, test on subset of traffic:
    - 20% traffic uses new model (treatment)
    - 80% traffic uses current model (control)
    - Compare performance for test period
    - Deploy if new model is better
    """
    
    def __init__(
        self,
        current_model: Any,
        new_model: Any,
        split_ratio: float = None
    ):
        """
        Initialize A/B tester.
        
        Args:
            current_model: Current production model
            new_model: New model to test
            split_ratio: Fraction of traffic for new model (default: uses AB_TEST_SPLIT_RATIO_DEFAULT constant)
        """
        if split_ratio is None:
            split_ratio = AB_TEST_SPLIT_RATIO_DEFAULT
        
        if not 0 < split_ratio < 1:
            raise ValueError(f"Split ratio must be between 0 and 1, got {split_ratio}")
        
        self.current_model = current_model
        self.new_model = new_model
        self.split_ratio = split_ratio
        self.test_results: Dict[str, List[Dict]] = {
            'control': [],
            'treatment': []
        }
        
        logger.info(f"A/B tester initialized with split ratio {split_ratio}")
    
    def should_use_new_model(self, vehicle_id: str) -> bool:
        """
        Determine if request should use new model.
        
        Uses consistent hashing to ensure same vehicle always gets
        same model during test period.
        
        Args:
            vehicle_id: Vehicle identifier
            
        Returns:
            True if should use new model, False otherwise
        """
        # Use utility function for consistent A/B testing
        group = get_ab_test_group(
            identifier=vehicle_id,
            split_ratio=self.split_ratio,
            group_names=('control', 'treatment')
        )
        return group == 'treatment'
    
    def generate_prediction(
        self,
        vehicle_id: str,
        features: pd.DataFrame,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Generate prediction using appropriate model.
        
        Args:
            vehicle_id: Vehicle identifier
            features: Feature DataFrame
            metadata: Optional metadata about the request
            
        Returns:
            Prediction dictionary with model version and predictions
        """
        use_new = self.should_use_new_model(vehicle_id)
        
        model = self.new_model if use_new else self.current_model
        model_version = 'new' if use_new else 'current'
        group = 'treatment' if use_new else 'control'
        
        try:
            # Generate prediction
            if hasattr(model, 'predict'):
                predictions = model.predict(features)
            elif callable(model):
                predictions = model(features)
            else:
                raise ValueError(f"Model does not support prediction: {type(model)}")
            
            # Store result for later analysis
            result = {
                'vehicle_id': vehicle_id,
                'model_version': model_version,
                'group': group,
                'predictions': predictions,
                'metadata': metadata or {},
                'timestamp': datetime.utcnow()
            }
            
            self.test_results[group].append(result)
            
            # Add metadata to predictions
            if isinstance(predictions, dict):
                predictions['model_version'] = model_version
                predictions['ab_test_group'] = group
            elif isinstance(predictions, list):
                # If predictions is a list, wrap it
                predictions = {
                    'recommendations': predictions,
                    'model_version': model_version,
                    'ab_test_group': group
                }
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error generating prediction in A/B test: {e}")
            # Fallback to current model on error
            logger.warning(f"Falling back to current model for {vehicle_id}")
            if hasattr(self.current_model, 'predict'):
                return self.current_model.predict(features)
            raise
    
    def compare_models(
        self,
        feedback_data: List[Dict],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Compare performance of both models during A/B test.
        
        Args:
            feedback_data: List of feedback records with vehicle_id, predictions, actuals
                          Format: [{'vehicle_id': '...', 'group': 'control'/'treatment',
                                   'predictions': [...], 'actuals': [...]}, ...]
            start_date: Start of test period (optional)
            end_date: End of test period (optional)
            
        Returns:
            Dictionary with comparison results
        """
        try:
            # Filter by date if provided
            filtered_data = feedback_data
            if start_date or end_date:
                filtered_data = [
                    f for f in feedback_data
                    if (start_date is None or f.get('timestamp', datetime.min) >= start_date) and
                       (end_date is None or f.get('timestamp', datetime.max) <= end_date)
                ]
            
            # Separate by group
            control_data = [f for f in filtered_data if f.get('group') == 'control']
            treatment_data = [f for f in filtered_data if f.get('group') == 'treatment']
            
            if not control_data or not treatment_data:
                raise ValueError("Insufficient data for comparison")
            
            # Extract predictions and actuals
            def extract_predictions_actuals(data):
                predictions = []
                actuals = []
                for record in data:
                    pred = record.get('predictions', [])
                    actual = record.get('actuals', [])
                    
                    # Normalize to lists
                    if isinstance(pred, dict) and 'recommendations' in pred:
                        pred = pred['recommendations']
                    if not isinstance(pred, list):
                        pred = [pred] if pred else []
                    if not isinstance(actual, list):
                        actual = [actual] if actual else []
                    
                    predictions.append(pred)
                    actuals.append(actual)
                
                return predictions, actuals
            
            control_pred, control_actual = extract_predictions_actuals(control_data)
            treatment_pred, treatment_actual = extract_predictions_actuals(treatment_data)
            
            # Calculate metrics for both
            current_metrics = AccuracyMetrics.calculate_multilabel_metrics(
                control_pred, control_actual
            )
            new_metrics = AccuracyMetrics.calculate_multilabel_metrics(
                treatment_pred, treatment_actual
            )
            
            # Calculate improvement
            improvement = {
                'precision': new_metrics['precision'] - current_metrics['precision'],
                'recall': new_metrics['recall'] - current_metrics['recall'],
                'f1_score': new_metrics['f1_score'] - current_metrics['f1_score']
            }
            
            # Decision criteria - use constants from utils
            MIN_IMPROVEMENT = 0.01  # 1%
            
            should_deploy = (
                improvement['f1_score'] > MIN_IMPROVEMENT and
                new_metrics['precision'] >= MIN_PRECISION_THRESHOLD and
                new_metrics['recall'] >= MIN_RECALL_THRESHOLD
            )
            
            return {
                'current_metrics': current_metrics,
                'new_metrics': new_metrics,
                'improvement': improvement,
                'should_deploy': should_deploy,
                'recommendation': (
                    'DEPLOY new model - shows improvement'
                    if should_deploy
                    else 'KEEP current model - insufficient improvement'
                ),
                'control_sample_size': len(control_data),
                'treatment_sample_size': len(treatment_data),
                'test_period': {
                    'start': start_date.isoformat() if start_date else None,
                    'end': end_date.isoformat() if end_date else None
                }
            }
            
        except Exception as e:
            logger.error(f"Error comparing models: {e}")
            raise ValidationError(f"Failed to compare models: {e}") from e
    
    def get_test_summary(self) -> Dict[str, Any]:
        """
        Get summary of A/B test results.
        
        Returns:
            Dictionary with test summary
        """
        return {
            'control_samples': len(self.test_results['control']),
            'treatment_samples': len(self.test_results['treatment']),
            'split_ratio': self.split_ratio,
            'test_start': (
                min(
                    min((r['timestamp'] for r in self.test_results['control']), default=datetime.max),
                    min((r['timestamp'] for r in self.test_results['treatment']), default=datetime.max)
                ).isoformat()
                if self.test_results['control'] or self.test_results['treatment']
                else None
            ),
            'test_end': datetime.utcnow().isoformat()
        }

