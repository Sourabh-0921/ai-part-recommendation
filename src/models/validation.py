"""
Model validation and accuracy testing module.

This module implements comprehensive model accuracy testing through:
1. Historical validation (backtesting)
2. Accuracy metrics calculation
3. Model validation frameworks

CRITICAL: All validation must be production-grade with proper error handling,
logging, and monitoring.
"""

from typing import List, Dict, Set, Tuple, Optional, Any
import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class AccuracyMetrics:
    """
    Calculate comprehensive accuracy metrics for model validation.
    
    Implements standard classification metrics:
    - Precision: Of recommended parts, % actually needed
    - Recall: Of needed parts, % we recommended
    - F1 Score: Harmonic mean of precision and recall
    """
    
    @staticmethod
    def calculate_binary_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate binary classification metrics.
        
        Args:
            y_true: True labels (0 or 1)
            y_pred: Predicted labels (0 or 1)
            
        Returns:
            Dictionary with precision, recall, f1_score, support
            
        Raises:
            ValueError: If inputs are invalid
        """
        try:
            if len(y_true) != len(y_pred):
                raise ValueError(
                    f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}"
                )
            
            if len(y_true) == 0:
                raise ValueError("Empty input arrays")
            
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # Calculate confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
            
            return {
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'true_positives': int(tp),
                'false_positives': int(fp),
                'true_negatives': int(tn),
                'false_negatives': int(fn),
                'support': int(len(y_true))
            }
        except Exception as e:
            logger.error(f"Error calculating binary metrics: {e}")
            raise ValidationError(f"Failed to calculate binary metrics: {e}") from e
    
    @staticmethod
    def calculate_multilabel_metrics(
        predictions: List[List[str]],
        actuals: List[List[str]]
    ) -> Dict[str, float]:
        """
        Calculate metrics for multi-label classification (multiple parts).
        
        This is the main metric for parts recommendation:
        - predictions: List of recommended part lists
        - actuals: List of actually replaced part lists
        
        Args:
            predictions: List of predicted part codes for each service
                        Example: [['BP001', 'AF001'], ['EO001'], ...]
            actuals: List of actual part codes for each service
                    Example: [['BP001', 'BD001'], ['EO001', 'AF001'], ...]
                    
        Returns:
            Dictionary with aggregated precision, recall, f1_score
            
        Raises:
            ValueError: If inputs are invalid
            
        Example:
            >>> predictions = [['BP001', 'AF001'], ['EO001']]
            >>> actuals = [['BP001', 'BD001'], ['EO001', 'AF001']]
            >>> metrics = AccuracyMetrics.calculate_multilabel_metrics(
            ...     predictions, actuals
            ... )
            >>> print(metrics)
            {
                'precision': 0.667,  # 2 correct out of 3 recommended
                'recall': 0.667,     # 2 caught out of 3 actual
                'f1_score': 0.667,
                'true_positives': 2,
                'false_positives': 1,
                'false_negatives': 1
            }
        """
        try:
            if len(predictions) != len(actuals):
                raise ValueError(
                    f"Length mismatch: predictions={len(predictions)}, "
                    f"actuals={len(actuals)}"
                )
            
            if len(predictions) == 0:
                raise ValueError("Empty input lists")
            
            # Normalize inputs (handle None, empty strings, etc.)
            predictions = [
                [p for p in (pred if isinstance(pred, list) else []) if p]
                for pred in predictions
            ]
            actuals = [
                [a for a in (actual if isinstance(actual, list) else []) if a]
                for actual in actuals
            ]
            
            true_positives = 0
            false_positives = 0
            false_negatives = 0
            
            for pred, actual in zip(predictions, actuals):
                pred_set = set(pred)
                actual_set = set(actual)
                
                # Parts we got right
                true_positives += len(pred_set & actual_set)
                
                # Parts we recommended but weren't needed
                false_positives += len(pred_set - actual_set)
                
                # Parts we missed
                false_negatives += len(actual_set - pred_set)
            
            # Calculate metrics
            precision = (
                true_positives / (true_positives + false_positives)
                if (true_positives + false_positives) > 0 else 0
            )
            
            recall = (
                true_positives / (true_positives + false_negatives)
                if (true_positives + false_negatives) > 0 else 0
            )
            
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0 else 0
            )
            
            return {
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1_score': round(f1, 4),
                'true_positives': true_positives,
                'false_positives': false_positives,
                'false_negatives': false_negatives,
                'support': len(predictions)
            }
        except Exception as e:
            logger.error(f"Error calculating multilabel metrics: {e}")
            raise ValidationError(f"Failed to calculate multilabel metrics: {e}") from e
    
    @staticmethod
    def calculate_part_level_metrics(
        predictions: List[List[str]],
        actuals: List[List[str]],
        part_codes: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate metrics for each individual part.
        
        Args:
            predictions: List of predicted part lists
            actuals: List of actual part lists
            part_codes: Optional list of part codes to analyze.
                       If None, analyzes all parts in predictions/actuals
                       
        Returns:
            Dictionary mapping part_code to its metrics
        """
        try:
            if len(predictions) != len(actuals):
                raise ValueError("Length mismatch between predictions and actuals")
            
            # Collect all part codes
            if part_codes is None:
                all_parts = set()
                for pred_list, actual_list in zip(predictions, actuals):
                    all_parts.update(pred_list if isinstance(pred_list, list) else [])
                    all_parts.update(actual_list if isinstance(actual_list, list) else [])
                part_codes = list(all_parts)
            
            part_metrics = {}
            
            for part_code in part_codes:
                # Convert to binary: did we recommend/need this part?
                y_pred = [
                    1 if part_code in (pred if isinstance(pred, list) else [])
                    else 0
                    for pred in predictions
                ]
                y_true = [
                    1 if part_code in (actual if isinstance(actual, list) else [])
                    else 0
                    for actual in actuals
                ]
                
                # Calculate binary metrics for this part
                metrics = AccuracyMetrics.calculate_binary_metrics(
                    np.array(y_true),
                    np.array(y_pred)
                )
                
                part_metrics[part_code] = metrics
            
            return part_metrics
            
        except Exception as e:
            logger.error(f"Error calculating part-level metrics: {e}")
            raise ValidationError(f"Failed to calculate part-level metrics: {e}") from e


def split_temporal_data(
    data: pd.DataFrame,
    train_end_date: datetime,
    test_start_date: datetime
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data temporally for proper backtesting.
    
    CRITICAL: Never use random split. Model must NEVER see future data
    during training. This simulates real-world deployment where we only
    know the past, not the future.
    
    Args:
        data: Full dataset with 'service_date' column
        train_end_date: Last date to include in training set
        test_start_date: First date to include in test set
        
    Returns:
        Tuple of (train_data, test_data)
        
    Raises:
        ValueError: If train_end_date >= test_start_date
        ValueError: If data is missing 'service_date' column
        ValueError: If split results in empty datasets
        
    Example:
        >>> data = load_service_history()  # 2 years: Jan 2023 - Dec 2024
        >>> train, test = split_temporal_data(
        ...     data,
        ...     train_end_date=datetime(2024, 6, 30),
        ...     test_start_date=datetime(2024, 7, 1)
        ... )
        >>> # Train: Jan 2023 - Jun 2024 (18 months)
        >>> # Test: Jul 2024 - Dec 2024 (6 months)
    """
    if 'service_date' not in data.columns:
        raise ValueError("Data must contain 'service_date' column")
    
    if train_end_date >= test_start_date:
        raise ValueError(
            f"train_end_date ({train_end_date}) must be before "
            f"test_start_date ({test_start_date})"
        )
    
    try:
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(data['service_date']):
            data['service_date'] = pd.to_datetime(data['service_date'])
        
        # Split
        train_data = data[data['service_date'] <= train_end_date].copy()
        test_data = data[data['service_date'] >= test_start_date].copy()
        
        logger.info(
            f"Data split: Train={len(train_data)} records "
            f"({train_data['service_date'].min()} to {train_data['service_date'].max()}), "
            f"Test={len(test_data)} records "
            f"({test_data['service_date'].min()} to {test_data['service_date'].max()})"
        )
        
        # Validation
        if len(train_data) == 0:
            raise ValueError("Training set is empty")
        if len(test_data) == 0:
            raise ValueError("Test set is empty")
        
        # Check for data leakage (no overlap)
        overlap = set(train_data['service_date'].dt.date) & set(
            test_data['service_date'].dt.date
        )
        if overlap:
            logger.warning(f"Date overlap detected: {len(overlap)} dates")
        
        return train_data, test_data
        
    except Exception as e:
        logger.error(f"Error splitting temporal data: {e}")
        raise ValidationError(f"Failed to split temporal data: {e}") from e


class ModelAccuracyValidator:
    """
    Validates model accuracy against baseline thresholds and PM schedule.
    
    Enforces:
    - Minimum precision/recall/F1 thresholds
    - ML must outperform PM schedule on F1 (and preferably on precision/recall)
    
    Raises ValidationError if gates are not met.
    """
    def __init__(
        self,
        min_precision: float,
        min_recall: float,
        min_f1: float,
        require_better_than_pm: bool = True
    ) -> None:
        """Initialize the accuracy validator.
        
        Args:
            min_precision: Minimum acceptable precision (0-1)
            min_recall: Minimum acceptable recall (0-1)
            min_f1: Minimum acceptable F1 score (0-1)
            require_better_than_pm: Whether ML must beat PM on F1
        """
        if not (0 <= min_precision <= 1 and 0 <= min_recall <= 1 and 0 <= min_f1 <= 1):
            raise ValueError("Thresholds must be between 0 and 1")
        self.min_precision = min_precision
        self.min_recall = min_recall
        self.min_f1 = min_f1
        self.require_better_than_pm = require_better_than_pm
        logger.info(
            "ModelAccuracyValidator initialized: min_precision=%.2f, min_recall=%.2f, min_f1=%.2f, require_better_than_pm=%s",
            min_precision, min_recall, min_f1, require_better_than_pm
        )

    def validate_backtest_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Validate backtest metrics and return a summary.
        
        Args:
            metrics: Dictionary with 'ml_metrics', 'pm_metrics', and 'ml_advantage'
        
        Returns:
            Validation summary including pass/fail flags and reasons
        
        Raises:
            ValidationError: If validation gates fail
        """
        try:
            ml = metrics.get('ml_metrics', {})
            pm = metrics.get('pm_metrics', {})
            advantage = metrics.get('ml_advantage', {})

            missing = [k for k in ['precision', 'recall', 'f1_score'] if k not in ml or k not in pm]
            if missing:
                raise ValidationError(f"Missing required metric fields: {missing}")

            failures = []

            if ml['precision'] < self.min_precision:
                failures.append(f"Precision {ml['precision']:.2%} < {self.min_precision:.2%}")
            if ml['recall'] < self.min_recall:
                failures.append(f"Recall {ml['recall']:.2%} < {self.min_recall:.2%}")
            if ml['f1_score'] < self.min_f1:
                failures.append(f"F1 {ml['f1_score']:.2%} < {self.min_f1:.2%}")

            if self.require_better_than_pm:
                if ml['f1_score'] <= pm['f1_score']:
                    failures.append(
                        f"ML F1 {ml['f1_score']:.2%} not greater than PM {pm['f1_score']:.2%}"
                    )

            passed = len(failures) == 0

            summary: Dict[str, Any] = {
                'passed': passed,
                'ml': ml,
                'pm': pm,
                'advantage': advantage,
                'failures': failures,
                'sample_size': metrics.get('sample_size', 0)
            }

            if not passed:
                logger.error("Model accuracy validation failed: %s", failures)
                raise ValidationError("; ".join(failures))

            logger.info("Model accuracy validation PASSED")
            return summary

        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Error validating backtest metrics: {e}")
            raise ValidationError(f"Validation failed: {e}") from e

