"""
Unit tests for model validation and accuracy testing.

Tests accuracy metrics calculation, data splitting, and validation frameworks.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List

from src.models.validation import (
    AccuracyMetrics,
    ValidationError,
    split_temporal_data,
    ModelAccuracyValidator
)
from src.testing.backtesting import BacktestingFramework, BacktestResult
from src.testing.pilot_analysis import (
    PilotDataCollector,
    PilotMetricsCalculator,
    RecommendationAction
)


class TestAccuracyMetrics:
    """Unit tests for AccuracyMetrics class."""
    
    def test_perfect_predictions(self):
        """Test with perfect predictions."""
        predictions = [['BP001', 'AF001'], ['EO001']]
        actuals = [['BP001', 'AF001'], ['EO001']]
        
        metrics = AccuracyMetrics.calculate_multilabel_metrics(
            predictions, actuals
        )
        
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1_score'] == 1.0
        assert metrics['true_positives'] == 3
        assert metrics['false_positives'] == 0
        assert metrics['false_negatives'] == 0
    
    def test_no_overlap(self):
        """Test with no overlap between predictions and actuals."""
        predictions = [['BP001', 'AF001']]
        actuals = [['EO001', 'SP001']]
        
        metrics = AccuracyMetrics.calculate_multilabel_metrics(
            predictions, actuals
        )
        
        assert metrics['precision'] == 0.0
        assert metrics['recall'] == 0.0
        assert metrics['f1_score'] == 0.0
        assert metrics['true_positives'] == 0
        assert metrics['false_positives'] == 2
        assert metrics['false_negatives'] == 2
    
    def test_partial_overlap(self):
        """Test with partial overlap."""
        predictions = [['BP001', 'AF001', 'BD001']]
        actuals = [['BP001', 'AF001', 'EO001']]
        
        metrics = AccuracyMetrics.calculate_multilabel_metrics(
            predictions, actuals
        )
        
        assert metrics['precision'] == pytest.approx(0.6667, rel=1e-3)
        assert metrics['recall'] == pytest.approx(0.6667, rel=1e-3)
        assert metrics['f1_score'] == pytest.approx(0.6667, rel=1e-3)
        assert metrics['true_positives'] == 2
        assert metrics['false_positives'] == 1
        assert metrics['false_negatives'] == 1
    
    def test_empty_predictions(self):
        """Test with empty predictions."""
        predictions = [[]]
        actuals = [['BP001', 'AF001']]
        
        metrics = AccuracyMetrics.calculate_multilabel_metrics(
            predictions, actuals
        )
        
        assert metrics['precision'] == 0.0
        assert metrics['recall'] == 0.0
        assert metrics['true_positives'] == 0
        assert metrics['false_negatives'] == 2
    
    def test_empty_actuals(self):
        """Test with empty actuals."""
        predictions = [['BP001', 'AF001']]
        actuals = [[]]
        
        metrics = AccuracyMetrics.calculate_multilabel_metrics(
            predictions, actuals
        )
        
        assert metrics['precision'] == 0.0
        assert metrics['recall'] == 0.0
        assert metrics['true_positives'] == 0
        assert metrics['false_positives'] == 2
    
    def test_length_mismatch(self):
        """Test with mismatched lengths."""
        predictions = [['BP001'], ['AF001']]
        actuals = [['BP001']]
        
        with pytest.raises(ValueError):
            AccuracyMetrics.calculate_multilabel_metrics(predictions, actuals)
    
    def test_binary_metrics(self):
        """Test binary metrics calculation."""
        y_true = np.array([1, 1, 0, 0, 1])
        y_pred = np.array([1, 0, 0, 1, 1])
        
        metrics = AccuracyMetrics.calculate_binary_metrics(y_true, y_pred)
        
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert metrics['support'] == 5
        assert metrics['true_positives'] == 2
        assert metrics['false_positives'] == 1
        assert metrics['false_negatives'] == 1
        assert metrics['true_negatives'] == 1


class TestTemporalDataSplit:
    """Tests for temporal data splitting."""
    
    def test_basic_split(self):
        """Test basic temporal split."""
        dates = pd.date_range('2023-01-01', '2024-12-31', freq='D')
        data = pd.DataFrame({
            'service_date': dates,
            'vehicle_id': [f'V{i}' for i in range(len(dates))],
            'value': range(len(dates))
        })
        
        train, test = split_temporal_data(
            data,
            train_end_date=datetime(2024, 6, 30),
            test_start_date=datetime(2024, 7, 1)
        )
        
        assert len(train) > 0
        assert len(test) > 0
        assert train['service_date'].max() <= datetime(2024, 6, 30)
        assert test['service_date'].min() >= datetime(2024, 7, 1)
    
    def test_missing_date_column(self):
        """Test error when date column is missing."""
        data = pd.DataFrame({
            'vehicle_id': ['V1', 'V2'],
            'value': [1, 2]
        })
        
        with pytest.raises(ValueError, match="service_date"):
            split_temporal_data(
                data,
                train_end_date=datetime(2024, 6, 30),
                test_start_date=datetime(2024, 7, 1)
            )
    
    def test_invalid_date_range(self):
        """Test error when date range is invalid."""
        data = pd.DataFrame({
            'service_date': pd.date_range('2023-01-01', '2024-12-31', freq='D')
        })
        
        with pytest.raises(ValueError):
            split_temporal_data(
                data,
                train_end_date=datetime(2024, 7, 1),
                test_start_date=datetime(2024, 6, 30)  # Invalid: before train_end
            )


class TestBacktestingFramework:
    """Tests for backtesting framework."""
    
    def test_backtest_result_creation(self):
        """Test creating backtest result."""
        result = BacktestResult(
            vehicle_id='V123',
            service_date=datetime.now(),
            ml_recommendations=['BP001', 'AF001'],
            pm_recommendations=['BP001'],
            actual_replaced=['BP001', 'EO001'],
            ml_confidences={'BP001': 0.85, 'AF001': 0.82}
        )
        
        assert result.vehicle_id == 'V123'
        assert len(result.ml_recommendations) == 2
        assert len(result.actual_replaced) == 2
    
    def test_backtesting_framework_init(self):
        """Test backtesting framework initialization."""
        framework = BacktestingFramework(
            ml_model=None,
            pm_schedule_rules={},
            confidence_threshold=0.80
        )
        
        assert framework.threshold == 0.80
        assert framework.pm_engine is not None
    
    def test_invalid_threshold(self):
        """Test error with invalid threshold."""
        with pytest.raises(ValueError):
            BacktestingFramework(
                ml_model=None,
                pm_schedule_rules={},
                confidence_threshold=1.5  # Invalid
            )


class TestValidatorGates:
    """Tests for ModelAccuracyValidator gates."""

    def test_validator_pass(self):
        metrics = {
            'ml_metrics': {'precision': 0.75, 'recall': 0.72, 'f1_score': 0.735},
            'pm_metrics': {'precision': 0.60, 'recall': 0.58, 'f1_score': 0.59},
            'ml_advantage': {'precision_improvement': 0.15, 'recall_improvement': 0.14, 'f1_improvement': 0.145},
            'sample_size': 100
        }
        v = ModelAccuracyValidator(min_precision=0.70, min_recall=0.70, min_f1=0.70, require_better_than_pm=True)
        summary = v.validate_backtest_metrics(metrics)
        assert summary['passed'] is True

    def test_validator_fail_thresholds(self):
        metrics = {
            'ml_metrics': {'precision': 0.68, 'recall': 0.72, 'f1_score': 0.70},
            'pm_metrics': {'precision': 0.60, 'recall': 0.58, 'f1_score': 0.59},
            'ml_advantage': {'precision_improvement': 0.08, 'recall_improvement': 0.14, 'f1_improvement': 0.11},
            'sample_size': 100
        }
        v = ModelAccuracyValidator(min_precision=0.70, min_recall=0.70, min_f1=0.70, require_better_than_pm=True)
        with pytest.raises(ValidationError):
            v.validate_backtest_metrics(metrics)

    def test_validator_fail_pm_advantage(self):
        metrics = {
            'ml_metrics': {'precision': 0.72, 'recall': 0.71, 'f1_score': 0.715},
            'pm_metrics': {'precision': 0.74, 'recall': 0.72, 'f1_score': 0.73},
            'ml_advantage': {'precision_improvement': -0.02, 'recall_improvement': -0.01, 'f1_improvement': -0.015},
            'sample_size': 100
        }
        v = ModelAccuracyValidator(min_precision=0.70, min_recall=0.70, min_f1=0.70, require_better_than_pm=True)
        with pytest.raises(ValidationError):
            v.validate_backtest_metrics(metrics)


class TestPilotDataCollector:
    """Tests for pilot data collector."""
    
    def test_record_recommendation(self):
        """Test recording recommendations."""
        collector = PilotDataCollector()
        
        recommendations = [
            {'part_code': 'BP001', 'confidence_score': 0.85, 'advisor_id': 'SA001'}
        ]
        
        collector.record_recommendation(
            job_card_number='JC001',
            vehicle_id='V123',
            recommendations=recommendations
        )
        
        assert len(collector.feedback_records) == 1
        assert collector.feedback_records[0].part_code == 'BP001'
        assert collector.feedback_records[0].action_taken == RecommendationAction.IGNORED
    
    def test_update_action(self):
        """Test updating action."""
        collector = PilotDataCollector()
        
        collector.record_recommendation(
            job_card_number='JC001',
            vehicle_id='V123',
            recommendations=[{'part_code': 'BP001', 'confidence_score': 0.85}]
        )
        
        collector.update_action(
            job_card_number='JC001',
            part_code='BP001',
            action=RecommendationAction.ACCEPTED
        )
        
        assert collector.feedback_records[0].action_taken == RecommendationAction.ACCEPTED


class TestPilotMetricsCalculator:
    """Tests for pilot metrics calculator."""
    
    def test_acceptance_rate(self):
        """Test calculating acceptance rate."""
        calculator = PilotMetricsCalculator()
        
        from src.testing.pilot_analysis import PilotFeedback
        
        feedback_records = [
            PilotFeedback(
                job_card_number='JC001',
                vehicle_id='V1',
                service_date=datetime.now(),
                part_code='BP001',
                confidence_score=0.85,
                action_taken=RecommendationAction.ACCEPTED,
                rejection_reason=None,
                service_advisor_id='SA001'
            ),
            PilotFeedback(
                job_card_number='JC002',
                vehicle_id='V2',
                service_date=datetime.now(),
                part_code='AF001',
                confidence_score=0.75,
                action_taken=RecommendationAction.REJECTED,
                rejection_reason='Not needed',
                service_advisor_id='SA001'
            ),
        ]
        
        metrics = calculator.calculate_acceptance_rate(feedback_records)
        
        assert metrics['total_recommendations'] == 2
        assert metrics['accepted'] == 1
        assert metrics['rejected'] == 1
        assert metrics['acceptance_rate'] == 0.5
    
    def test_empty_feedback(self):
        """Test with empty feedback."""
        calculator = PilotMetricsCalculator()
        
        metrics = calculator.calculate_acceptance_rate([])
        
        assert metrics['total_recommendations'] == 0
        assert metrics['acceptance_rate'] == 0.0

