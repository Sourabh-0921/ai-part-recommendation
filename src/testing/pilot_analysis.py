"""
Pilot testing implementation for model validation.

This module handles:
- Pilot data collection
- Pilot metrics calculation
- Real-world accuracy measurement
"""

from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import logging
from collections import defaultdict

from ..models.validation import AccuracyMetrics, ValidationError

logger = logging.getLogger(__name__)


class RecommendationAction(Enum):
    """Action taken on recommendation."""
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    IGNORED = "IGNORED"


@dataclass
class PilotFeedback:
    """Container for pilot testing feedback."""
    job_card_number: str
    vehicle_id: str
    service_date: datetime
    part_code: str
    confidence_score: float
    action_taken: RecommendationAction
    rejection_reason: Optional[str]
    service_advisor_id: str
    actual_replaced: Optional[bool] = None


class PilotDataCollector:
    """
    Collect and store pilot testing data.
    
    During pilot, we track:
    - What was recommended
    - What was accepted/rejected
    - What was actually replaced
    - Service advisor feedback
    """
    
    def __init__(self, db_session=None):
        """
        Initialize pilot data collector.
        
        Args:
            db_session: Optional database session for persistence
        """
        self.db = db_session
        self.feedback_records: List[PilotFeedback] = []
        logger.info("Pilot data collector initialized")
    
    def record_recommendation(
        self,
        job_card_number: str,
        vehicle_id: str,
        recommendations: List[Dict],
        service_date: Optional[datetime] = None
    ):
        """
        Record recommendations shown during pilot.
        
        Args:
            job_card_number: Job card ID
            vehicle_id: Vehicle identifier
            recommendations: List of recommendations with confidence scores
                           Format: [{'part_code': 'BP001', 'confidence_score': 0.85, ...}, ...]
            service_date: Service date (defaults to now)
        """
        try:
            if service_date is None:
                service_date = datetime.utcnow()
            
            for rec in recommendations:
                feedback = PilotFeedback(
                    job_card_number=job_card_number,
                    vehicle_id=vehicle_id,
                    service_date=service_date,
                    part_code=rec.get('part_code', ''),
                    confidence_score=rec.get('confidence_score', 0.0),
                    action_taken=RecommendationAction.IGNORED,  # Default
                    rejection_reason=None,
                    service_advisor_id=rec.get('advisor_id', ''),
                    actual_replaced=None  # Will be updated later
                )
                
                # Store in memory
                self.feedback_records.append(feedback)
                
                # Store in database if available
                if self.db:
                    # Note: This would require a PilotFeedback database model
                    # For now, we just log
                    logger.debug(f"Would store feedback in DB: {job_card_number}, {rec.get('part_code')}")
            
            logger.info(
                f"Recorded {len(recommendations)} recommendations "
                f"for job card {job_card_number}"
            )
            
        except Exception as e:
            logger.error(f"Error recording recommendations: {e}")
            raise
    
    def update_action(
        self,
        job_card_number: str,
        part_code: str,
        action: RecommendationAction,
        rejection_reason: Optional[str] = None
    ):
        """
        Update action taken on recommendation.
        
        Args:
            job_card_number: Job card ID
            part_code: Part code
            action: Action taken (ACCEPTED/REJECTED/IGNORED)
            rejection_reason: Reason if rejected
        """
        try:
            # Find recommendation
            feedback = None
            for f in self.feedback_records:
                if f.job_card_number == job_card_number and f.part_code == part_code:
                    feedback = f
                    break
            
            if feedback:
                feedback.action_taken = action
                feedback.rejection_reason = rejection_reason
                
                logger.info(
                    f"Updated action for {part_code} in {job_card_number}: {action.value}"
                )
            else:
                logger.warning(
                    f"Recommendation not found: {job_card_number}, {part_code}"
                )
                
        except Exception as e:
            logger.error(f"Error updating action: {e}")
            raise
    
    def update_actual_replacement(
        self,
        job_card_number: str,
        part_code: str,
        actual_replaced: bool
    ):
        """
        Update whether part was actually replaced.
        
        Args:
            job_card_number: Job card ID
            part_code: Part code
            actual_replaced: Whether part was actually replaced
        """
        try:
            # Find recommendation
            for feedback in self.feedback_records:
                if feedback.job_card_number == job_card_number and feedback.part_code == part_code:
                    feedback.actual_replaced = actual_replaced
                    logger.info(
                        f"Updated actual replacement for {part_code} in {job_card_number}: {actual_replaced}"
                    )
                    return
            
            logger.warning(
                f"Recommendation not found for actual replacement: {job_card_number}, {part_code}"
            )
                
        except Exception as e:
            logger.error(f"Error updating actual replacement: {e}")
            raise


class PilotMetricsCalculator:
    """
    Calculate metrics from pilot testing data.
    
    Tracks:
    - Acceptance rates
    - Real-world accuracy
    - Comparison with PM schedule
    - Service advisor satisfaction
    """
    
    def __init__(self, db_session=None):
        """
        Initialize pilot metrics calculator.
        
        Args:
            db_session: Optional database session
        """
        self.db = db_session
    
    def calculate_acceptance_rate(
        self,
        feedback_records: List[PilotFeedback],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, float]:
        """
        Calculate acceptance rate for pilot period.
        
        Args:
            feedback_records: List of pilot feedback records
            start_date: Start of period (optional)
            end_date: End of period (optional)
            
        Returns:
            Dictionary with acceptance metrics
        """
        try:
            # Filter by date if provided
            filtered_feedback = feedback_records
            if start_date or end_date:
                filtered_feedback = [
                    f for f in feedback_records
                    if (start_date is None or f.service_date >= start_date) and
                       (end_date is None or f.service_date <= end_date)
                ]
            
            total = len(filtered_feedback)
            accepted = sum(
                1 for f in filtered_feedback 
                if f.action_taken == RecommendationAction.ACCEPTED
            )
            rejected = sum(
                1 for f in filtered_feedback 
                if f.action_taken == RecommendationAction.REJECTED
            )
            ignored = sum(
                1 for f in filtered_feedback 
                if f.action_taken == RecommendationAction.IGNORED
            )
            
            return {
                'total_recommendations': total,
                'accepted': accepted,
                'rejected': rejected,
                'ignored': ignored,
                'acceptance_rate': accepted / total if total > 0 else 0,
                'rejection_rate': rejected / total if total > 0 else 0,
                'ignore_rate': ignored / total if total > 0 else 0,
            }
        except Exception as e:
            logger.error(f"Error calculating acceptance rate: {e}")
            raise
    
    def calculate_real_world_accuracy(
        self,
        feedback_records: List[PilotFeedback],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, float]:
        """
        Calculate real-world accuracy from pilot.
        
        This requires follow-up verification:
        - Were accepted recommendations actually correct?
        - What parts were missed?
        
        Args:
            feedback_records: List of pilot feedback records
            start_date: Start of period (optional)
            end_date: End of period (optional)
            
        Returns:
            Dictionary with accuracy metrics
        """
        try:
            # Filter by date if provided
            filtered_feedback = feedback_records
            if start_date or end_date:
                filtered_feedback = [
                    f for f in feedback_records
                    if (start_date is None or f.service_date >= start_date) and
                       (end_date is None or f.service_date <= end_date)
                ]
            
            # Filter to only completed services with actual replacement data
            completed_services = [
                f for f in filtered_feedback
                if f.actual_replaced is not None
            ]
            
            # Group by job card
            services = defaultdict(list)
            for feedback in completed_services:
                services[feedback.job_card_number].append(feedback)
            
            # Calculate metrics
            ml_predictions = []
            actuals = []
            
            for job_card, feedbacks in services.items():
                # What we recommended and was accepted
                ml_pred = [
                    f.part_code for f in feedbacks
                    if f.action_taken == RecommendationAction.ACCEPTED
                ]
                
                # What was actually replaced
                actual = [
                    f.part_code for f in feedbacks
                    if f.actual_replaced
                ]
                
                ml_predictions.append(ml_pred)
                actuals.append(actual)
            
            if not ml_predictions:
                logger.warning("No completed services with replacement data")
                return {
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'support': 0
                }
            
            # Calculate accuracy
            metrics = AccuracyMetrics.calculate_multilabel_metrics(
                ml_predictions, actuals
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating real-world accuracy: {e}")
            raise ValidationError(f"Failed to calculate real-world accuracy: {e}") from e
    
    def calculate_rejection_reasons(
        self,
        feedback_records: List[PilotFeedback]
    ) -> Dict[str, int]:
        """
        Analyze rejection reasons.
        
        Args:
            feedback_records: List of pilot feedback records
            
        Returns:
            Dictionary mapping rejection reasons to counts
        """
        rejection_reasons = defaultdict(int)
        
        for feedback in feedback_records:
            if feedback.action_taken == RecommendationAction.REJECTED:
                reason = feedback.rejection_reason or 'No reason provided'
                rejection_reasons[reason] += 1
        
        return dict(rejection_reasons)
    
    def calculate_confidence_distribution(
        self,
        feedback_records: List[PilotFeedback]
    ) -> Dict[str, Any]:
        """
        Analyze confidence score distribution by action.
        
        Args:
            feedback_records: List of pilot feedback records
            
        Returns:
            Dictionary with confidence statistics
        """
        accepted_confidences = [
            f.confidence_score for f in feedback_records
            if f.action_taken == RecommendationAction.ACCEPTED
        ]
        rejected_confidences = [
            f.confidence_score for f in feedback_records
            if f.action_taken == RecommendationAction.REJECTED
        ]
        ignored_confidences = [
            f.confidence_score for f in feedback_records
            if f.action_taken == RecommendationAction.IGNORED
        ]
        
        def calculate_stats(values):
            if not values:
                return {}
            import numpy as np
            return {
                'mean': float(np.mean(values)),
                'median': float(np.median(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'count': len(values)
            }
        
        return {
            'accepted': calculate_stats(accepted_confidences),
            'rejected': calculate_stats(rejected_confidences),
            'ignored': calculate_stats(ignored_confidences),
        }

