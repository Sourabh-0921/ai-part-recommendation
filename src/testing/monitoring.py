"""
Production monitoring for model accuracy.

This module handles:
- Real-time metrics tracking
- Model drift detection
- Automated alert system
"""

from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
import logging
import smtplib
from email.mime.text import MIMEText
import json

try:
    from prometheus_client import Counter, Histogram, Gauge
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not available, metrics will not be exported")

from ..models.validation import AccuracyMetrics, ValidationError
from ..data.repositories import ModelPredictionRepository
from ..data.database import get_db_session

logger = logging.getLogger(__name__)


# Prometheus metrics (if available)
if PROMETHEUS_AVAILABLE:
    recommendation_counter = Counter(
        'recommendations_generated_total',
        'Total recommendations generated',
        ['dealer_code', 'accepted']
    )
    
    recommendation_accuracy = Gauge(
        'recommendation_accuracy',
        'Current recommendation accuracy',
        ['metric_type']
    )
    
    recommendation_latency = Histogram(
        'recommendation_latency_seconds',
        'Recommendation generation latency',
        ['dealer_code']
    )
else:
    # Dummy metrics if prometheus not available
    recommendation_counter = None
    recommendation_accuracy = None
    recommendation_latency = None


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AlertService:
    """
    Send alerts for model performance issues.
    
    Supports:
    - Email alerts
    - Slack notifications (placeholder)
    - Database logging
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize alert service.
        
        Args:
            config: Configuration dictionary with alert settings
        """
        self.config = config
        self.email_enabled = config.get('email_alerts_enabled', False)
        self.slack_enabled = config.get('slack_alerts_enabled', False)
        self.log_alerts = config.get('log_alerts', True)
        logger.info(f"Alert service initialized: email={self.email_enabled}, slack={self.slack_enabled}")
    
    def send_alert(
        self,
        level: str,
        message: str,
        metrics: Optional[Dict] = None
    ):
        """
        Send alert via configured channels.
        
        Args:
            level: Alert level (INFO/WARNING/CRITICAL)
            message: Alert message
            metrics: Optional metrics dictionary
        """
        log_level_map = {
            'CRITICAL': logging.CRITICAL,
            'WARNING': logging.WARNING,
            'INFO': logging.INFO
        }
        
        log_level = log_level_map.get(level.upper(), logging.WARNING)
        
        if self.log_alerts:
            logger.log(
                log_level,
                f"ALERT [{level}]: {message}"
            )
        
        # Store in database (would need database model)
        self._log_alert_to_db(level, message, metrics)
        
        # Send email if enabled
        if self.email_enabled and level in ['WARNING', 'CRITICAL']:
            try:
                self._send_email_alert(level, message, metrics)
            except Exception as e:
                logger.error(f"Failed to send email alert: {e}")
        
        # Send Slack if enabled
        if self.slack_enabled and level in ['WARNING', 'CRITICAL']:
            try:
                self._send_slack_alert(level, message, metrics)
            except Exception as e:
                logger.error(f"Failed to send Slack alert: {e}")
    
    def _log_alert_to_db(self, level: str, message: str, metrics: Optional[Dict]):
        """Log alert to database (placeholder)."""
        # Would implement with database model
        logger.debug(f"Would log alert to DB: {level}, {message[:50]}")
    
    def _send_email_alert(
        self,
        level: str,
        message: str,
        metrics: Optional[Dict]
    ):
        """Send email alert."""
        try:
            recipients = self.config.get('alert_email_recipients', [])
            if not recipients:
                logger.warning("No email recipients configured")
                return
            
            smtp_host = self.config.get('smtp_host', 'localhost')
            smtp_port = self.config.get('smtp_port', 587)
            smtp_from = self.config.get('smtp_from', 'alerts@parts-recommendation.com')
            
            # Compose email
            subject = f"[{level}] Model Performance Alert - Parts Recommendation System"
            body = f"""
            Alert Level: {level}
            Message: {message}
            Timestamp: {datetime.utcnow().isoformat()}
            
            Metrics:
            {self._format_metrics(metrics) if metrics else 'N/A'}
            
            Please investigate and take appropriate action.
            
            ---
            This is an automated alert from the AI Parts Recommendation System.
            """
            
            msg = MIMEText(body)
            msg['Subject'] = subject
            msg['From'] = smtp_from
            msg['To'] = ', '.join(recipients)
            
            # Send email
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                if self.config.get('smtp_use_tls', True):
                    server.starttls()
                if self.config.get('smtp_user'):
                    server.login(
                        self.config['smtp_user'],
                        self.config.get('smtp_password', '')
                    )
                server.send_message(msg)
            
            logger.info(f"Email alert sent to {len(recipients)} recipients")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            raise
    
    def _send_slack_alert(
        self,
        level: str,
        message: str,
        metrics: Optional[Dict]
    ):
        """Send Slack alert (placeholder - would need slack_sdk)."""
        # Placeholder implementation
        logger.debug(f"Would send Slack alert: {level}, {message[:50]}")
    
    @staticmethod
    def _format_metrics(metrics: Dict) -> str:
        """Format metrics for display."""
        if not metrics:
            return 'N/A'
        
        lines = []
        for key, value in metrics.items():
            if isinstance(value, dict):
                lines.append(f"  {key}:")
                for sub_key, sub_value in value.items():
                    lines.append(f"    {sub_key}: {sub_value}")
            else:
                lines.append(f"  {key}: {value}")
        
        return '\n'.join(lines)


class ProductionMonitor:
    """
    Monitor model accuracy in production.
    
    Tracks:
    - Real-time accuracy metrics
    - Model drift detection
    - Alerts for degradation
    """
    
    def __init__(
        self,
        db_session=None,
        alert_service: Optional[AlertService] = None,
        baseline_metrics: Optional[Dict[str, float]] = None
    ):
        """
        Initialize production monitor.
        
        Args:
            db_session: Optional database session
            alert_service: Alert service instance
            baseline_metrics: Baseline metrics for drift detection
        """
        self.db = db_session
        self.alert_service = alert_service
        self.baseline_metrics = baseline_metrics or {}
        self.predictions_cache: List[Dict] = []
        logger.info("Production monitor initialized")
    
    def track_recommendation(
        self,
        vehicle_id: str,
        recommendations: List[Dict],
        dealer_code: str = 'unknown'
    ):
        """
        Track recommendation in production.
        
        Args:
            vehicle_id: Vehicle identifier
            recommendations: List of recommendations made
                           Format: [{'part_code': 'BP001', 'confidence_score': 0.85, ...}, ...]
            dealer_code: Dealer code
        """
        try:
            # Store in memory cache
            prediction_record = {
                'vehicle_id': vehicle_id,
                'dealer_code': dealer_code,
                'recommendations': recommendations,
                'timestamp': datetime.utcnow(),
                'feedback_received': False
            }
            self.predictions_cache.append(prediction_record)
            
            # Store in database if available
            if self.db:
                try:
                    repo = ModelPredictionRepository(self.db)
                    records = []
                    for rec in recommendations:
                        records.append({
                            'vehicle_id': vehicle_id,
                            'job_card_number': rec.get('job_card_number'),
                            'part_code': rec.get('part_code'),
                            'confidence_score': float(rec.get('confidence_score', 0.0)),
                            'model_version': rec.get('model_version', ''),
                            'dealer_code': dealer_code,
                            'ab_test_group': rec.get('ab_test_group')
                        })
                    if records:
                        repo.create_batch(records)
                except Exception as e:
                    logger.error(f"DB store failed for predictions: {e}")
            
            # Update Prometheus metrics
            if recommendation_counter:
                for rec in recommendations:
                    recommendation_counter.labels(
                        dealer_code=dealer_code,
                        accepted='pending'
                    ).inc()
            
            logger.debug(f"Tracked recommendation for {vehicle_id}")
            
        except Exception as e:
            logger.error(f"Error tracking recommendation: {e}")
    
    def track_feedback(
        self,
        vehicle_id: str,
        part_code: str,
        accepted: bool,
        actual_replaced: Optional[bool] = None
    ):
        """
        Track feedback on recommendation.
        
        Args:
            vehicle_id: Vehicle identifier
            part_code: Part that was recommended
            accepted: Was recommendation accepted?
            actual_replaced: Was part actually replaced?
        """
        try:
            # Update cache
            for pred in self.predictions_cache:
                if pred['vehicle_id'] == vehicle_id:
                    for rec in pred['recommendations']:
                        if rec.get('part_code') == part_code:
                            rec['accepted'] = accepted
                            rec['actual_replaced'] = actual_replaced
                            pred['feedback_received'] = True
                            break
            
            # Update database if available
            if self.db:
                try:
                    repo = ModelPredictionRepository(self.db)
                    repo.update_feedback(vehicle_id, part_code, accepted, actual_replaced if actual_replaced is not None else False)
                except Exception as e:
                    logger.error(f"DB feedback update failed: {e}")
            
            # Update Prometheus metrics
            if recommendation_counter:
                recommendation_counter.labels(
                    dealer_code='all',
                    accepted=str(accepted).lower()
                ).inc()
            
            logger.debug(f"Tracked feedback for {vehicle_id}, {part_code}")
            
        except Exception as e:
            logger.error(f"Error tracking feedback: {e}")
    
    def calculate_weekly_metrics(self) -> Dict[str, float]:
        """
        Calculate metrics for past week.
        
        Returns:
            Dictionary with weekly accuracy metrics
        """
        try:
            one_week_ago = datetime.utcnow() - timedelta(days=7)
            
            # Get predictions from last week with feedback
            recent_predictions = [
                p for p in self.predictions_cache
                if p['timestamp'] >= one_week_ago and p['feedback_received']
            ]
            
            if not recent_predictions:
                logger.warning("No recent predictions with feedback")
                return {
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'support': 0
                }
            
            # Calculate metrics
            ml_predictions = []
            actuals = []
            
            for pred in recent_predictions:
                pred_parts = [
                    rec['part_code'] for rec in pred['recommendations']
                    if rec.get('accepted', False)
                ]
                actual_parts = [
                    rec['part_code'] for rec in pred['recommendations']
                    if rec.get('actual_replaced', False)
                ]
                
                ml_predictions.append(pred_parts)
                actuals.append(actual_parts)
            
            metrics = AccuracyMetrics.calculate_multilabel_metrics(
                ml_predictions, actuals
            )
            
            # Update Prometheus gauges
            if recommendation_accuracy:
                recommendation_accuracy.labels(metric_type='precision').set(
                    metrics['precision']
                )
                recommendation_accuracy.labels(metric_type='recall').set(
                    metrics['recall']
                )
                recommendation_accuracy.labels(metric_type='f1_score').set(
                    metrics['f1_score']
                )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating weekly metrics: {e}")
            raise ValidationError(f"Failed to calculate weekly metrics: {e}") from e
    
    def detect_model_drift(
        self,
        current_metrics: Optional[Dict[str, float]] = None
    ) -> bool:
        """
        Detect if model performance has drifted.
        
        Args:
            current_metrics: Current performance metrics (if None, calculates from cache)
            
        Returns:
            True if significant drift detected, False otherwise
        """
        try:
            if not self.baseline_metrics:
                logger.warning("No baseline metrics available for drift detection")
                return False
            
            if current_metrics is None:
                current_metrics = self.calculate_weekly_metrics()
            
            # Calculate drift
            precision_drift = abs(
                current_metrics['precision'] - self.baseline_metrics.get('precision', 0)
            )
            recall_drift = abs(
                current_metrics['recall'] - self.baseline_metrics.get('recall', 0)
            )
            f1_drift = abs(
                current_metrics['f1_score'] - self.baseline_metrics.get('f1_score', 0)
            )
            
            # Alert thresholds
            WARNING_THRESHOLD = 0.03  # 3%
            CRITICAL_THRESHOLD = 0.05  # 5%
            
            if f1_drift > CRITICAL_THRESHOLD and self.alert_service:
                self.alert_service.send_alert(
                    level='CRITICAL',
                    message=f"Model drift detected: F1 dropped by {f1_drift:.2%}",
                    metrics={
                        'baseline': self.baseline_metrics,
                        'current': current_metrics,
                        'drift': {
                            'precision': precision_drift,
                            'recall': recall_drift,
                            'f1_score': f1_drift
                        }
                    }
                )
                return True
            
            elif f1_drift > WARNING_THRESHOLD and self.alert_service:
                self.alert_service.send_alert(
                    level='WARNING',
                    message=f"Model drift warning: F1 dropped by {f1_drift:.2%}",
                    metrics={
                        'baseline': self.baseline_metrics,
                        'current': current_metrics,
                        'drift': {
                            'precision': precision_drift,
                            'recall': recall_drift,
                            'f1_score': f1_drift
                        }
                    }
                )
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error detecting model drift: {e}")
            return False
    
    def update_baseline_metrics(self, metrics: Dict[str, float]):
        """
        Update baseline metrics for drift detection.
        
        Args:
            metrics: Baseline metrics dictionary
        """
        self.baseline_metrics = metrics
        logger.info(f"Baseline metrics updated: {metrics}")

