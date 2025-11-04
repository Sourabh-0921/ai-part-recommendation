#!/usr/bin/env python3
"""
Script to monitor production model accuracy.

This script:
1. Calculates weekly accuracy metrics
2. Detects model drift
3. Sends alerts if needed
4. Updates baseline metrics
"""

import argparse
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.testing.monitoring import ProductionMonitor, AlertService
from src.config.settings import get_settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_alert_config() -> dict:
    """
    Get alert service configuration.
    
    Returns:
        Alert configuration dictionary
    """
    settings = get_settings()
    
    return {
        'email_alerts_enabled': os.getenv('EMAIL_ALERTS_ENABLED', 'false').lower() == 'true',
        'slack_alerts_enabled': os.getenv('SLACK_ALERTS_ENABLED', 'false').lower() == 'true',
        'log_alerts': True,
        'alert_email_recipients': os.getenv('ALERT_EMAIL_RECIPIENTS', '').split(','),
        'smtp_host': os.getenv('SMTP_HOST', 'localhost'),
        'smtp_port': int(os.getenv('SMTP_PORT', '587')),
        'smtp_from': os.getenv('SMTP_FROM', 'alerts@parts-recommendation.com'),
        'smtp_user': os.getenv('SMTP_USER', ''),
        'smtp_password': os.getenv('SMTP_PASSWORD', ''),
        'smtp_use_tls': os.getenv('SMTP_USE_TLS', 'true').lower() == 'true'
    }


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Monitor production model accuracy'
    )
    
    parser.add_argument(
        '--baseline-metrics',
        type=str,
        help='Path to baseline metrics JSON file'
    )
    
    parser.add_argument(
        '--check-drift',
        action='store_true',
        help='Check for model drift and send alerts'
    )
    
    parser.add_argument(
        '--update-baseline',
        action='store_true',
        help='Update baseline metrics from current weekly metrics'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize services
        alert_config = get_alert_config()
        alert_service = AlertService(alert_config)
        
        monitor = ProductionMonitor(
            db_session=None,  # Would connect to DB in production
            alert_service=alert_service if args.check_drift else None
        )
        
        # Load baseline metrics if provided
        if args.baseline_metrics and os.path.exists(args.baseline_metrics):
            import json
            with open(args.baseline_metrics, 'r') as f:
                baseline = json.load(f)
            monitor.update_baseline_metrics(baseline)
            logger.info(f"Loaded baseline metrics from {args.baseline_metrics}")
        
        # Calculate weekly metrics
        logger.info("Calculating weekly accuracy metrics...")
        weekly_metrics = monitor.calculate_weekly_metrics()
        
        logger.info("Weekly Metrics:")
        logger.info(f"  Precision: {weekly_metrics['precision']:.2%}")
        logger.info(f"  Recall: {weekly_metrics['recall']:.2%}")
        logger.info(f"  F1 Score: {weekly_metrics['f1_score']:.2%}")
        logger.info(f"  Sample Size: {weekly_metrics['support']}")
        
        # Check for drift
        if args.check_drift:
            logger.info("Checking for model drift...")
            drift_detected = monitor.detect_model_drift(weekly_metrics)
            
            if drift_detected:
                logger.warning("Model drift detected - alerts sent")
            else:
                logger.info("No significant drift detected")
        
        # Update baseline if requested
        if args.update_baseline:
            baseline_path = args.baseline_metrics or 'baseline_metrics.json'
            import json
            with open(baseline_path, 'w') as f:
                json.dump(weekly_metrics, f, indent=2)
            logger.info(f"Baseline metrics updated: {baseline_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error monitoring production accuracy: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())

