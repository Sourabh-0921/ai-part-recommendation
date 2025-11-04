#!/usr/bin/env python3
"""
Script to run historical validation (backtesting).

This script:
1. Loads historical service data
2. Splits data temporally
3. Runs backtest with ML model
4. Calculates metrics
5. Generates validation report
"""

import argparse
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd
from typing import Optional
from src.models.validation import split_temporal_data, AccuracyMetrics, ModelAccuracyValidator
from src.utils.constants import (
    MIN_PRECISION_THRESHOLD,
    MIN_RECALL_THRESHOLD,
    MIN_F1_THRESHOLD,
)
from src.testing.backtesting import BacktestingFramework
from src.testing.validation_report import ValidationReportGenerator
from src.models.model_loader import load_model as load_any_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_service_data(data_path: str) -> pd.DataFrame:
    """
    Load service history data.
    
    Args:
        data_path: Path to service data CSV/Parquet file
        
    Returns:
        DataFrame with service data
    """
    try:
        if data_path.endswith('.parquet'):
            df = pd.read_parquet(data_path)
        else:
            df = pd.read_csv(data_path)
        
        logger.info(f"Loaded {len(df)} service records from {data_path}")
        
        # Validate required columns
        required_cols = ['service_date', 'vehicle_id', 'parts_replaced']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading service data: {e}")
        raise


def load_pm_rules(pm_rules_path: Optional[str] = None) -> dict:
    """
    Load PM schedule rules.
    
    Args:
        pm_rules_path: Path to PM rules JSON file (optional)
        
    Returns:
        Dictionary of PM rules
    """
    if not pm_rules_path or not os.path.exists(pm_rules_path):
        logger.warning("No PM rules file provided, using empty rules")
        return {}
    
    try:
        import json
        with open(pm_rules_path, 'r') as f:
            rules = json.load(f)
        logger.info(f"Loaded PM rules for {len(rules)} vehicle models")
        return rules
    except Exception as e:
        logger.error(f"Error loading PM rules: {e}")
        return {}


def load_model(model_path: str):
    """Load model via model_loader and return (model, metadata)."""
    model, meta = load_any_model(model_path)
    logger.info(f"Model loaded: framework={meta.get('framework')}, uri={meta.get('model_uri')}\nversion={meta.get('model_version','')}")
    return model, meta


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Run historical validation (backtesting)'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to service history data file (CSV or Parquet)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        help='Path to trained ML model'
    )
    
    parser.add_argument(
        '--pm-rules',
        type=str,
        help='Path to PM schedule rules JSON file'
    )
    
    parser.add_argument(
        '--train-end-date',
        type=str,
        required=True,
        help='Last date for training data (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--test-start-date',
        type=str,
        required=True,
        help='First date for test data (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.80,
        help='Confidence threshold for recommendations (default: 0.80)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='reports',
        help='Output directory for reports (default: reports)'
    )
    parser.add_argument(
        '--threshold-sweep',
        type=float,
        nargs='+',
        help='List of thresholds to sweep (e.g., 0.6 0.7 0.8 0.9)'
    )
    parser.add_argument(
        '--target-precision',
        type=float,
        help='Target precision to pick operating threshold'
    )
    parser.add_argument(
        '--target-recall',
        type=float,
        help='Target recall to pick operating threshold'
    )
    
    parser.add_argument(
        '--parts-list',
        type=str,
        nargs='+',
        help='List of part codes to test'
    )
    
    args = parser.parse_args()
    
    try:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        logger.info("Loading service data...")
        data = load_service_data(args.data)
        
        # Parse dates
        train_end = datetime.strptime(args.train_end_date, '%Y-%m-%d')
        test_start = datetime.strptime(args.test_start_date, '%Y-%m-%d')
        
        # Split data
        logger.info("Splitting data temporally...")
        train_data, test_data = split_temporal_data(
            data,
            train_end_date=train_end,
            test_start_date=test_start
        )
        
        # Load PM rules
        pm_rules = load_pm_rules(args.pm_rules)
        
        # Load model (if provided)
        model = None
        model_meta = {}
        if args.model:
            model, model_meta = load_model(args.model)
        
        # Initialize backtesting framework
        logger.info("Initializing backtesting framework...")
        framework = BacktestingFramework(
            ml_model=model,
            pm_schedule_rules=pm_rules,
            confidence_threshold=args.confidence_threshold,
            parts_list=args.parts_list
        )
        
        # Identify feature columns
        exclude_cols = [
            'service_date', 'vehicle_id', 'parts_replaced',
            'job_card_number', 'vehicle_model', 'odometer_reading',
            'service_type', 'invoice_date'
        ]
        feature_columns = [
            col for col in test_data.columns
            if col not in exclude_cols
        ]
        
        if not feature_columns:
            logger.warning("No feature columns found, using all numeric columns")
            feature_columns = test_data.select_dtypes(include=['number']).columns.tolist()
        
        logger.info(f"Using {len(feature_columns)} feature columns")
        
        # Run backtest (or sweep)
        logger.info(f"Running backtest on {len(test_data)} test records...")
        if args.threshold_sweep:
            sweep = framework.sweep_thresholds(test_data, feature_columns, args.threshold_sweep)
            # Pick best threshold meeting targets if provided
            chosen_threshold = None
            if args.target_precision or args.target_recall:
                for row in sorted(sweep, key=lambda r: r['ml_metrics']['f1_score'], reverse=True):
                    meets_p = args.target_precision is None or row['ml_metrics']['precision'] >= args.target_precision
                    meets_r = args.target_recall is None or row['ml_metrics']['recall'] >= args.target_recall
                    if meets_p and meets_r:
                        chosen_threshold = row['threshold']
                        metrics = {k: v for k, v in row.items() if k != 'threshold'}
                        break
            # If no chosen threshold, fall back to default threshold
            if chosen_threshold is not None:
                logger.info(f"Selected operating threshold: {chosen_threshold:.2f}")
                framework.threshold = chosen_threshold
            # Run backtest with current (possibly updated) threshold
            results = framework.run_backtest(test_data=test_data, feature_columns=feature_columns)
        else:
            results = framework.run_backtest(
                test_data=test_data,
                feature_columns=feature_columns
            )
        
        if not results:
            logger.error("No backtest results generated")
            return 1
        
        # Calculate metrics
        logger.info("Calculating metrics...")
        metrics = framework.calculate_metrics(results)
        
        logger.info("Backtest Results:")
        logger.info(f"  ML Precision: {metrics['ml_metrics']['precision']:.2%}")
        logger.info(f"  ML Recall: {metrics['ml_metrics']['recall']:.2%}")
        logger.info(f"  ML F1: {metrics['ml_metrics']['f1_score']:.2%}")
        logger.info(f"  PM Precision: {metrics['pm_metrics']['precision']:.2%}")
        logger.info(f"  PM Recall: {metrics['pm_metrics']['recall']:.2%}")
        logger.info(f"  PM F1: {metrics['pm_metrics']['f1_score']:.2%}")
        logger.info(f"  F1 Improvement: {metrics['ml_advantage']['f1_improvement']:.2%}")
        
        # Enforce accuracy gates
        logger.info("Validating metrics against thresholds and PM baseline...")
        validator = ModelAccuracyValidator(
            min_precision=MIN_PRECISION_THRESHOLD,
            min_recall=MIN_RECALL_THRESHOLD,
            min_f1=MIN_F1_THRESHOLD,
            require_better_than_pm=True,
        )
        try:
            validator.validate_backtest_metrics(metrics)
        except Exception as e:
            logger.error(f"Validation gates failed: {e}")
            # Still generate a report for analysis before exiting
            # Generate report
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_path = output_dir / f'validation_report_{timestamp}.xlsx'
            logger.info(f"Generating validation report (despite failure): {report_path}")
            generator = ValidationReportGenerator(str(report_path))
            generator.generate_report(
                backtest_results=results,
                metrics=metrics
            )
            return 1

        # Generate report (on pass)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = output_dir / f'validation_report_{timestamp}.xlsx'
        
        logger.info(f"Generating validation report: {report_path}")
        generator = ValidationReportGenerator(str(report_path))
        generator.generate_report(
            backtest_results=results,
            metrics={**metrics, 'model_info': model_meta}
        )
        
        logger.info("Backtesting complete!")
        logger.info(f"Report saved to: {report_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during backtesting: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())

