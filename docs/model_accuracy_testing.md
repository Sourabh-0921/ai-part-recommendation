# Model Accuracy Testing Strategy

This document describes the comprehensive model accuracy testing strategy implemented for the AI Parts Recommendation System.

## Overview

The model accuracy testing strategy ensures that ML models are validated at every stage before and after deployment. The strategy consists of three main stages:

1. **Historical Validation (Backtesting)** - Validate model on historical data
2. **Pilot Testing** - Test model with real users in controlled environment
3. **Production Monitoring** - Continuously monitor model performance

## Components

### 1. Core Validation Module (`src/models/validation.py`)

Provides core accuracy metrics calculation:

- `AccuracyMetrics.calculate_multilabel_metrics()` - Multi-label classification metrics
- `AccuracyMetrics.calculate_binary_metrics()` - Binary classification metrics
- `AccuracyMetrics.calculate_part_level_metrics()` - Per-part metrics
- `split_temporal_data()` - Time-based data splitting (CRITICAL: Never use random split)

### 2. Backtesting Framework (`src/testing/backtesting.py`)

Implements historical validation:

- `BacktestingFramework` - Main backtesting class
- `PMScheduleEngine` - PM schedule recommendation engine
- `BacktestResult` - Container for backtest results

### 3. Validation Report Generator (`src/testing/validation_report.py`)

Generates comprehensive Excel reports with:

- Executive summary
- Overall performance metrics
- Part-level analysis
- Vehicle model analysis
- Detailed results
- Cost impact analysis

### 4. Pilot Testing (`src/testing/pilot_analysis.py`)

Tracks and analyzes pilot testing:

- `PilotDataCollector` - Collects pilot feedback
- `PilotMetricsCalculator` - Calculates pilot metrics
- `RecommendationAction` - Enum for action types (ACCEPTED/REJECTED/IGNORED)

### 5. Production Monitoring (`src/testing/monitoring.py`)

Monitors model performance in production:

- `ProductionMonitor` - Tracks recommendations and feedback
- `AlertService` - Sends alerts for performance issues
- Model drift detection

### 6. A/B Testing (`src/testing/ab_testing.py`)

Tests new models before deployment:

- `ModelABTester` - A/B testing framework
- Consistent hashing for traffic splitting
- Model comparison and deployment recommendation

## Database Models

The following database models track validation data:

- `ModelPrediction` - Tracks all model predictions
- `PilotFeedback` - Tracks pilot testing feedback
- `ModelValidation` - Stores validation results
- `ModelAlert` - Tracks performance alerts

## Usage

### Running Historical Validation (Backtesting)

```bash
python scripts/run_backtest.py \
  --data path/to/service_history.csv \
  --train-end-date 2024-06-30 \
  --test-start-date 2024-07-01 \
  --confidence-threshold 0.80 \
  --output-dir reports \
  --parts-list BP001 AF001 EO001
```

### Monitoring Production Accuracy

```bash
python scripts/monitor_production_accuracy.py \
  --baseline-metrics baseline_metrics.json \
  --check-drift \
  --update-baseline
```

### Example: Using Validation in Code

```python
from src.models.validation import AccuracyMetrics, split_temporal_data
from src.testing.backtesting import BacktestingFramework
from src.testing.validation_report import ValidationReportGenerator

# Split data temporally
train_data, test_data = split_temporal_data(
    data,
    train_end_date=datetime(2024, 6, 30),
    test_start_date=datetime(2024, 7, 1)
)

# Run backtest
framework = BacktestingFramework(
    ml_model=trained_model,
    pm_schedule_rules=pm_rules,
    confidence_threshold=0.80
)

results = framework.run_backtest(test_data, feature_columns)
metrics = framework.calculate_metrics(results)

# Generate report
generator = ValidationReportGenerator('validation_report.xlsx')
generator.generate_report(results, metrics)
```

## Key Principles

1. **ALWAYS use temporal split** - Never use random split for time-series data
2. **NO data leakage** - Test data must not be seen during training
3. **Compare with baseline** - Always compare ML vs PM schedule
4. **Comprehensive metrics** - Calculate precision, recall, F1 score
5. **Error handling** - All functions have proper error handling
6. **Logging** - Extensive logging for audit trail
7. **Unit tests** - All accuracy functions are unit tested

## Validation Checklist

Before deploying a model, ensure:

- [ ] Historical validation completed (precision ≥ 70%, recall ≥ 70%)
- [ ] Model outperforms PM schedule
- [ ] Pilot testing completed (acceptance rate ≥ 55%)
- [ ] Real-world accuracy validated
- [ ] Production monitoring configured
- [ ] Alert system tested
- [ ] Baseline metrics established

## Testing Workflow

```
1. Historical Validation (Week 1-2)
   ├── Split data temporally
   ├── Train model on training set
   ├── Generate predictions on test set
   ├── Calculate metrics (precision, recall, F1)
   ├── Compare with PM schedule
   ├── Generate validation report
   └── Get stakeholder approval

2. Pilot Testing (Week 3-10)
   ├── Deploy to 3-5 dealers
   ├── Track recommendations and outcomes
   ├── Calculate acceptance rates
   ├── Measure real-world accuracy
   ├── Collect service advisor feedback
   ├── Generate pilot report
   └── Make Go/No-Go decision

3. Production Monitoring (Ongoing)
   ├── Track weekly metrics
   ├── Detect model drift
   ├── Send alerts for degradation
   ├── Analyze rejection reasons
   ├── Monthly model review
   └── Trigger retraining when needed

4. Model Retraining (Quarterly or as needed)
   ├── Collect feedback data
   ├── Retrain model
   ├── Validate on holdout set
   ├── A/B test vs current model
   ├── Deploy if better
   └── Update baseline metrics
```

## Metrics Thresholds

### Historical Validation
- **Minimum Precision**: 70%
- **Minimum Recall**: 70%
- **Minimum F1 Score**: 70%
- **ML Advantage**: At least 15% improvement over PM schedule

### Pilot Testing
- **Minimum Acceptance Rate**: 55%
- **Minimum Real-world Precision**: 65%
- **Minimum Real-world Recall**: 65%

### Production Monitoring
- **Drift Warning Threshold**: 3% drop in F1
- **Drift Critical Threshold**: 5% drop in F1
- **Weekly Review**: Required if metrics drop below baseline

## Support

For questions or issues, refer to:
- `tests/unit/test_validation.py` - Unit test examples
- `scripts/run_backtest.py` - Backtesting script example

