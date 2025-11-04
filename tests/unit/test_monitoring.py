from datetime import datetime
import pytest

from src.testing.monitoring import ProductionMonitor, AlertService


class DummyAlertService(AlertService):
    def __init__(self):
        super().__init__({
            'email_alerts_enabled': False,
            'slack_alerts_enabled': False,
            'log_alerts': False,
        })
        self.sent = []

    def send_alert(self, level: str, message: str, metrics=None):
        self.sent.append({'level': level, 'message': message, 'metrics': metrics})


def test_detect_model_drift_no_baseline():
    monitor = ProductionMonitor(db_session=None, alert_service=None, baseline_metrics=None)
    assert monitor.detect_model_drift({'precision': 0.7, 'recall': 0.7, 'f1_score': 0.7}) is False


def test_detect_model_drift_warning_threshold():
    alerts = DummyAlertService()
    baseline = {'precision': 0.75, 'recall': 0.75, 'f1_score': 0.75}
    monitor = ProductionMonitor(db_session=None, alert_service=alerts, baseline_metrics=baseline)
    # Create a small drift above warning (3%) but below critical (5%)
    current = {'precision': 0.73, 'recall': 0.73, 'f1_score': 0.72}
    detected = monitor.detect_model_drift(current)
    assert detected is True
    assert any(a['level'] == 'WARNING' for a in alerts.sent)


def test_detect_model_drift_critical_threshold():
    alerts = DummyAlertService()
    baseline = {'precision': 0.80, 'recall': 0.80, 'f1_score': 0.80}
    monitor = ProductionMonitor(db_session=None, alert_service=alerts, baseline_metrics=baseline)
    # Create a drift above critical (5%)
    current = {'precision': 0.70, 'recall': 0.75, 'f1_score': 0.74}
    detected = monitor.detect_model_drift(current)
    assert detected is True
    assert any(a['level'] == 'CRITICAL' for a in alerts.sent)
