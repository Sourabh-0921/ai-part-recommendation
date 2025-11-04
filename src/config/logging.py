"""
Logging configuration for AI Parts Recommendation System.

This module sets up structured logging with JSON format for production
and human-readable format for development.
"""

import logging
import sys
import json
from logging.handlers import RotatingFileHandler
from typing import Dict, Any
from datetime import datetime

from .settings import get_settings


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage(),
            'module': record.module,
            'process': record.process,
            'thread': record.thread,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in log_entry and not key.startswith('_'):
                log_entry[key] = value
        
        return json.dumps(log_entry, default=str)


class ContextFilter(logging.Filter):
    """Filter to add context information to log records."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add context information to log record."""
        # Add request ID if available
        if hasattr(record, 'request_id'):
            record.request_id = getattr(record, 'request_id', 'N/A')
        
        # Add user ID if available
        if hasattr(record, 'user_id'):
            record.user_id = getattr(record, 'user_id', 'N/A')
        
        # Add vehicle ID if available
        if hasattr(record, 'vehicle_id'):
            record.vehicle_id = getattr(record, 'vehicle_id', 'N/A')
        
        return True


def setup_logging() -> None:
    """Configure application logging."""
    settings = get_settings()
    
    # Create formatter based on environment
    if settings.log_format == 'json':
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(ContextFilter())
    
    # File handler (rotating)
    file_handler = RotatingFileHandler(
        settings.log_file,
        maxBytes=settings.log_max_bytes,
        backupCount=settings.log_backup_count
    )
    file_handler.setFormatter(formatter)
    file_handler.addFilter(ContextFilter())
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level))
    root_logger.handlers.clear()  # Clear existing handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Set specific logger levels
    logging.getLogger('uvicorn').setLevel(logging.INFO)
    logging.getLogger('uvicorn.access').setLevel(logging.WARNING)
    logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
    logging.getLogger('lightgbm').setLevel(logging.WARNING)
    
    # Log configuration
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured - Level: {settings.log_level}, Format: {settings.log_format}")


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name."""
    return logging.getLogger(name)


def log_function_call(func_name: str, **kwargs: Any) -> None:
    """Log function call with parameters."""
    logger = get_logger('function_calls')
    logger.info(f"Function call: {func_name}", extra={'function': func_name, **kwargs})


def log_performance(operation: str, duration: float, **kwargs: Any) -> None:
    """Log performance metrics."""
    logger = get_logger('performance')
    logger.info(f"Performance: {operation}", extra={
        'operation': operation,
        'duration_seconds': duration,
        **kwargs
    })


def log_prediction(vehicle_id: str, part_code: str, confidence: float, **kwargs: Any) -> None:
    """Log prediction results."""
    logger = get_logger('predictions')
    logger.info(f"Prediction: {part_code} for {vehicle_id}", extra={
        'vehicle_id': vehicle_id,
        'part_code': part_code,
        'confidence_score': confidence,
        **kwargs
    })


def log_error(error: Exception, context: Dict[str, Any] = None) -> None:
    """Log error with context."""
    logger = get_logger('errors')
    logger.error(f"Error occurred: {str(error)}", extra=context or {}, exc_info=True)
