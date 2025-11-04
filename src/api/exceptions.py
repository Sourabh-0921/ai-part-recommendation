"""
Custom exceptions for the API layer.

This module defines all custom exceptions used in the API endpoints.
"""

from typing import Optional, Dict, Any


class APIException(Exception):
    """Base exception for API errors."""
    
    def __init__(
        self, 
        message: str, 
        error_code: str = "API_ERROR",
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class VehicleNotFoundError(APIException):
    """Raised when vehicle is not found."""
    
    def __init__(self, vehicle_id: str):
        super().__init__(
            message=f"Vehicle not found: {vehicle_id}",
            error_code="VEHICLE_NOT_FOUND",
            status_code=404,
            details={"vehicle_id": vehicle_id}
        )


class PredictionError(APIException):
    """Raised when model prediction fails."""
    
    def __init__(self, message: str = "Model prediction failed", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="PREDICTION_ERROR",
            status_code=500,
            details=details
        )


class DataQualityError(APIException):
    """Raised when data quality checks fail."""
    
    def __init__(self, message: str, field: Optional[str] = None):
        details = {"field": field} if field else {}
        super().__init__(
            message=message,
            error_code="DATA_QUALITY_ERROR",
            status_code=400,
            details=details
        )


class ModelLoadError(APIException):
    """Raised when model cannot be loaded."""
    
    def __init__(self, model_path: str, reason: str = "Unknown error"):
        super().__init__(
            message=f"Failed to load model from {model_path}: {reason}",
            error_code="MODEL_LOAD_ERROR",
            status_code=500,
            details={"model_path": model_path, "reason": reason}
        )


class ValidationError(APIException):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, field: str, value: Any):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            status_code=400,
            details={"field": field, "value": str(value)}
        )


class DatabaseError(APIException):
    """Raised when database operations fail."""
    
    def __init__(self, message: str, operation: str):
        super().__init__(
            message=message,
            error_code="DATABASE_ERROR",
            status_code=500,
            details={"operation": operation}
        )


class CacheError(APIException):
    """Raised when cache operations fail."""
    
    def __init__(self, message: str, operation: str):
        super().__init__(
            message=message,
            error_code="CACHE_ERROR",
            status_code=500,
            details={"operation": operation}
        )


class RateLimitError(APIException):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, limit: int, window: int):
        super().__init__(
            message=f"Rate limit exceeded: {limit} requests per {window} seconds",
            error_code="RATE_LIMIT_EXCEEDED",
            status_code=429,
            details={"limit": limit, "window": window}
        )


class ServiceUnavailableError(APIException):
    """Raised when a required service is unavailable."""
    
    def __init__(self, service: str, reason: str = "Service unavailable"):
        super().__init__(
            message=f"{service} service unavailable: {reason}",
            error_code="SERVICE_UNAVAILABLE",
            status_code=503,
            details={"service": service, "reason": reason}
        )


class AuthenticationError(APIException):
    """Raised when authentication fails."""
    
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_ERROR",
            status_code=401
        )


class AuthorizationError(APIException):
    """Raised when authorization fails."""
    
    def __init__(self, message: str = "Insufficient permissions"):
        super().__init__(
            message=message,
            error_code="AUTHORIZATION_ERROR",
            status_code=403
        )


class BatchProcessingError(APIException):
    """Raised when batch processing fails."""
    
    def __init__(self, message: str, failed_count: int, total_count: int):
        super().__init__(
            message=message,
            error_code="BATCH_PROCESSING_ERROR",
            status_code=500,
            details={"failed_count": failed_count, "total_count": total_count}
        )


class ConfigurationError(APIException):
    """Raised when configuration is invalid."""
    
    def __init__(self, message: str, config_key: str):
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            status_code=500,
            details={"config_key": config_key}
        )
