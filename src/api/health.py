"""
Health check and status API endpoints.

This module contains endpoints for health checks, model status,
and system monitoring.
"""

import time
import logging
import psutil
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status

from .models import (
    HealthResponse,
    ModelStatusResponse,
    ErrorResponse
)
from src.api.dependencies import (
    get_ml_model,
    get_redis_client,
    get_database_session,
    get_settings_cached
)
from src.config.settings import Settings
from src.models.lightgbm_model import LightGBMPartModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/health", tags=["health"])

# Global variables for uptime tracking
_start_time = time.time()


@router.get(
    "/",
    response_model=HealthResponse,
    summary="Health check",
    description="Check if the service is healthy and running"
)
async def health_check(
    settings: Settings = Depends(get_settings_cached)
) -> HealthResponse:
    """
    Perform a basic health check.
    
    Returns:
        HealthResponse with service status and basic info
    """
    try:
        uptime = time.time() - _start_time
        
        return HealthResponse(
            status="healthy",
            timestamp=time.time(),
            version=settings.api_version,
            environment=settings.environment,
            uptime_seconds=uptime
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unhealthy"
        )


@router.get(
    "/detailed",
    response_model=Dict[str, Any],
    summary="Detailed health check",
    description="Perform a comprehensive health check of all services"
)
async def detailed_health_check(
    db_session = Depends(get_database_session),
    redis_client = Depends(get_redis_client),
    model: LightGBMPartModel = Depends(get_ml_model),
    settings: Settings = Depends(get_settings_cached)
) -> Dict[str, Any]:
    """
    Perform a detailed health check of all services.
    
    Returns:
        Dict with detailed health information for all components
    """
    try:
        health_status = {
            "overall_status": "healthy",
            "timestamp": time.time(),
            "version": settings.api_version,
            "environment": settings.environment,
            "uptime_seconds": time.time() - _start_time,
            "components": {}
        }
        
        # Check database
        try:
            db_session.execute("SELECT 1")
            health_status["components"]["database"] = {
                "status": "healthy",
                "response_time_ms": 0  # TODO: Measure actual response time
            }
        except Exception as e:
            health_status["components"]["database"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["overall_status"] = "degraded"
        
        # Check Redis
        try:
            redis_client.ping()
            health_status["components"]["redis"] = {
                "status": "healthy",
                "response_time_ms": 0  # TODO: Measure actual response time
            }
        except Exception as e:
            health_status["components"]["redis"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["overall_status"] = "degraded"
        
        # Check ML model
        try:
            # Test model prediction with dummy data
            import pandas as pd
            dummy_data = pd.DataFrame({
                'vehicle_model': ['Test'],
                'current_odometer': [10000],
                'ema_value': [500],
                'terrain_type': ['Urban'],
                'season_code': ['Summer']
            })
            model.predict(dummy_data)
            health_status["components"]["ml_model"] = {
                "status": "healthy",
                "model_version": settings.model_version,
                "model_path": settings.model_path
            }
        except Exception as e:
            health_status["components"]["ml_model"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["overall_status"] = "degraded"
        
        # System metrics
        health_status["system"] = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
        }
        
        return health_status
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Detailed health check failed"
        )


@router.get(
    "/model",
    response_model=ModelStatusResponse,
    summary="Model status",
    description="Get detailed status of the ML model"
)
async def model_status(
    model: LightGBMPartModel = Depends(get_ml_model),
    settings: Settings = Depends(get_settings_cached)
) -> ModelStatusResponse:
    """
    Get detailed status of the ML model.
    
    Returns:
        ModelStatusResponse with model information
    """
    try:
        # Get model information
        model_info = await model.get_model_info()
        
        return ModelStatusResponse(
            model_loaded=True,
            model_version=settings.model_version,
            model_path=settings.model_path,
            last_updated=model_info.get("last_updated"),
            confidence_threshold=settings.confidence_threshold,
            total_predictions=model_info.get("total_predictions"),
            average_confidence=model_info.get("average_confidence")
        )
        
    except Exception as e:
        logger.error(f"Model status check failed: {e}")
        return ModelStatusResponse(
            model_loaded=False,
            model_version="unknown",
            model_path=settings.model_path,
            last_updated=None,
            confidence_threshold=settings.confidence_threshold,
            total_predictions=None,
            average_confidence=None
        )


@router.get(
    "/metrics",
    response_model=Dict[str, Any],
    summary="System metrics",
    description="Get system performance metrics"
)
async def get_metrics(
    settings: Settings = Depends(get_settings_cached)
) -> Dict[str, Any]:
    """
    Get system performance metrics.
    
    Returns:
        Dict with system metrics
    """
    try:
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Process metrics
        process = psutil.Process()
        process_memory = process.memory_info()
        
        metrics = {
            "timestamp": time.time(),
            "system": {
                "cpu_percent": cpu_percent,
                "memory_total_gb": memory.total / (1024**3),
                "memory_available_gb": memory.available / (1024**3),
                "memory_percent": memory.percent,
                "disk_total_gb": disk.total / (1024**3),
                "disk_free_gb": disk.free / (1024**3),
                "disk_percent": (disk.used / disk.total) * 100
            },
            "process": {
                "pid": process.pid,
                "memory_rss_mb": process_memory.rss / (1024**2),
                "memory_vms_mb": process_memory.vms / (1024**2),
                "cpu_percent": process.cpu_percent(),
                "num_threads": process.num_threads(),
                "create_time": process.create_time()
            },
            "application": {
                "uptime_seconds": time.time() - _start_time,
                "version": settings.api_version,
                "environment": settings.environment,
                "max_concurrent_requests": settings.max_concurrent_requests,
                "cache_ttl_seconds": settings.cache_ttl_seconds
            }
        }
        
        # Add load average if available
        try:
            metrics["system"]["load_average"] = psutil.getloadavg()
        except AttributeError:
            pass  # Not available on Windows
        
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve metrics"
        )


@router.get(
    "/readiness",
    response_model=Dict[str, str],
    summary="Readiness check",
    description="Check if the service is ready to accept requests"
)
async def readiness_check(
    db_session = Depends(get_database_session),
    redis_client = Depends(get_redis_client),
    model: LightGBMPartModel = Depends(get_ml_model)
) -> Dict[str, str]:
    """
    Check if the service is ready to accept requests.
    
    This is used by Kubernetes readiness probes.
    
    Returns:
        Dict with readiness status
    """
    try:
        # Check database connectivity
        db_session.execute("SELECT 1")
        
        # Check Redis connectivity
        redis_client.ping()
        
        # Check model availability
        if not model.is_loaded():
            raise Exception("Model not loaded")
        
        return {
            "status": "ready",
            "message": "Service is ready to accept requests"
        }
        
    except Exception as e:
        logger.warning(f"Readiness check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service not ready: {e}"
        )


@router.get(
    "/liveness",
    response_model=Dict[str, str],
    summary="Liveness check",
    description="Check if the service is alive"
)
async def liveness_check() -> Dict[str, str]:
    """
    Check if the service is alive.
    
    This is used by Kubernetes liveness probes.
    
    Returns:
        Dict with liveness status
    """
    return {
        "status": "alive",
        "message": "Service is alive",
        "timestamp": time.time()
    }
