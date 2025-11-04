"""
Dependency injection for FastAPI endpoints.

This module provides dependency functions for database connections,
service instances, and other shared resources.
"""

from functools import lru_cache
from typing import Generator, Optional
from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
import redis
import logging

from src.config.settings import get_settings, Settings
from src.data.database import get_db_session
from src.data.repositories import VehicleRepository, ServiceHistoryRepository, RecommendationRepository
from src.models.model_factory import ModelFactory
from src.models.lightgbm_model import LightGBMPartModel
from src.services.recommendation_service import RecommendationService
from src.api.exceptions import ModelLoadError, DatabaseError, CacheError

logger = logging.getLogger(__name__)


@lru_cache()
def get_settings_cached() -> Settings:
    """Get cached settings instance."""
    return get_settings()


def get_database_session() -> Generator[Session, None, None]:
    """
    Dependency for database session.
    
    Yields:
        SQLAlchemy session instance
        
    Raises:
        HTTPException: If database connection fails
    """
    try:
        with get_db_session() as session:
            yield session
    except Exception as e:
        logger.error(f"Database session error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service unavailable"
        )


def get_redis_client() -> redis.Redis:
    """
    Dependency for Redis client.
    
    Returns:
        Redis client instance
        
    Raises:
        HTTPException: If Redis connection fails
    """
    try:
        settings = get_settings_cached()
        client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            password=settings.redis_password,
            db=settings.redis_db,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5
        )
        # Test connection
        client.ping()
        return client
    except Exception as e:
        logger.error(f"Redis connection error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cache service unavailable"
        )


def get_vehicle_repository(
    db: Session = Depends(get_database_session)
) -> VehicleRepository:
    """
    Dependency for vehicle repository.
    
    Args:
        db: Database session
        
    Returns:
        VehicleRepository instance
    """
    return VehicleRepository(db)


def get_service_history_repository(
    db: Session = Depends(get_database_session)
) -> ServiceHistoryRepository:
    """
    Dependency for service history repository.
    
    Args:
        db: Database session
        
    Returns:
        ServiceHistoryRepository instance
    """
    return ServiceHistoryRepository(db)


def get_part_recommendation_repository(
    db: Session = Depends(get_database_session)
) -> RecommendationRepository:
    """
    Dependency for part recommendation repository.
    
    Args:
        db: Database session
        
    Returns:
        RecommendationRepository instance
    """
    return RecommendationRepository(db)


@lru_cache()
def get_ml_model() -> LightGBMPartModel:
    """
    Dependency for ML model (cached).
    
    Returns:
        LightGBM model instance
        
    Raises:
        HTTPException: If model loading fails
    """
    try:
        settings = get_settings_cached()
        model = ModelFactory.create_model(
            model_type="lightgbm",
            model_path=settings.model_path
        )
        logger.info(f"ML model loaded successfully from {settings.model_path}")
        return model
    except Exception as e:
        logger.error(f"Model loading error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML model service unavailable"
        )


def get_recommendation_service(
    model: LightGBMPartModel = Depends(get_ml_model),
    vehicle_repo: VehicleRepository = Depends(get_vehicle_repository),
    service_repo: ServiceHistoryRepository = Depends(get_service_history_repository),
    recommendation_repo: RecommendationRepository = Depends(get_part_recommendation_repository),
    redis_client: redis.Redis = Depends(get_redis_client),
    settings: Settings = Depends(get_settings_cached)
) -> RecommendationService:
    """
    Dependency for recommendation service.
    
    Args:
        model: ML model instance
        vehicle_repo: Vehicle repository
        service_repo: Service history repository
        recommendation_repo: Part recommendation repository
        redis_client: Redis client
        settings: Application settings
        
    Returns:
        RecommendationService instance
    """
    return RecommendationService(
        model=model,
        vehicle_repository=vehicle_repo,
        service_repository=service_repo,
        recommendation_repository=recommendation_repo,
        redis_client=redis_client,
        config={
            "confidence_threshold": settings.confidence_threshold,
            "cache_ttl": settings.cache_ttl_seconds,
            "model_version": settings.model_version
        }
    )


def get_current_user_id() -> str:
    """
    Dependency for getting current user ID.
    
    In a real implementation, this would extract user ID from JWT token.
    For now, returns a default user ID.
    
    Returns:
        User ID string
    """
    # TODO: Implement proper JWT token extraction
    return "default_user"


def verify_api_key(api_key: Optional[str] = None) -> bool:
    """
    Dependency for API key verification.
    
    Args:
        api_key: API key from request header
        
    Returns:
        True if API key is valid
        
    Raises:
        HTTPException: If API key is invalid
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required"
        )
    
    # TODO: Implement proper API key validation
    # For now, accept any non-empty key
    if len(api_key) < 10:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    return True


def check_rate_limit(
    redis_client: redis.Redis = Depends(get_redis_client),
    user_id: str = Depends(get_current_user_id)
) -> bool:
    """
    Dependency for rate limiting.
    
    Args:
        redis_client: Redis client
        user_id: Current user ID
        
    Returns:
        True if request is allowed
        
    Raises:
        HTTPException: If rate limit exceeded
    """
    try:
        settings = get_settings_cached()
        key = f"rate_limit:{user_id}"
        current_count = redis_client.incr(key)
        
        if current_count == 1:
            redis_client.expire(key, 60)  # 1 minute window
        
        if current_count > 100:  # 100 requests per minute
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded"
            )
        
        return True
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"Rate limiting error: {e}")
        # Allow request if rate limiting fails
        return True


def get_request_id() -> str:
    """
    Dependency for generating request ID.
    
    Returns:
        Unique request ID
    """
    import uuid
    return str(uuid.uuid4())


def validate_vehicle_access(
    vehicle_id: str,
    user_id: str = Depends(get_current_user_id)
) -> str:
    """
    Dependency for validating vehicle access.
    
    Args:
        vehicle_id: Vehicle ID to check access for
        user_id: Current user ID
        
    Returns:
        Vehicle ID if access is allowed
        
    Raises:
        HTTPException: If access is denied
    """
    # TODO: Implement proper vehicle access control
    # For now, allow access to all vehicles
    return vehicle_id


def get_batch_size() -> int:
    """
    Dependency for batch processing size.
    
    Returns:
        Maximum batch size
    """
    settings = get_settings_cached()
    return settings.batch_size


def get_processing_timeout() -> int:
    """
    Dependency for processing timeout.
    
    Returns:
        Timeout in seconds
    """
    return 30  # 30 seconds timeout for processing
