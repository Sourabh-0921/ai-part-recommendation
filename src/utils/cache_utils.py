"""
Cache key generation and management utilities.

This module provides functions for generating consistent cache keys
for Redis and other caching systems.
"""

from typing import Optional, Dict, Any, Union
import logging

from .hashing_utils import generate_cache_key, hash_dict
from .json_utils import safe_dumps

logger = logging.getLogger(__name__)


# Cache key prefixes
CACHE_PREFIX_RECOMMENDATION = "rec"
CACHE_PREFIX_VEHICLE = "vehicle"
CACHE_PREFIX_SERVICE_HISTORY = "service"
CACHE_PREFIX_MODEL = "model"
CACHE_PREFIX_EMA = "ema"
CACHE_PREFIX_PREDICTION = "pred"


def get_recommendation_cache_key(
    vehicle_id: str,
    odometer: Optional[float] = None,
    complaints: Optional[str] = None,
    dealer_code: Optional[str] = None
) -> str:
    """
    Generate cache key for vehicle recommendations.
    
    Args:
        vehicle_id: Vehicle identifier
        odometer: Optional odometer reading
        complaints: Optional customer complaints
        dealer_code: Optional dealer code
        
    Returns:
        Cache key string
        
    Example:
        >>> get_recommendation_cache_key("VH123", odometer=50000, dealer_code="DL001")
        'rec:VH123:dealer_code=DL001:odometer=50000'
    """
    kwargs = {}
    
    if odometer is not None:
        kwargs['odometer'] = round(odometer, 1)  # Round for cache consistency
    
    if dealer_code:
        kwargs['dealer_code'] = dealer_code
    
    if complaints:
        # Hash complaints for consistent key (complaints can be long/variable)
        from .hashing_utils import hash_string
        kwargs['complaints_hash'] = hash_string(complaints, algorithm='md5')[:8]
    
    return generate_cache_key(CACHE_PREFIX_RECOMMENDATION, vehicle_id, **kwargs)


def get_vehicle_cache_key(vehicle_id: str, data_type: str = "full") -> str:
    """
    Generate cache key for vehicle data.
    
    Args:
        vehicle_id: Vehicle identifier
        data_type: Type of data ('full', 'features', 'history')
        
    Returns:
        Cache key string
        
    Example:
        >>> get_vehicle_cache_key("VH123", "features")
        'vehicle:VH123:features'
    """
    return generate_cache_key(CACHE_PREFIX_VEHICLE, vehicle_id, data_type)


def get_service_history_cache_key(
    vehicle_id: str,
    limit: Optional[int] = None,
    days: Optional[int] = None
) -> str:
    """
    Generate cache key for service history.
    
    Args:
        vehicle_id: Vehicle identifier
        limit: Optional limit on number of records
        days: Optional number of days to look back
        
    Returns:
        Cache key string
        
    Example:
        >>> get_service_history_cache_key("VH123", limit=10, days=365)
        'service:VH123:days=365:limit=10'
    """
    kwargs = {}
    
    if limit is not None:
        kwargs['limit'] = limit
    
    if days is not None:
        kwargs['days'] = days
    
    return generate_cache_key(CACHE_PREFIX_SERVICE_HISTORY, vehicle_id, **kwargs)


def get_model_cache_key(model_version: str, model_type: str = "lightgbm") -> str:
    """
    Generate cache key for model metadata.
    
    Args:
        model_version: Model version string
        model_type: Model type ('lightgbm', 'ensemble', etc.)
        
    Returns:
        Cache key string
        
    Example:
        >>> get_model_cache_key("1.0.0", "lightgbm")
        'model:lightgbm:1.0.0'
    """
    return generate_cache_key(CACHE_PREFIX_MODEL, model_type, model_version)


def get_ema_cache_key(
    vehicle_id: str,
    part_code: Optional[str] = None
) -> str:
    """
    Generate cache key for EMA calculations.
    
    Args:
        vehicle_id: Vehicle identifier
        part_code: Optional part code (for part-specific EMA)
        
    Returns:
        Cache key string
        
    Example:
        >>> get_ema_cache_key("VH123", "BP001")
        'ema:VH123:BP001'
        >>> get_ema_cache_key("VH123")
        'ema:VH123'
    """
    if part_code:
        return generate_cache_key(CACHE_PREFIX_EMA, vehicle_id, part_code)
    else:
        return generate_cache_key(CACHE_PREFIX_EMA, vehicle_id)


def get_prediction_cache_key(
    vehicle_id: str,
    features_hash: Optional[str] = None
) -> str:
    """
    Generate cache key for model predictions.
    
    Args:
        vehicle_id: Vehicle identifier
        features_hash: Optional hash of features (for cache consistency)
        
    Returns:
        Cache key string
        
    Example:
        >>> get_prediction_cache_key("VH123", "abc123")
        'pred:VH123:abc123'
    """
    if features_hash:
        return generate_cache_key(CACHE_PREFIX_PREDICTION, vehicle_id, features_hash)
    else:
        return generate_cache_key(CACHE_PREFIX_PREDICTION, vehicle_id)


def generate_features_hash(features: Dict[str, Any]) -> str:
    """
    Generate hash of features dictionary for cache key.
    
    Args:
        features: Features dictionary
        
    Returns:
        Hash string (first 16 characters)
        
    Example:
        >>> features = {'odometer': 50000, 'age_days': 365}
        >>> generate_features_hash(features)
        'a1b2c3d4e5f6g7h8'
    """
    # Serialize to JSON string and hash
    json_str = safe_dumps(features, sort_keys=True)
    return hash_dict({"features": json_str}, algorithm='md5')[:16]


def get_cache_key_with_ttl(
    base_key: str,
    ttl_seconds: Optional[int] = None
) -> Dict[str, Any]:
    """
    Generate cache key with TTL information.
    
    Args:
        base_key: Base cache key
        ttl_seconds: Optional TTL in seconds
        
    Returns:
        Dictionary with 'key' and 'ttl' fields
        
    Example:
        >>> get_cache_key_with_ttl("rec:VH123", ttl_seconds=3600)
        {'key': 'rec:VH123', 'ttl': 3600}
    """
    result = {'key': base_key}
    
    if ttl_seconds is not None:
        result['ttl'] = ttl_seconds
    
    return result


def invalidate_vehicle_caches(vehicle_id: str) -> list:
    """
    Generate list of all cache keys to invalidate for a vehicle.
    
    Args:
        vehicle_id: Vehicle identifier
        
    Returns:
        List of cache keys to invalidate
        
    Example:
        >>> invalidate_vehicle_caches("VH123")
        ['rec:VH123', 'vehicle:VH123:full', 'vehicle:VH123:features', ...]
    """
    keys = [
        get_recommendation_cache_key(vehicle_id),
        get_vehicle_cache_key(vehicle_id, "full"),
        get_vehicle_cache_key(vehicle_id, "features"),
        get_service_history_cache_key(vehicle_id),
        get_ema_cache_key(vehicle_id),
    ]
    
    # Add pattern-based keys (vehicle:*:*)
    keys.append(f"{CACHE_PREFIX_VEHICLE}:{vehicle_id}:*")
    keys.append(f"{CACHE_PREFIX_RECOMMENDATION}:{vehicle_id}:*")
    keys.append(f"{CACHE_PREFIX_SERVICE_HISTORY}:{vehicle_id}:*")
    keys.append(f"{CACHE_PREFIX_EMA}:{vehicle_id}:*")
    keys.append(f"{CACHE_PREFIX_PREDICTION}:{vehicle_id}:*")
    
    return keys

