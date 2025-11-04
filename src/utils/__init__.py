"""
Utilities package for AI Parts Recommendation System.

This package contains utility functions, helpers, and common functionality used
throughout the system.

Modules:
    - datetime_utils: Date and time operations
    - hashing_utils: Consistent hashing functions
    - json_utils: JSON serialization/deserialization
    - validation_utils: Data validation functions
    - cache_utils: Cache key generation
    - constants: Application-wide constants
"""

# DateTime utilities
from .datetime_utils import (
    parse_date,
    to_utc,
    format_iso,
    days_between,
    months_between,
    is_date_in_range,
    get_current_utc,
    add_days,
    add_months,
    get_age_in_days,
    get_season,
    is_weekend,
)

# Hashing utilities
from .hashing_utils import (
    hash_string,
    hash_to_int,
    get_ab_test_group,
    generate_cache_key,
    hash_dict,
)

# JSON utilities
from .json_utils import (
    DateTimeEncoder,
    safe_dumps,
    safe_loads,
    safe_load_file,
    safe_dump_file,
    is_valid_json,
    parse_datetime_from_json,
)

# Validation utilities
from .validation_utils import (
    is_valid_vehicle_id,
    is_valid_part_code,
    is_valid_odometer,
    is_valid_confidence_score,
    is_valid_email,
    is_valid_phone,
    validate_not_none,
    validate_not_empty,
    validate_in_range,
    validate_one_of,
    validate_type,
    validate_list_of_type,
    sanitize_string,
    normalize_part_code,
)

# Cache utilities
from .cache_utils import (
    CACHE_PREFIX_RECOMMENDATION,
    CACHE_PREFIX_VEHICLE,
    CACHE_PREFIX_SERVICE_HISTORY,
    CACHE_PREFIX_MODEL,
    CACHE_PREFIX_EMA,
    CACHE_PREFIX_PREDICTION,
    get_recommendation_cache_key,
    get_vehicle_cache_key,
    get_service_history_cache_key,
    get_model_cache_key,
    get_ema_cache_key,
    get_prediction_cache_key,
    generate_features_hash,
    get_cache_key_with_ttl,
    invalidate_vehicle_caches,
)

# Constants
from .constants import (
    # Part Categories
    PART_CATEGORY_BRAKE,
    PART_CATEGORY_ENGINE,
    PART_CATEGORY_FILTER,
    PART_CATEGORY_FLUID,
    PART_CATEGORY_ELECTRICAL,
    PART_CATEGORY_SUSPENSION,
    PART_CATEGORY_TRANSMISSION,
    PART_CATEGORY_COOLING,
    PART_CATEGORY_EXHAUST,
    PART_CATEGORY_OTHER,
    ALL_PART_CATEGORIES,
    # Seasons
    SEASON_SUMMER,
    SEASON_MONSOON,
    SEASON_SPRING,
    SEASON_WINTER,
    ALL_SEASONS,
    SEASON_MONTHS,
    # Terrain Types
    TERRAIN_URBAN,
    TERRAIN_HIGHWAY,
    TERRAIN_MIXED,
    TERRAIN_OFF_ROAD,
    TERRAIN_HILLY,
    ALL_TERRAIN_TYPES,
    # Model Configuration
    DEFAULT_CONFIDENCE_THRESHOLD,
    MIN_CONFIDENCE_THRESHOLD,
    MAX_CONFIDENCE_THRESHOLD,
    MAX_RECOMMENDATIONS,
    MIN_RECOMMENDATIONS,
    # Validation Thresholds
    MIN_ODOMETER,
    MAX_ODOMETER,
    MIN_VEHICLE_AGE_DAYS,
    MAX_VEHICLE_AGE_DAYS,
    MIN_SERVICE_INTERVAL_DAYS,
    MAX_SERVICE_INTERVAL_DAYS,
    # EMA Configuration
    DEFAULT_EMA_PERIODS,
    MIN_EMA_PERIODS,
    MAX_EMA_PERIODS,
    MIN_SERVICES_FOR_EMA,
    # Cache TTL
    CACHE_TTL_RECOMMENDATION,
    CACHE_TTL_VEHICLE,
    CACHE_TTL_SERVICE_HISTORY,
    CACHE_TTL_MODEL,
    CACHE_TTL_EMA,
    CACHE_TTL_PREDICTION,
    # API Configuration
    API_MAX_REQUEST_SIZE,
    API_TIMEOUT_SECONDS,
    API_MAX_CONCURRENT_REQUESTS,
    # Database Configuration
    DB_CONNECTION_TIMEOUT,
    DB_POOL_SIZE,
    DB_MAX_OVERFLOW,
    DB_POOL_RECYCLE,
    # Recommendation Status
    REC_STATUS_PENDING,
    REC_STATUS_ACCEPTED,
    REC_STATUS_REJECTED,
    REC_STATUS_IGNORED,
    REC_STATUS_EXPIRED,
    ALL_REC_STATUSES,
    # Feedback Actions
    FEEDBACK_ACTION_ACCEPTED,
    FEEDBACK_ACTION_REJECTED,
    FEEDBACK_ACTION_IGNORED,
    ALL_FEEDBACK_ACTIONS,
    # Model Training
    TRAIN_MIN_SAMPLES,
    TRAIN_TEST_SPLIT_RATIO,
    MIN_TRAINING_SET_SIZE,
    MIN_TEST_SET_SIZE,
    # Validation Metrics
    MIN_PRECISION_THRESHOLD,
    MIN_RECALL_THRESHOLD,
    MIN_F1_THRESHOLD,
    TARGET_PRECISION_THRESHOLD,
    TARGET_RECALL_THRESHOLD,
    TARGET_F1_THRESHOLD,
    # Model Drift
    DRIFT_WARNING_THRESHOLD,
    DRIFT_CRITICAL_THRESHOLD,
    # A/B Testing
    AB_TEST_SPLIT_RATIO_DEFAULT,
    AB_TEST_MIN_DURATION_DAYS,
    AB_TEST_MIN_SAMPLES,
    # Date Formats
    DATE_FORMAT_ISO,
    DATETIME_FORMAT_ISO,
    DATETIME_FORMAT_ISO_WITH_TZ,
    # Regex Patterns
    PATTERN_VEHICLE_ID,
    PATTERN_PART_CODE,
    PATTERN_EMAIL,
    PATTERN_PHONE,
    # Error Codes
    ERROR_CODE_VEHICLE_NOT_FOUND,
    ERROR_CODE_INVALID_INPUT,
    ERROR_CODE_MODEL_ERROR,
    ERROR_CODE_DATABASE_ERROR,
    ERROR_CODE_CACHE_ERROR,
    ERROR_CODE_VALIDATION_ERROR,
    ERROR_CODE_PREDICTION_ERROR,
    # Logging
    LOG_LEVEL_DEBUG,
    LOG_LEVEL_INFO,
    LOG_LEVEL_WARNING,
    LOG_LEVEL_ERROR,
    LOG_LEVEL_CRITICAL,
    ALL_LOG_LEVELS,
)

__all__ = [
    # DateTime utilities
    'parse_date',
    'to_utc',
    'format_iso',
    'days_between',
    'months_between',
    'is_date_in_range',
    'get_current_utc',
    'add_days',
    'add_months',
    'get_age_in_days',
    'get_season',
    'is_weekend',
    # Hashing utilities
    'hash_string',
    'hash_to_int',
    'get_ab_test_group',
    'generate_cache_key',
    'hash_dict',
    # JSON utilities
    'DateTimeEncoder',
    'safe_dumps',
    'safe_loads',
    'safe_load_file',
    'safe_dump_file',
    'is_valid_json',
    'parse_datetime_from_json',
    # Validation utilities
    'is_valid_vehicle_id',
    'is_valid_part_code',
    'is_valid_odometer',
    'is_valid_confidence_score',
    'is_valid_email',
    'is_valid_phone',
    'validate_not_none',
    'validate_not_empty',
    'validate_in_range',
    'validate_one_of',
    'validate_type',
    'validate_list_of_type',
    'sanitize_string',
    'normalize_part_code',
    # Cache utilities
    'CACHE_PREFIX_RECOMMENDATION',
    'CACHE_PREFIX_VEHICLE',
    'CACHE_PREFIX_SERVICE_HISTORY',
    'CACHE_PREFIX_MODEL',
    'CACHE_PREFIX_EMA',
    'CACHE_PREFIX_PREDICTION',
    'get_recommendation_cache_key',
    'get_vehicle_cache_key',
    'get_service_history_cache_key',
    'get_model_cache_key',
    'get_ema_cache_key',
    'get_prediction_cache_key',
    'generate_features_hash',
    'get_cache_key_with_ttl',
    'invalidate_vehicle_caches',
    # Constants (partial list - all constants are exported)
    'PART_CATEGORY_BRAKE',
    'PART_CATEGORY_ENGINE',
    'DEFAULT_CONFIDENCE_THRESHOLD',
    'MIN_PRECISION_THRESHOLD',
    'TARGET_PRECISION_THRESHOLD',
    # ... (all constants from constants module are available via import)
]
