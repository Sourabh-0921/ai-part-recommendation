"""
Configuration management for AI Parts Recommendation System.

This module handles all application settings using Pydantic for validation
and type safety.
"""

from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from typing import Optional, List
import os


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Database Configuration
    database_url: str = Field(..., env='DATABASE_URL')
    db_pool_size: int = Field(20, env='DB_POOL_SIZE')
    db_max_overflow: int = Field(40, env='DB_MAX_OVERFLOW')
    
    # MongoDB Configuration
    mongodb_url: str = Field(..., env='MONGODB_URL')
    
    # Redis Configuration
    redis_host: str = Field('localhost', env='REDIS_HOST')
    redis_port: int = Field(6379, env='REDIS_PORT')
    redis_password: Optional[str] = Field(None, env='REDIS_PASSWORD')
    redis_db: int = Field(0, env='REDIS_DB')
    
    # Model Configuration
    model_path: str = Field('models/latest', env='MODEL_PATH')
    confidence_threshold: float = Field(80.0, env='CONFIDENCE_THRESHOLD')
    model_version: str = Field('1.0.0', env='MODEL_VERSION')
    
    # API Configuration
    api_host: str = Field('0.0.0.0', env='API_HOST')
    api_port: int = Field(8000, env='API_PORT')
    api_title: str = Field('AI Parts Recommendation API', env='API_TITLE')
    api_version: str = Field('1.0.0', env='API_VERSION')
    
    # EMA Configuration
    ema_periods: int = Field(6, env='EMA_PERIODS')
    min_services_for_ema: int = Field(2, env='MIN_SERVICES_FOR_EMA')
    
    # Logging Configuration
    log_level: str = Field('INFO', env='LOG_LEVEL')
    log_format: str = Field('json', env='LOG_FORMAT')
    log_file: str = Field('app.log', env='LOG_FILE')
    log_max_bytes: int = Field(10485760, env='LOG_MAX_BYTES')  # 10MB
    log_backup_count: int = Field(5, env='LOG_BACKUP_COUNT')
    
    # Security
    secret_key: str = Field(..., env='SECRET_KEY')
    jwt_algorithm: str = Field('HS256', env='JWT_ALGORITHM')
    jwt_expire_minutes: int = Field(30, env='JWT_EXPIRE_MINUTES')
    
    # Performance
    max_concurrent_requests: int = Field(500, env='MAX_CONCURRENT_REQUESTS')
    cache_ttl_seconds: int = Field(1800, env='CACHE_TTL_SECONDS')  # 30 minutes
    batch_size: int = Field(100, env='BATCH_SIZE')
    
    # Monitoring
    prometheus_port: int = Field(9090, env='PROMETHEUS_PORT')
    metrics_enabled: bool = Field(True, env='METRICS_ENABLED')
    
    # Airflow
    airflow_home: str = Field('/opt/airflow', env='AIRFLOW_HOME')
    airflow_db_url: str = Field(..., env='AIRFLOW_DB_URL')
    
    # Message Queue
    rabbitmq_url: Optional[str] = Field(None, env='RABBITMQ_URL')
    kafka_bootstrap_servers: Optional[str] = Field(None, env='KAFKA_BOOTSTRAP_SERVERS')
    
    # External APIs
    oem_api_url: Optional[str] = Field(None, env='OEM_API_URL')
    weather_api_url: Optional[str] = Field(None, env='WEATHER_API_URL')
    weather_api_key: Optional[str] = Field(None, env='WEATHER_API_KEY')
    
    # Environment
    environment: str = Field('development', env='ENVIRONMENT')
    debug: bool = Field(False, env='DEBUG')
    
    @field_validator('confidence_threshold')
    @classmethod
    def validate_confidence_threshold(cls, v):
        """Validate confidence threshold is between 0 and 100."""
        if not 0 <= v <= 100:
            raise ValueError('Confidence threshold must be between 0 and 100')
        return v
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level is valid."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Log level must be one of {valid_levels}')
        return v.upper()
    
    @field_validator('environment')
    @classmethod
    def validate_environment(cls, v):
        """Validate environment is valid."""
        valid_envs = ['development', 'staging', 'production']
        if v.lower() not in valid_envs:
            raise ValueError(f'Environment must be one of {valid_envs}')
        return v.lower()
    
    class Config:
        """Pydantic configuration."""
        env_file = '.env'
        case_sensitive = False
        validate_assignment = True


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings


def is_development() -> bool:
    """Check if running in development mode."""
    return settings.environment == 'development'


def is_production() -> bool:
    """Check if running in production mode."""
    return settings.environment == 'production'
