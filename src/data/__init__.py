"""
Data access layer for AI Parts Recommendation System.

This package contains all database models, repositories, and data access patterns.
"""

from .models import Vehicle, ServiceHistory, PartRecommendation, UserFeedback
from .repositories import VehicleRepository, ServiceHistoryRepository, RecommendationRepository
from .database import get_db_session, init_database

__all__ = [
    'Vehicle',
    'ServiceHistory', 
    'PartRecommendation',
    'UserFeedback',
    'VehicleRepository',
    'ServiceHistoryRepository',
    'RecommendationRepository',
    'get_db_session',
    'init_database'
]
