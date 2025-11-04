"""
Configuration package for AI Parts Recommendation System.
"""

from .settings import Settings, get_settings, is_development, is_production

__all__ = ['Settings', 'get_settings', 'is_development', 'is_production']
