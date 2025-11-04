"""
Database configuration and session management.

This module handles database connections, session management, and initialization.
"""

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
from typing import Generator, Optional
import logging

from ..config.settings import get_settings
from .models import Base

logger = logging.getLogger(__name__)

# Global variables
engine = None
SessionLocal = None


def init_database() -> None:
    """Initialize database connection and create tables."""
    global engine, SessionLocal
    
    settings = get_settings()
    
    # Create engine with connection pooling
    engine = create_engine(
        settings.database_url,
        poolclass=QueuePool,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_pre_ping=True,
        echo=settings.debug,  # Log SQL queries in debug mode
        echo_pool=settings.debug
    )
    
    # Create session factory
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # Add connection event listeners
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        """Set database connection parameters."""
        if 'postgresql' in settings.database_url:
            # PostgreSQL specific settings
            with dbapi_connection.cursor() as cursor:
                cursor.execute("SET timezone TO 'UTC'")
                cursor.execute("SET statement_timeout TO '30s'")
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    logger.info(f"Database initialized with URL: {settings.database_url}")


def get_engine():
    """Get database engine."""
    if engine is None:
        init_database()
    return engine


def get_session_factory():
    """Get session factory."""
    if SessionLocal is None:
        init_database()
    return SessionLocal


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.
    
    Provides automatic session management with proper error handling
    and cleanup.
    
    Yields:
        Session: SQLAlchemy session
        
    Raises:
        Exception: Any database-related exceptions
    """
    session = None
    try:
        session_factory = get_session_factory()
        session = session_factory()
        yield session
        session.commit()
        logger.debug("Database session committed successfully")
    except Exception as e:
        if session:
            session.rollback()
            logger.error(f"Database session rolled back due to error: {e}")
        raise
    finally:
        if session:
            session.close()
            logger.debug("Database session closed")


def get_db_session_dependency() -> Generator[Session, None, None]:
    """
    Dependency for FastAPI to get database session.
    
    This is used with FastAPI's Depends() for dependency injection.
    """
    with get_db_session() as session:
        yield session


def test_connection() -> bool:
    """
    Test database connection.
    
    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        with get_db_session() as session:
            session.execute("SELECT 1")
            logger.info("Database connection test successful")
            return True
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False


def get_connection_info() -> dict:
    """
    Get database connection information.
    
    Returns:
        dict: Connection information
    """
    settings = get_settings()
    return {
        'database_url': settings.database_url,
        'pool_size': settings.db_pool_size,
        'max_overflow': settings.db_max_overflow,
        'engine_info': str(engine) if engine else None
    }


def close_connections() -> None:
    """Close all database connections."""
    global engine
    if engine:
        engine.dispose()
        logger.info("Database connections closed")


# Initialize database on module import
try:
    init_database()
except Exception as e:
    logger.error(f"Failed to initialize database: {e}")
    raise
