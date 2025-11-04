"""
Main FastAPI application for AI Parts Recommendation System.

This module creates and configures the FastAPI application with all routes,
middleware, and error handlers.
"""

import time
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import uvicorn

from .models import ErrorResponse
from .exceptions import APIException
from .health import router as health_router
from .recommendations import router as recommendations_router
from .feedback import router as feedback_router
from ..config.settings import get_settings
from ..config.logging import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Global startup time
_startup_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting AI Parts Recommendation API...")
    startup_time = time.time()
    
    try:
        # Initialize services
        settings = get_settings()
        logger.info(f"Application configured for {settings.environment} environment")
        
        # TODO: Initialize ML model
        # TODO: Initialize database connections
        # TODO: Initialize Redis connections
        # TODO: Load configuration from MongoDB
        
        startup_duration = time.time() - startup_time
        logger.info(f"Application startup completed in {startup_duration:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Application startup failed: {e}", exc_info=True)
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI Parts Recommendation API...")
    shutdown_start = time.time()
    
    try:
        # TODO: Close database connections
        # TODO: Close Redis connections
        # TODO: Save any pending data
        
        shutdown_duration = time.time() - shutdown_start
        logger.info(f"Application shutdown completed in {shutdown_duration:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Application shutdown error: {e}", exc_info=True)


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application instance
    """
    settings = get_settings()
    
    # Create FastAPI app
    app = FastAPI(
        title=settings.api_title,
        description="AI-powered vehicle parts recommendation system using LightGBM",
        version=settings.api_version,
        # Expose OpenAPI/Swagger UI in all environments
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )
    
    # Add middleware
    _add_middleware(app, settings)
    
    # Add exception handlers
    _add_exception_handlers(app)
    
    # Add routes
    _add_routes(app)
    
    # Add startup event
    @app.on_event("startup")
    async def startup_event():
        """Application startup event."""
        logger.info("AI Parts Recommendation API started successfully")
    
    # Add shutdown event
    @app.on_event("shutdown")
    async def shutdown_event():
        """Application shutdown event."""
        logger.info("AI Parts Recommendation API shutting down")
    
    return app


def _add_middleware(app: FastAPI, settings) -> None:
    """Add middleware to the FastAPI application."""
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.debug else ["https://yourdomain.com"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )
    
    # Trusted host middleware
    if not settings.debug:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["yourdomain.com", "*.yourdomain.com"]
        )
    
    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log all HTTP requests."""
        start_time = time.time()
        
        # Log request
        logger.info(
            f"Request: {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )
        
        # Process request
        response = await call_next(request)
        
        # Log response
        process_time = time.time() - start_time
        logger.info(
            f"Response: {response.status_code} "
            f"in {process_time:.3f}s for {request.method} {request.url.path}"
        )
        
        # Add processing time header
        response.headers["X-Process-Time"] = str(process_time)
        
        return response


def _add_exception_handlers(app: FastAPI) -> None:
    """Add exception handlers to the FastAPI application."""
    
    @app.exception_handler(APIException)
    async def api_exception_handler(request: Request, exc: APIException):
        """Handle custom API exceptions."""
        logger.warning(f"API exception: {exc.error_code} - {exc.message}")
        
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                status="error",
                error_code=exc.error_code,
                message=exc.message,
                details=exc.details,
                timestamp=time.time()
            ).dict()
        )
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions."""
        logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")
        
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                status="error",
                error_code=f"HTTP_{exc.status_code}",
                message=str(exc.detail),
                timestamp=time.time()
            ).dict()
        )
    
    @app.exception_handler(StarletteHTTPException)
    async def starlette_exception_handler(request: Request, exc: StarletteHTTPException):
        """Handle Starlette HTTP exceptions."""
        logger.warning(f"Starlette exception: {exc.status_code} - {exc.detail}")
        
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                status="error",
                error_code=f"STARLETTE_{exc.status_code}",
                message=str(exc.detail),
                timestamp=time.time()
            ).dict()
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle request validation errors."""
        logger.warning(f"Validation error: {exc.errors()}")
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                status="error",
                error_code="VALIDATION_ERROR",
                message="Request validation failed",
                details={"errors": exc.errors()},
                timestamp=time.time()
            ).dict()
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions."""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                status="error",
                error_code="INTERNAL_SERVER_ERROR",
                message="An unexpected error occurred",
                timestamp=time.time()
            ).dict()
        )


def _add_routes(app: FastAPI) -> None:
    """Add routes to the FastAPI application."""
    
    # Health and status routes
    app.include_router(health_router)
    
    # Recommendation routes
    app.include_router(recommendations_router)
    
    # Feedback routes
    app.include_router(feedback_router)
    
    # Root endpoint
    @app.get("/", tags=["root"])
    async def root():
        """Root endpoint with basic API information."""
        return {
            "message": "AI Parts Recommendation API",
            "version": "1.0.0",
            "status": "running",
            "timestamp": time.time(),
            "uptime_seconds": time.time() - _startup_time
        }


# Create the application instance
app = create_app()


def run_server():
    """Run the development server."""
    settings = get_settings()
    
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        access_log=True
    )


if __name__ == "__main__":
    run_server()
