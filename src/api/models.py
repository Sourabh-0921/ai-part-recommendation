"""
Pydantic models for API request/response validation.

This module defines all request and response models for the FastAPI endpoints.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

from ..utils import (
    is_valid_vehicle_id,
    is_valid_odometer,
    MAX_ODOMETER,
    PATTERN_VEHICLE_ID
)


class FeedbackType(str, Enum):
    """User feedback types."""
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    MODIFIED = "MODIFIED"


class ServiceType(str, Enum):
    """Service types."""
    REGULAR = "REGULAR"
    MAJOR = "MAJOR"
    EMERGENCY = "EMERGENCY"


class UsageCategory(str, Enum):
    """EMA usage categories."""
    HIGH_USAGE = "HIGH_USAGE"
    MEDIUM_USAGE = "MEDIUM_USAGE"
    LOW_USAGE = "LOW_USAGE"


# Request Models
class RecommendationRequest(BaseModel):
    """Request model for generating recommendations."""
    
    vehicle_id: str = Field(
        ..., 
        description="Vehicle identifier (format: 2-letter state code + 2-digit RTO + 1-2 letters + 4 digits)",
        min_length=5,
        max_length=20,
        example="MH12AB1234"
    )
    current_odometer: float = Field(
        ..., 
        gt=0, 
        description="Current odometer reading in kilometers",
        example=15250.5
    )
    customer_complaints: Optional[str] = Field(
        None, 
        description="Customer complaint text describing issues",
        max_length=1000,
        example="Brake making noise when stopping"
    )
    dealer_code: str = Field(
        ..., 
        description="Dealer code for location-based adjustments",
        min_length=3,
        max_length=50,
        example="DLR_MUM_01"
    )
    
    @validator('vehicle_id')
    def validate_vehicle_id(cls, v):
        """Validate vehicle ID format using utility function."""
        # Normalize to uppercase first
        v_upper = v.upper() if isinstance(v, str) else str(v).upper()
        
        # Use utility function for validation
        if not is_valid_vehicle_id(v_upper):
            raise ValueError(
                f'Vehicle ID must be in valid format (alphanumeric, 6-20 characters). '
                f'Got: {v}'
            )
        return v_upper
    
    @validator('current_odometer')
    def validate_odometer(cls, v):
        """Validate odometer reading using utility function."""
        if not is_valid_odometer(v, max_value=MAX_ODOMETER):
            raise ValueError(
                f'Odometer reading must be between 0 and {MAX_ODOMETER} km. Got: {v}'
            )
        return float(v)


class FeedbackRequest(BaseModel):
    """Request model for user feedback."""
    
    recommendation_id: int = Field(..., description="Recommendation ID to provide feedback for")
    feedback_type: FeedbackType = Field(..., description="Type of feedback")
    feedback_reason: Optional[str] = Field(None, description="Reason for feedback", max_length=500)
    alternative_part: Optional[str] = Field(None, description="Alternative part code if applicable", max_length=50)
    actual_cost: Optional[float] = Field(None, description="Actual cost if known", gt=0)
    service_date: Optional[datetime] = Field(None, description="Date when service was performed")
    
    @validator('alternative_part')
    def validate_alternative_part(cls, v):
        """Validate alternative part code format using utility function."""
        if v is None:
            return v
        
        from ..utils import normalize_part_code, is_valid_part_code
        
        # Normalize and validate part code
        normalized = normalize_part_code(v)
        if not is_valid_part_code(normalized):
            raise ValueError(
                f'Alternative part code must be in format: 2 letters + 3 digits (e.g., BP001). '
                f'Got: {v}'
            )
        return normalized


class BatchRecommendationRequest(BaseModel):
    """Request model for batch recommendations."""
    
    vehicles: List[RecommendationRequest] = Field(
        ..., 
        description="List of vehicles for batch processing",
        min_items=1,
        max_items=100
    )


# Response Models
class PartRecommendation(BaseModel):
    """Single part recommendation response."""
    
    rank: int = Field(..., description="Position in sorted list (1-10)", ge=1, le=10)
    part_code: str = Field(..., description="Unique part identifier", example="BP001")
    part_name: str = Field(..., description="Human-readable part name", example="Brake Pads - Front")
    confidence_score: float = Field(
        ..., 
        description="Confidence percentage (80-100)", 
        ge=80.0, 
        le=100.0,
        example=95.2
    )
    category: str = Field(..., description="Part category", example="Brakes")
    estimated_cost: float = Field(..., description="Estimated cost in INR", ge=0, example=2500.0)
    reasoning: Dict[str, Any] = Field(..., description="Explanation of why recommended")
    last_replaced: Optional[Dict[str, Any]] = Field(None, description="Replacement history if available")
    seasonal_adjustment: Optional[float] = Field(None, description="Seasonal adjustment applied")
    terrain_adjustment: Optional[float] = Field(None, description="Terrain adjustment applied")
    final_confidence: float = Field(..., description="Final confidence after adjustments")


class VehicleInfo(BaseModel):
    """Vehicle information in response."""
    
    vehicle_id: str = Field(..., description="Vehicle identifier")
    vehicle_model: str = Field(..., description="Vehicle model")
    current_odometer: float = Field(..., description="Current odometer reading")
    dealer_code: str = Field(..., description="Dealer code")
    region_code: str = Field(..., description="Region code")
    terrain_type: str = Field(..., description="Terrain type")
    season_code: str = Field(..., description="Current season")
    ema_value: Optional[float] = Field(None, description="EMA value in km/month")
    ema_category: Optional[UsageCategory] = Field(None, description="EMA usage category")
    last_service_date: Optional[datetime] = Field(None, description="Last service date")


class RecommendationResponse(BaseModel):
    """Response model for recommendations."""
    
    status: str = Field(..., description="Response status", example="success")
    vehicle_info: VehicleInfo = Field(..., description="Vehicle information")
    recommendations: List[PartRecommendation] = Field(
        ..., 
        description="List of recommended parts (max 10)",
        max_items=10
    )
    total_estimated_cost: float = Field(..., description="Total estimated cost of all recommendations")
    model_version: str = Field(..., description="Model version used")
    timestamp: datetime = Field(..., description="Response timestamp")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")


class BatchRecommendationResponse(BaseModel):
    """Response model for batch recommendations."""
    
    status: str = Field(..., description="Response status")
    total_vehicles: int = Field(..., description="Total number of vehicles processed")
    successful_vehicles: int = Field(..., description="Number of successful recommendations")
    failed_vehicles: int = Field(..., description="Number of failed recommendations")
    results: List[RecommendationResponse] = Field(..., description="Individual results")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")
    timestamp: datetime = Field(..., description="Response timestamp")


class FeedbackResponse(BaseModel):
    """Response model for feedback submission."""
    
    status: str = Field(..., description="Response status")
    recommendation_id: int = Field(..., description="Recommendation ID")
    feedback_type: FeedbackType = Field(..., description="Feedback type")
    message: str = Field(..., description="Confirmation message")
    timestamp: datetime = Field(..., description="Response timestamp")


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Check timestamp")
    version: str = Field(..., description="API version")
    environment: str = Field(..., description="Environment")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")


class ModelStatusResponse(BaseModel):
    """Model status response."""
    
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_version: str = Field(..., description="Model version")
    model_path: str = Field(..., description="Model file path")
    last_updated: Optional[datetime] = Field(None, description="Last model update")
    confidence_threshold: float = Field(..., description="Confidence threshold")
    total_predictions: Optional[int] = Field(None, description="Total predictions made")
    average_confidence: Optional[float] = Field(None, description="Average confidence score")


class ErrorResponse(BaseModel):
    """Error response model."""
    
    status: str = Field(..., description="Error status", example="error")
    error_code: str = Field(..., description="Error code", example="VEHICLE_NOT_FOUND")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")


# Statistics Models
class RecommendationStats(BaseModel):
    """Recommendation statistics."""
    
    total_recommendations: int = Field(..., description="Total recommendations generated")
    average_confidence: float = Field(..., description="Average confidence score")
    acceptance_rate: float = Field(..., description="Recommendation acceptance rate")
    top_categories: List[Dict[str, Any]] = Field(..., description="Top recommended categories")
    model_performance: Dict[str, Any] = Field(..., description="Model performance metrics")


class VehicleStats(BaseModel):
    """Vehicle statistics."""
    
    total_vehicles: int = Field(..., description="Total vehicles in system")
    vehicles_with_ema: int = Field(..., description="Vehicles with EMA calculated")
    usage_distribution: Dict[str, int] = Field(..., description="Usage category distribution")
    average_odometer: float = Field(..., description="Average odometer reading")
    top_models: List[Dict[str, Any]] = Field(..., description="Most common vehicle models")
