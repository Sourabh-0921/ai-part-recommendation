"""
Recommendation API endpoints.

This module contains all endpoints related to parts recommendations.
"""

import time
import logging
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.responses import JSONResponse

from src.api.models import (
    RecommendationRequest, 
    RecommendationResponse, 
    BatchRecommendationRequest,
    BatchRecommendationResponse,
    PartRecommendation,
    VehicleInfo,
    ErrorResponse
)
from src.api.dependencies import (
    get_recommendation_service,
    get_current_user_id,
    check_rate_limit,
    get_request_id,
    validate_vehicle_access,
    get_batch_size
)
from src.api.exceptions import (
    VehicleNotFoundError,
    PredictionError,
    DataQualityError,
    BatchProcessingError
)
from src.services.recommendation_service import RecommendationService
from src.models.ema_calculator import EMACalculator
from src.data.models import Vehicle

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/recommendations", tags=["recommendations"])


@router.post(
    "/generate",
    response_model=RecommendationResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate parts recommendations",
    description="Generate top 10 parts recommendations for a vehicle based on ML model predictions"
)
async def generate_recommendations(
    request: RecommendationRequest,
    background_tasks: BackgroundTasks,
    service: RecommendationService = Depends(get_recommendation_service),
    user_id: str = Depends(get_current_user_id),
    request_id: str = Depends(get_request_id),
    _: bool = Depends(check_rate_limit)
) -> RecommendationResponse:
    """
    Generate parts recommendations for a vehicle.
    
    This endpoint coordinates the entire recommendation pipeline:
    1. Fetch vehicle features from database
    2. Generate base predictions using LightGBM model
    3. Apply seasonal and terrain adjustments
    4. Filter and sort recommendations
    5. Check rejection history
    
    Args:
        request: Recommendation request with vehicle details
        background_tasks: FastAPI background tasks
        service: Recommendation service instance
        user_id: Current user ID
        request_id: Unique request ID
        
    Returns:
        RecommendationResponse with top 10 recommendations
        
    Raises:
        HTTPException: For various error conditions
    """
    start_time = time.time()
    
    try:
        logger.info(
            f"Generating recommendations for vehicle {request.vehicle_id} "
            f"(user: {user_id}, request: {request_id})"
        )
        
        # Validate vehicle access
        vehicle_id = validate_vehicle_access(request.vehicle_id, user_id)
        
        # Generate recommendations
        recommendations_data = await service.generate_recommendations(
            vehicle_id=vehicle_id,
            odometer=request.current_odometer,
            complaints=request.customer_complaints,
            dealer_code=request.dealer_code
        )
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Build response
        response = RecommendationResponse(
            status="success",
            vehicle_info=VehicleInfo(**recommendations_data["vehicle_info"]),
            recommendations=[
                PartRecommendation(**rec) for rec in recommendations_data["recommendations"]
            ],
            total_estimated_cost=recommendations_data["total_estimated_cost"],
            model_version=recommendations_data["model_version"],
            timestamp=recommendations_data["timestamp"],
            processing_time_ms=processing_time
        )
        
        # Log successful recommendation
        logger.info(
            f"Successfully generated {len(response.recommendations)} recommendations "
            f"for vehicle {vehicle_id} in {processing_time:.2f}ms"
        )
        
        # Store recommendation in background (for analytics)
        background_tasks.add_task(
            _store_recommendation_analytics,
            vehicle_id=vehicle_id,
            recommendations_count=len(response.recommendations),
            processing_time=processing_time,
            user_id=user_id
        )
        
        return response
        
    except VehicleNotFoundError as e:
        logger.warning(f"Vehicle not found: {request.vehicle_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Vehicle not found: {request.vehicle_id}"
        )
        
    except PredictionError as e:
        logger.error(f"Prediction error for vehicle {request.vehicle_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate recommendations"
        )
        
    except DataQualityError as e:
        logger.warning(f"Data quality error for vehicle {request.vehicle_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid vehicle data: {e.message}"
        )
        
    except Exception as e:
        logger.error(f"Unexpected error generating recommendations: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post(
    "/batch",
    response_model=BatchRecommendationResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate batch recommendations",
    description="Generate recommendations for multiple vehicles in a single request"
)
async def generate_batch_recommendations(
    request: BatchRecommendationRequest,
    background_tasks: BackgroundTasks,
    service: RecommendationService = Depends(get_recommendation_service),
    user_id: str = Depends(get_current_user_id),
    batch_size: int = Depends(get_batch_size),
    _: bool = Depends(check_rate_limit)
) -> BatchRecommendationResponse:
    """
    Generate recommendations for multiple vehicles.
    
    Args:
        request: Batch recommendation request
        background_tasks: FastAPI background tasks
        service: Recommendation service
        user_id: Current user ID
        batch_size: Maximum batch size
        
    Returns:
        BatchRecommendationResponse with results for all vehicles
    """
    start_time = time.time()
    
    try:
        # Validate batch size
        if len(request.vehicles) > batch_size:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Batch size exceeds maximum of {batch_size} vehicles"
            )
        
        logger.info(f"Processing batch recommendations for {len(request.vehicles)} vehicles")
        
        results = []
        successful_count = 0
        failed_count = 0
        
        # Process each vehicle
        for vehicle_request in request.vehicles:
            try:
                # Validate vehicle access
                vehicle_id = validate_vehicle_access(vehicle_request.vehicle_id, user_id)
                
                # Generate recommendations
                recommendations_data = await service.generate_recommendations(
                    vehicle_id=vehicle_id,
                    odometer=vehicle_request.current_odometer,
                    complaints=vehicle_request.customer_complaints,
                    dealer_code=vehicle_request.dealer_code
                )
                
                # Build response for this vehicle
                response = RecommendationResponse(
                    status="success",
                    vehicle_info=VehicleInfo(**recommendations_data["vehicle_info"]),
                    recommendations=[
                        PartRecommendation(**rec) for rec in recommendations_data["recommendations"]
                    ],
                    total_estimated_cost=recommendations_data["total_estimated_cost"],
                    model_version=recommendations_data["model_version"],
                    timestamp=recommendations_data["timestamp"]
                )
                
                results.append(response)
                successful_count += 1
                
            except Exception as e:
                logger.warning(f"Failed to process vehicle {vehicle_request.vehicle_id}: {e}")
                failed_count += 1
                
                # Create error response for this vehicle
                error_response = RecommendationResponse(
                    status="error",
                    vehicle_info=VehicleInfo(
                        vehicle_id=vehicle_request.vehicle_id,
                        vehicle_model="Unknown",
                        current_odometer=vehicle_request.current_odometer,
                        dealer_code=vehicle_request.dealer_code,
                        region_code="Unknown",
                        terrain_type="Unknown",
                        season_code="Unknown"
                    ),
                    recommendations=[],
                    total_estimated_cost=0.0,
                    model_version="unknown",
                    timestamp=time.time()
                )
                results.append(error_response)
        
        # Calculate total processing time
        total_processing_time = (time.time() - start_time) * 1000
        
        # Log batch results
        logger.info(
            f"Batch processing completed: {successful_count} successful, "
            f"{failed_count} failed in {total_processing_time:.2f}ms"
        )
        
        # Store batch analytics in background
        background_tasks.add_task(
            _store_batch_analytics,
            total_vehicles=len(request.vehicles),
            successful_count=successful_count,
            failed_count=failed_count,
            processing_time=total_processing_time,
            user_id=user_id
        )
        
        return BatchRecommendationResponse(
            status="completed",
            total_vehicles=len(request.vehicles),
            successful_vehicles=successful_count,
            failed_vehicles=failed_count,
            results=results,
            processing_time_ms=total_processing_time,
            timestamp=time.time()
        )
        
    except Exception as e:
        logger.error(f"Batch processing error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch processing failed"
        )


@router.get(
    "/history/{vehicle_id}",
    response_model=List[RecommendationResponse],
    summary="Get recommendation history",
    description="Get historical recommendations for a vehicle"
)
async def get_recommendation_history(
    vehicle_id: str,
    limit: int = 10,
    service: RecommendationService = Depends(get_recommendation_service),
    user_id: str = Depends(get_current_user_id)
) -> List[RecommendationResponse]:
    """
    Get historical recommendations for a vehicle.
    
    Args:
        vehicle_id: Vehicle ID
        limit: Maximum number of historical recommendations
        service: Recommendation service
        user_id: Current user ID
        
    Returns:
        List of historical recommendations
    """
    try:
        # Validate vehicle access
        validate_vehicle_access(vehicle_id, user_id)
        
        logger.info(f"Fetching recommendation history for vehicle {vehicle_id}")
        
        # Get historical recommendations
        history = await service.get_recommendation_history(
            vehicle_id=vehicle_id,
            limit=limit
        )
        
        return [
            RecommendationResponse(**rec) for rec in history
        ]
        
    except Exception as e:
        logger.error(f"Error fetching recommendation history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch recommendation history"
        )


async def _store_recommendation_analytics(
    vehicle_id: str,
    recommendations_count: int,
    processing_time: float,
    user_id: str
) -> None:
    """Store recommendation analytics in background."""
    try:
        # TODO: Implement analytics storage
        logger.debug(f"Stored analytics for vehicle {vehicle_id}")
    except Exception as e:
        logger.warning(f"Failed to store analytics: {e}")


async def _store_batch_analytics(
    total_vehicles: int,
    successful_count: int,
    failed_count: int,
    processing_time: float,
    user_id: str
) -> None:
    """Store batch analytics in background."""
    try:
        # TODO: Implement batch analytics storage
        logger.debug(f"Stored batch analytics for {total_vehicles} vehicles")
    except Exception as e:
        logger.warning(f"Failed to store batch analytics: {e}")
