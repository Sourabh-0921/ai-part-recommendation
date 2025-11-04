"""
Feedback API endpoints.

This module contains all endpoints related to user feedback on recommendations.
"""

import time
import logging
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks

from .models import (
    FeedbackRequest,
    FeedbackResponse,
    RecommendationStats,
    ErrorResponse
)
from .dependencies import (
    get_part_recommendation_repository,
    get_current_user_id,
    check_rate_limit,
    get_request_id
)
from src.api.exceptions import ValidationError, DatabaseError
from src.data.repositories import RecommendationRepository
from src.data.models import UserFeedback
from src.utils import (
    normalize_part_code,
    is_valid_part_code
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/feedback", tags=["feedback"])


@router.post(
    "/submit",
    response_model=FeedbackResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit user feedback",
    description="Submit user feedback on a recommendation"
)
async def submit_feedback(
    request: FeedbackRequest,
    background_tasks: BackgroundTasks,
    repository: RecommendationRepository = Depends(get_part_recommendation_repository),
    user_id: str = Depends(get_current_user_id),
    request_id: str = Depends(get_request_id),
    _: bool = Depends(check_rate_limit)
) -> FeedbackResponse:
    """
    Submit user feedback on a recommendation.
    
    This endpoint allows users to provide feedback on recommendations,
    which is used to improve the ML model through retraining.
    
    Args:
        request: Feedback request with recommendation details
        background_tasks: FastAPI background tasks
        repository: Part recommendation repository
        user_id: Current user ID
        request_id: Unique request ID
        
    Returns:
        FeedbackResponse confirming feedback submission
        
    Raises:
        HTTPException: For various error conditions
    """
    try:
        logger.info(
            f"Submitting feedback for recommendation {request.recommendation_id} "
            f"(user: {user_id}, request: {request_id})"
        )
        
        # Validate recommendation exists
        recommendation = await repository.get_by_id(request.recommendation_id)
        if not recommendation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Recommendation {request.recommendation_id} not found"
            )
        
        # Validate and normalize alternative part code if provided
        normalized_alternative_part = None
        if request.alternative_part:
            normalized_alternative_part = normalize_part_code(request.alternative_part)
            if not is_valid_part_code(normalized_alternative_part):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid alternative part code format: {request.alternative_part}"
                )
        
        # Create feedback record
        feedback = UserFeedback(
            recommendation_id=request.recommendation_id,
            user_id=user_id,
            feedback_type=request.feedback_type.value,
            feedback_reason=request.feedback_reason,
            alternative_part=normalized_alternative_part,
            actual_cost=request.actual_cost,
            service_date=request.service_date
        )
        
        # Save feedback
        await repository.create_feedback(feedback)
        
        # Update recommendation with feedback
        await repository.update_feedback_status(
            recommendation_id=request.recommendation_id,
            is_accepted=request.feedback_type.value == "ACCEPTED",
            feedback_date=time.time()
        )
        
        # Store feedback analytics in background
        background_tasks.add_task(
            _store_feedback_analytics,
            recommendation_id=request.recommendation_id,
            feedback_type=request.feedback_type.value,
            user_id=user_id
        )
        
        logger.info(f"Successfully submitted feedback for recommendation {request.recommendation_id}")
        
        return FeedbackResponse(
            status="success",
            recommendation_id=request.recommendation_id,
            feedback_type=request.feedback_type,
            message="Feedback submitted successfully",
            timestamp=time.time()
        )
        
    except ValidationError as e:
        logger.warning(f"Validation error in feedback submission: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid feedback data: {e.message}"
        )
        
    except DatabaseError as e:
        logger.error(f"Database error in feedback submission: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save feedback"
        )
        
    except Exception as e:
        logger.error(f"Unexpected error submitting feedback: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get(
    "/stats",
    response_model=RecommendationStats,
    summary="Get recommendation statistics",
    description="Get statistics about recommendations and feedback"
)
async def get_recommendation_stats(
    repository: RecommendationRepository = Depends(get_part_recommendation_repository),
    user_id: str = Depends(get_current_user_id),
    days: int = 30
) -> RecommendationStats:
    """
    Get recommendation statistics.
    
    Args:
        repository: Part recommendation repository
        user_id: Current user ID
        days: Number of days to include in statistics
        
    Returns:
        RecommendationStats with various metrics
    """
    try:
        logger.info(f"Fetching recommendation statistics for user {user_id}")
        
        # Get statistics from repository
        stats = await repository.get_recommendation_stats(
            user_id=user_id,
            days=days
        )
        
        return RecommendationStats(**stats)
        
    except Exception as e:
        logger.error(f"Error fetching recommendation stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch recommendation statistics"
        )


@router.get(
    "/history/{vehicle_id}",
    response_model=List[Dict[str, Any]],
    summary="Get feedback history",
    description="Get feedback history for a vehicle"
)
async def get_feedback_history(
    vehicle_id: str,
    repository: RecommendationRepository = Depends(get_part_recommendation_repository),
    user_id: str = Depends(get_current_user_id),
    limit: int = 50
) -> List[Dict[str, Any]]:
    """
    Get feedback history for a vehicle.
    
    Args:
        vehicle_id: Vehicle ID
        repository: Part recommendation repository
        user_id: Current user ID
        limit: Maximum number of feedback records
        
    Returns:
        List of feedback records
    """
    try:
        logger.info(f"Fetching feedback history for vehicle {vehicle_id}")
        
        # Get feedback history
        history = await repository.get_feedback_history(
            vehicle_id=vehicle_id,
            user_id=user_id,
            limit=limit
        )
        
        return [feedback.to_dict() for feedback in history]
        
    except Exception as e:
        logger.error(f"Error fetching feedback history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch feedback history"
        )


@router.put(
    "/update/{feedback_id}",
    response_model=FeedbackResponse,
    summary="Update feedback",
    description="Update existing feedback"
)
async def update_feedback(
    feedback_id: int,
    request: FeedbackRequest,
    repository: RecommendationRepository = Depends(get_part_recommendation_repository),
    user_id: str = Depends(get_current_user_id)
) -> FeedbackResponse:
    """
    Update existing feedback.
    
    Args:
        feedback_id: Feedback ID to update
        request: Updated feedback data
        repository: Part recommendation repository
        user_id: Current user ID
        
    Returns:
        FeedbackResponse confirming update
    """
    try:
        logger.info(f"Updating feedback {feedback_id} for user {user_id}")
        
        # Check if feedback exists and belongs to user
        feedback = await repository.get_feedback_by_id(feedback_id)
        if not feedback:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Feedback {feedback_id} not found"
            )
        
        if feedback.user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to update this feedback"
            )
        
        # Validate and normalize alternative part code if provided
        normalized_alternative_part = None
        if request.alternative_part:
            normalized_alternative_part = normalize_part_code(request.alternative_part)
            if not is_valid_part_code(normalized_alternative_part):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid alternative part code format: {request.alternative_part}"
                )
        
        # Update feedback
        await repository.update_feedback(
            feedback_id=feedback_id,
            feedback_type=request.feedback_type.value,
            feedback_reason=request.feedback_reason,
            alternative_part=normalized_alternative_part,
            actual_cost=request.actual_cost,
            service_date=request.service_date
        )
        
        logger.info(f"Successfully updated feedback {feedback_id}")
        
        return FeedbackResponse(
            status="success",
            recommendation_id=request.recommendation_id,
            feedback_type=request.feedback_type,
            message="Feedback updated successfully",
            timestamp=time.time()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update feedback"
        )


@router.delete(
    "/delete/{feedback_id}",
    response_model=Dict[str, str],
    summary="Delete feedback",
    description="Delete existing feedback"
)
async def delete_feedback(
    feedback_id: int,
    repository: RecommendationRepository = Depends(get_part_recommendation_repository),
    user_id: str = Depends(get_current_user_id)
) -> Dict[str, str]:
    """
    Delete existing feedback.
    
    Args:
        feedback_id: Feedback ID to delete
        repository: Part recommendation repository
        user_id: Current user ID
        
    Returns:
        Confirmation message
    """
    try:
        logger.info(f"Deleting feedback {feedback_id} for user {user_id}")
        
        # Check if feedback exists and belongs to user
        feedback = await repository.get_feedback_by_id(feedback_id)
        if not feedback:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Feedback {feedback_id} not found"
            )
        
        if feedback.user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to delete this feedback"
            )
        
        # Delete feedback
        await repository.delete_feedback(feedback_id)
        
        logger.info(f"Successfully deleted feedback {feedback_id}")
        
        return {
            "status": "success",
            "message": f"Feedback {feedback_id} deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete feedback"
        )


async def _store_feedback_analytics(
    recommendation_id: int,
    feedback_type: str,
    user_id: str
) -> None:
    """Store feedback analytics in background."""
    try:
        # TODO: Implement feedback analytics storage
        logger.debug(f"Stored feedback analytics for recommendation {recommendation_id}")
    except Exception as e:
        logger.warning(f"Failed to store feedback analytics: {e}")
