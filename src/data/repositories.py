"""
Repository pattern implementation for data access.

This module contains repository classes that encapsulate data access logic
and provide a clean interface for database operations.
"""

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

from .models import Vehicle, ServiceHistory, PartRecommendation, UserFeedback, PartMaster, ModelPrediction
from .database import get_db_session

logger = logging.getLogger(__name__)


class BaseRepository:
    """Base repository class with common functionality."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def commit(self) -> None:
        """Commit the current transaction."""
        self.session.commit()
    
    def rollback(self) -> None:
        """Rollback the current transaction."""
        self.session.rollback()
    
    def refresh(self, instance) -> None:
        """Refresh an instance from the database."""
        self.session.refresh(instance)


class VehicleRepository(BaseRepository):
    """Repository for vehicle data access."""
    
    def get_by_id(self, vehicle_id: str) -> Optional[Vehicle]:
        """
        Get vehicle by ID.
        
        Args:
            vehicle_id: Vehicle identifier
            
        Returns:
            Vehicle object or None if not found
        """
        try:
            return self.session.query(Vehicle).filter_by(vehicle_id=vehicle_id).first()
        except Exception as e:
            logger.error(f"Error getting vehicle {vehicle_id}: {e}")
            raise
    
    def get_by_dealer(self, dealer_code: str, limit: int = 100) -> List[Vehicle]:
        """
        Get vehicles by dealer code.
        
        Args:
            dealer_code: Dealer identifier
            limit: Maximum number of vehicles to return
            
        Returns:
            List of Vehicle objects
        """
        try:
            return self.session.query(Vehicle).filter_by(
                dealer_code=dealer_code
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"Error getting vehicles for dealer {dealer_code}: {e}")
            raise
    
    def get_by_model(self, vehicle_model: str, limit: int = 100) -> List[Vehicle]:
        """
        Get vehicles by model.
        
        Args:
            vehicle_model: Vehicle model name
            limit: Maximum number of vehicles to return
            
        Returns:
            List of Vehicle objects
        """
        try:
            return self.session.query(Vehicle).filter_by(
                vehicle_model=vehicle_model
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"Error getting vehicles for model {vehicle_model}: {e}")
            raise
    
    def create(self, vehicle_data: Dict[str, Any]) -> Vehicle:
        """
        Create a new vehicle.
        
        Args:
            vehicle_data: Dictionary containing vehicle data
            
        Returns:
            Created Vehicle object
        """
        try:
            vehicle = Vehicle(**vehicle_data)
            self.session.add(vehicle)
            self.session.commit()
            logger.info(f"Created vehicle {vehicle.vehicle_id}")
            return vehicle
        except Exception as e:
            logger.error(f"Error creating vehicle: {e}")
            self.session.rollback()
            raise
    
    def update(self, vehicle_id: str, update_data: Dict[str, Any]) -> Optional[Vehicle]:
        """
        Update vehicle data.
        
        Args:
            vehicle_id: Vehicle identifier
            update_data: Dictionary containing updated data
            
        Returns:
            Updated Vehicle object or None if not found
        """
        try:
            vehicle = self.get_by_id(vehicle_id)
            if not vehicle:
                return None
            
            for key, value in update_data.items():
                if hasattr(vehicle, key):
                    setattr(vehicle, key, value)
            
            vehicle.last_updated = datetime.utcnow()
            self.session.commit()
            logger.info(f"Updated vehicle {vehicle_id}")
            return vehicle
        except Exception as e:
            logger.error(f"Error updating vehicle {vehicle_id}: {e}")
            self.session.rollback()
            raise
    
    def update_odometer(self, vehicle_id: str, new_odometer: float) -> Optional[Vehicle]:
        """
        Update vehicle odometer reading.
        
        Args:
            vehicle_id: Vehicle identifier
            new_odometer: New odometer reading
            
        Returns:
            Updated Vehicle object or None if not found
        """
        return self.update(vehicle_id, {'current_odometer': new_odometer})
    
    def update_ema(self, vehicle_id: str, ema_value: float, ema_category: str) -> Optional[Vehicle]:
        """
        Update vehicle EMA values.
        
        Args:
            vehicle_id: Vehicle identifier
            ema_value: EMA value in km/month
            ema_category: EMA category (HIGH_USAGE, MEDIUM_USAGE, LOW_USAGE)
            
        Returns:
            Updated Vehicle object or None if not found
        """
        return self.update(vehicle_id, {
            'ema_value': ema_value,
            'ema_category': ema_category
        })


class ServiceHistoryRepository(BaseRepository):
    """Repository for service history data access."""
    
    def get_by_vehicle(self, vehicle_id: str, limit: int = 50) -> List[ServiceHistory]:
        """
        Get service history for a vehicle.
        
        Args:
            vehicle_id: Vehicle identifier
            limit: Maximum number of records to return
            
        Returns:
            List of ServiceHistory objects
        """
        try:
            return self.session.query(ServiceHistory).filter_by(
                vehicle_id=vehicle_id
            ).order_by(desc(ServiceHistory.service_date)).limit(limit).all()
        except Exception as e:
            logger.error(f"Error getting service history for vehicle {vehicle_id}: {e}")
            raise
    
    def get_recent_services(self, vehicle_id: str, days: int = 365) -> List[ServiceHistory]:
        """
        Get recent service history for a vehicle.
        
        Args:
            vehicle_id: Vehicle identifier
            days: Number of days to look back
            
        Returns:
            List of recent ServiceHistory objects
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            return self.session.query(ServiceHistory).filter(
                and_(
                    ServiceHistory.vehicle_id == vehicle_id,
                    ServiceHistory.service_date >= cutoff_date
                )
            ).order_by(desc(ServiceHistory.service_date)).all()
        except Exception as e:
            logger.error(f"Error getting recent service history for vehicle {vehicle_id}: {e}")
            raise
    
    def create(self, service_data: Dict[str, Any]) -> ServiceHistory:
        """
        Create a new service history record.
        
        Args:
            service_data: Dictionary containing service data
            
        Returns:
            Created ServiceHistory object
        """
        try:
            service = ServiceHistory(**service_data)
            self.session.add(service)
            self.session.commit()
            logger.info(f"Created service history for vehicle {service.vehicle_id}")
            return service
        except Exception as e:
            logger.error(f"Error creating service history: {e}")
            self.session.rollback()
            raise
    
    def get_parts_replaced(self, vehicle_id: str, part_code: str = None) -> List[ServiceHistory]:
        """
        Get service history where specific parts were replaced.
        
        Args:
            vehicle_id: Vehicle identifier
            part_code: Optional specific part code to filter by
            
        Returns:
            List of ServiceHistory objects
        """
        try:
            query = self.session.query(ServiceHistory).filter_by(vehicle_id=vehicle_id)
            
            if part_code:
                query = query.filter(ServiceHistory.parts_replaced.contains([part_code]))
            
            return query.order_by(desc(ServiceHistory.service_date)).all()
        except Exception as e:
            logger.error(f"Error getting parts replacement history for vehicle {vehicle_id}: {e}")
            raise


class RecommendationRepository(BaseRepository):
    """Repository for recommendation data access."""
    
    def get_by_vehicle(self, vehicle_id: str, limit: int = 10) -> List[PartRecommendation]:
        """
        Get recommendations for a vehicle.
        
        Args:
            vehicle_id: Vehicle identifier
            limit: Maximum number of recommendations to return
            
        Returns:
            List of PartRecommendation objects
        """
        try:
            return self.session.query(PartRecommendation).filter_by(
                vehicle_id=vehicle_id
            ).order_by(asc(PartRecommendation.rank)).limit(limit).all()
        except Exception as e:
            logger.error(f"Error getting recommendations for vehicle {vehicle_id}: {e}")
            raise
    
    def get_recent_recommendations(self, vehicle_id: str, days: int = 30) -> List[PartRecommendation]:
        """
        Get recent recommendations for a vehicle.
        
        Args:
            vehicle_id: Vehicle identifier
            days: Number of days to look back
            
        Returns:
            List of recent PartRecommendation objects
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            return self.session.query(PartRecommendation).filter(
                and_(
                    PartRecommendation.vehicle_id == vehicle_id,
                    PartRecommendation.created_at >= cutoff_date
                )
            ).order_by(desc(PartRecommendation.created_at)).all()
        except Exception as e:
            logger.error(f"Error getting recent recommendations for vehicle {vehicle_id}: {e}")
            raise
    
    def create(self, recommendation_data: Dict[str, Any]) -> PartRecommendation:
        """
        Create a new recommendation.
        
        Args:
            recommendation_data: Dictionary containing recommendation data
            
        Returns:
            Created PartRecommendation object
        """
        try:
            recommendation = PartRecommendation(**recommendation_data)
            self.session.add(recommendation)
            self.session.commit()
            logger.info(f"Created recommendation for vehicle {recommendation.vehicle_id}")
            return recommendation
        except Exception as e:
            logger.error(f"Error creating recommendation: {e}")
            self.session.rollback()
            raise
    
    def create_batch(self, recommendations_data: List[Dict[str, Any]]) -> List[PartRecommendation]:
        """
        Create multiple recommendations in batch.
        
        Args:
            recommendations_data: List of dictionaries containing recommendation data
            
        Returns:
            List of created PartRecommendation objects
        """
        try:
            recommendations = []
            for data in recommendations_data:
                recommendation = PartRecommendation(**data)
                self.session.add(recommendation)
                recommendations.append(recommendation)
            
            self.session.commit()
            logger.info(f"Created {len(recommendations)} recommendations")
            return recommendations
        except Exception as e:
            logger.error(f"Error creating batch recommendations: {e}")
            self.session.rollback()
            raise
    
    def update_feedback(self, recommendation_id: int, is_accepted: bool) -> Optional[PartRecommendation]:
        """
        Update recommendation feedback.
        
        Args:
            recommendation_id: Recommendation identifier
            is_accepted: Whether recommendation was accepted
            
        Returns:
            Updated PartRecommendation object or None if not found
        """
        try:
            recommendation = self.session.query(PartRecommendation).filter_by(
                id=recommendation_id
            ).first()
            
            if not recommendation:
                return None
            
            recommendation.is_accepted = is_accepted
            recommendation.feedback_date = datetime.utcnow()
            self.session.commit()
            logger.info(f"Updated feedback for recommendation {recommendation_id}")
            return recommendation
        except Exception as e:
            logger.error(f"Error updating feedback for recommendation {recommendation_id}: {e}")
            self.session.rollback()
            raise


class UserFeedbackRepository(BaseRepository):
    """Repository for user feedback data access."""
    
    def create(self, feedback_data: Dict[str, Any]) -> UserFeedback:
        """
        Create a new user feedback record.
        
        Args:
            feedback_data: Dictionary containing feedback data
            
        Returns:
            Created UserFeedback object
        """
        try:
            feedback = UserFeedback(**feedback_data)
            self.session.add(feedback)
            self.session.commit()
            logger.info(f"Created feedback for recommendation {feedback.recommendation_id}")
            return feedback
        except Exception as e:
            logger.error(f"Error creating feedback: {e}")
            self.session.rollback()
            raise
    
    def get_by_recommendation(self, recommendation_id: int) -> List[UserFeedback]:
        """
        Get feedback for a specific recommendation.
        
        Args:
            recommendation_id: Recommendation identifier
            
        Returns:
            List of UserFeedback objects
        """
        try:
            return self.session.query(UserFeedback).filter_by(
                recommendation_id=recommendation_id
            ).order_by(desc(UserFeedback.created_at)).all()
        except Exception as e:
            logger.error(f"Error getting feedback for recommendation {recommendation_id}: {e}")
            raise
    
    def get_by_user(self, user_id: str, limit: int = 100) -> List[UserFeedback]:
        """
        Get feedback by user.
        
        Args:
            user_id: User identifier
            limit: Maximum number of records to return
            
        Returns:
            List of UserFeedback objects
        """
        try:
            return self.session.query(UserFeedback).filter_by(
                user_id=user_id
            ).order_by(desc(UserFeedback.created_at)).limit(limit).all()
        except Exception as e:
            logger.error(f"Error getting feedback for user {user_id}: {e}")
            raise


class PartMasterRepository(BaseRepository):
    """Repository for part master data access."""
    
    def get_by_code(self, part_code: str) -> Optional[PartMaster]:
        """
        Get part by code.
        
        Args:
            part_code: Part code
            
        Returns:
            PartMaster object or None if not found
        """
        try:
            return self.session.query(PartMaster).filter_by(part_code=part_code).first()
        except Exception as e:
            logger.error(f"Error getting part {part_code}: {e}")
            raise
    
    def get_by_category(self, category: str) -> List[PartMaster]:
        """
        Get parts by category.
        
        Args:
            category: Part category
            
        Returns:
            List of PartMaster objects
        """
        try:
            return self.session.query(PartMaster).filter_by(part_category=category).all()
        except Exception as e:
            logger.error(f"Error getting parts for category {category}: {e}")
            raise
    
    def get_critical_parts(self) -> List[PartMaster]:
        """
        Get critical parts.
        
        Returns:
            List of critical PartMaster objects
        """
        try:
            return self.session.query(PartMaster).filter_by(is_critical=True).all()
        except Exception as e:
            logger.error(f"Error getting critical parts: {e}")
            raise
    
    def get_seasonal_parts(self) -> List[PartMaster]:
        """
        Get seasonal parts.
        
        Returns:
            List of seasonal PartMaster objects
        """
        try:
            return self.session.query(PartMaster).filter_by(is_seasonal=True).all()
        except Exception as e:
            logger.error(f"Error getting seasonal parts: {e}")
            raise


class ModelPredictionRepository(BaseRepository):
    """Repository for model predictions (production monitoring)."""
    
    def create(self, data: Dict[str, Any]) -> ModelPrediction:
        try:
            pred = ModelPrediction(**data)
            self.session.add(pred)
            self.session.commit()
            logger.info(f"Created model prediction for vehicle {pred.vehicle_id} part {pred.part_code}")
            return pred
        except Exception as e:
            logger.error(f"Error creating model prediction: {e}")
            self.session.rollback()
            raise
    
    def create_batch(self, records: List[Dict[str, Any]]) -> List[ModelPrediction]:
        try:
            preds = []
            for d in records:
                p = ModelPrediction(**d)
                self.session.add(p)
                preds.append(p)
            self.session.commit()
            logger.info(f"Created {len(preds)} model predictions")
            return preds
        except Exception as e:
            logger.error(f"Error creating batch model predictions: {e}")
            self.session.rollback()
            raise
    
    def update_feedback(self, vehicle_id: str, part_code: str, accepted: bool, actual_replaced: bool) -> int:
        try:
            q = self.session.query(ModelPrediction).filter_by(vehicle_id=vehicle_id, part_code=part_code).order_by(desc(ModelPrediction.prediction_date))
            pred = q.first()
            if not pred:
                return 0
            pred.is_accepted = accepted
            pred.actual_replaced = actual_replaced
            pred.feedback_date = datetime.utcnow()
            self.session.commit()
            logger.info(f"Updated feedback for prediction {pred.id}")
            return 1
        except Exception as e:
            logger.error(f"Error updating prediction feedback: {e}")
            self.session.rollback()
            raise
