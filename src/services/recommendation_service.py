"""
Recommendation service for generating parts recommendations.

This service coordinates between the ML model, business rules engine,
and data access layer to generate intelligent parts recommendations.
"""

import time
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd
import redis

from src.models.lightgbm_model import LightGBMPartModel
from src.models.ema_calculator import EMACalculator
from src.models.seasonal_adjustments import SeasonalAdjustmentEngine
from src.data.repositories import (
    VehicleRepository,
    ServiceHistoryRepository,
    RecommendationRepository
)
from src.data.models import Vehicle, PartRecommendation
from src.api.exceptions import VehicleNotFoundError, PredictionError, DataQualityError

logger = logging.getLogger(__name__)


class RecommendationService:
    """
    Service for generating parts recommendations for vehicles.
    
    This service coordinates between the ML model, business rules engine,
    and data access layer to generate intelligent parts recommendations.
    
    Attributes:
        model: LightGBM model for predictions
        ema_calculator: EMA calculator for usage patterns
        seasonal_engine: Seasonal adjustment engine
        vehicle_repository: Vehicle data repository
        service_repository: Service history repository
        recommendation_repository: Part recommendation repository
        redis_client: Redis client for caching
        config: Service configuration
    """
    
    def __init__(
        self,
        model: LightGBMPartModel,
        vehicle_repository: VehicleRepository,
        service_repository: ServiceHistoryRepository,
        recommendation_repository: RecommendationRepository,
        redis_client: redis.Redis,
        config: Dict[str, Any]
    ):
        """
        Initialize the recommendation service.
        
        Args:
            model: LightGBM model for predictions
            vehicle_repository: Vehicle data repository
            service_repository: Service history repository
            recommendation_repository: Part recommendation repository
            redis_client: Redis client for caching
            config: Service configuration dictionary
        """
        self.model = model
        self.ema_calculator = EMACalculator()
        self.seasonal_engine = SeasonalAdjustmentEngine()
        self.vehicle_repository = vehicle_repository
        self.service_repository = service_repository
        self.recommendation_repository = recommendation_repository
        self.redis_client = redis_client
        self.config = config
        
        logger.info("Recommendation service initialized successfully")
    
    async def generate_recommendations(
        self,
        vehicle_id: str,
        odometer: float,
        complaints: Optional[str] = None,
        dealer_code: str = None
    ) -> Dict[str, Any]:
        """
        Generate parts recommendations for a vehicle.
        
        This function coordinates the entire recommendation pipeline:
        1. Fetch vehicle features from database
        2. Generate base predictions using LightGBM model
        3. Apply seasonal and terrain adjustments
        4. Filter and sort recommendations
        5. Check rejection history
        
        Args:
            vehicle_id: Unique identifier for the vehicle
            odometer: Current odometer reading in kilometers
            complaints: Optional customer complaint text
            dealer_code: Dealer identifier for location-based adjustments
            
        Returns:
            Dictionary containing recommendations and metadata
            
        Raises:
            VehicleNotFoundError: If vehicle doesn't exist
            PredictionError: If model prediction fails
            DataQualityError: If vehicle data is invalid
        """
        start_time = time.time()
        
        try:
            logger.info(f"Generating recommendations for vehicle {vehicle_id}")
            
            # Check cache first
            cache_key = f"recommendations:{vehicle_id}:{odometer}"
            cached_result = await self._get_from_cache(cache_key)
            if cached_result:
                logger.info(f"Returning cached recommendations for vehicle {vehicle_id}")
                return cached_result
            
            # Fetch vehicle data
            vehicle = await self._get_vehicle_data(vehicle_id)
            if not vehicle:
                raise VehicleNotFoundError(vehicle_id)
            
            # Update odometer if provided
            if odometer != vehicle.current_odometer:
                await self._update_vehicle_odometer(vehicle_id, odometer)
                vehicle.current_odometer = odometer
            
            # Get service history for EMA calculation
            service_history = await self._get_service_history(vehicle_id)
            
            # Calculate EMA if needed
            ema_value, ema_category = await self._calculate_ema(vehicle_id, service_history)
            
            # Prepare features for prediction
            features = await self._prepare_features(
                vehicle=vehicle,
                odometer=odometer,
                complaints=complaints,
                ema_value=ema_value,
                ema_category=ema_category
            )
            
            # Generate base predictions
            predictions = await self._generate_predictions(features)
            
            # Apply business rules and adjustments
            adjusted_predictions = await self._apply_adjustments(
                predictions=predictions,
                vehicle=vehicle,
                dealer_code=dealer_code
            )
            
            # Filter and sort recommendations
            recommendations = await self._filter_and_sort_recommendations(
                predictions=adjusted_predictions,
                vehicle_id=vehicle_id,
                confidence_threshold=self.config.get("confidence_threshold", 80.0)
            )
            
            # Build response
            result = {
                "vehicle_info": {
                    "vehicle_id": vehicle.vehicle_id,
                    "vehicle_model": vehicle.vehicle_model,
                    "current_odometer": vehicle.current_odometer,
                    "dealer_code": vehicle.dealer_code,
                    "region_code": vehicle.region_code,
                    "terrain_type": vehicle.terrain_type,
                    "season_code": vehicle.season_code,
                    "ema_value": ema_value,
                    "ema_category": ema_category,
                    "last_service_date": service_history[0].service_date if service_history else None
                },
                "recommendations": recommendations,
                "total_estimated_cost": sum(rec.get("estimated_cost", 0) for rec in recommendations),
                "model_version": self.config.get("model_version", "1.0.0"),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Cache the result
            await self._cache_result(cache_key, result)
            
            # Store recommendation in database
            await self._store_recommendations(vehicle_id, recommendations)
            
            processing_time = time.time() - start_time
            logger.info(
                f"Generated {len(recommendations)} recommendations for vehicle {vehicle_id} "
                f"in {processing_time:.2f} seconds"
            )
            
            return result
            
        except VehicleNotFoundError:
            raise
        except PredictionError:
            raise
        except DataQualityError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error generating recommendations: {e}", exc_info=True)
            raise PredictionError(f"Failed to generate recommendations: {e}")
    
    async def get_recommendation_history(
        self,
        vehicle_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get historical recommendations for a vehicle.
        
        Args:
            vehicle_id: Vehicle identifier
            limit: Maximum number of recommendations to return
            
        Returns:
            List of historical recommendations
        """
        try:
            logger.info(f"Fetching recommendation history for vehicle {vehicle_id}")
            
            recommendations = await self.recommendation_repository.get_by_vehicle_id(
                vehicle_id=vehicle_id,
                limit=limit
            )
            
            return [rec.to_dict() for rec in recommendations]
            
        except Exception as e:
            logger.error(f"Error fetching recommendation history: {e}")
            raise
    
    async def _get_vehicle_data(self, vehicle_id: str) -> Optional[Vehicle]:
        """Get vehicle data from database."""
        try:
            return await self.vehicle_repository.get_by_id(vehicle_id)
        except Exception as e:
            logger.error(f"Error fetching vehicle data: {e}")
            raise DataQualityError(f"Failed to fetch vehicle data: {e}")
    
    async def _get_service_history(self, vehicle_id: str) -> List:
        """Get service history for vehicle."""
        try:
            return await self.service_repository.get_by_vehicle_id(vehicle_id)
        except Exception as e:
            logger.warning(f"Error fetching service history: {e}")
            return []
    
    async def _calculate_ema(self, vehicle_id: str, service_history: List) -> tuple:
        """Calculate EMA for vehicle usage pattern."""
        try:
            if len(service_history) < 2:
                return None, None
            
            # Convert service history to DataFrame
            df = pd.DataFrame([{
                'service_date': sh.service_date,
                'odometer_reading': sh.odometer_reading
            } for sh in service_history])
            
            ema_value, method = self.ema_calculator.calculate_ema(df)
            ema_category = self.ema_calculator.categorize_ema(ema_value)
            
            # Update vehicle with EMA data
            await self.vehicle_repository.update_ema(vehicle_id, ema_value, ema_category)
            
            return ema_value, ema_category
            
        except Exception as e:
            logger.warning(f"Error calculating EMA: {e}")
            return None, None
    
    async def _prepare_features(
        self,
        vehicle: Vehicle,
        odometer: float,
        complaints: Optional[str],
        ema_value: Optional[float],
        ema_category: Optional[str]
    ) -> pd.DataFrame:
        """Prepare features for ML model prediction."""
        try:
            features = {
                'vehicle_model': vehicle.vehicle_model,
                'current_odometer': odometer,
                'dealer_code': vehicle.dealer_code,
                'region_code': vehicle.region_code,
                'terrain_type': vehicle.terrain_type,
                'season_code': vehicle.season_code,
                'ema_value': ema_value or 0,
                'ema_category': ema_category or 'UNKNOWN',
                'has_complaints': 1 if complaints else 0,
                'complaint_length': len(complaints) if complaints else 0
            }
            
            return pd.DataFrame([features])
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            raise DataQualityError(f"Failed to prepare features: {e}")
    
    async def _generate_predictions(self, features: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate predictions using ML model."""
        try:
            # Get predictions for all parts
            predictions = await self.model.predict_batch(features)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            raise PredictionError(f"Model prediction failed: {e}")
    
    async def _apply_adjustments(
        self,
        predictions: List[Dict[str, Any]],
        vehicle: Vehicle,
        dealer_code: str
    ) -> List[Dict[str, Any]]:
        """Apply seasonal and terrain adjustments to predictions."""
        try:
            adjusted_predictions = []
            
            for prediction in predictions:
                # Apply seasonal adjustment
                seasonal_adj = self.seasonal_engine.get_seasonal_adjustment(
                    part_code=prediction['part_code'],
                    season_code=vehicle.season_code,
                    terrain_type=vehicle.terrain_type
                )
                
                # Apply terrain adjustment
                terrain_adj = self.seasonal_engine.get_terrain_adjustment(
                    part_code=prediction['part_code'],
                    terrain_type=vehicle.terrain_type
                )
                
                # Calculate final confidence
                base_confidence = prediction['confidence_score']
                final_confidence = min(
                    base_confidence + seasonal_adj + terrain_adj,
                    99.0
                )
                
                prediction.update({
                    'seasonal_adjustment': seasonal_adj,
                    'terrain_adjustment': terrain_adj,
                    'final_confidence': final_confidence
                })
                
                adjusted_predictions.append(prediction)
            
            return adjusted_predictions
            
        except Exception as e:
            logger.error(f"Error applying adjustments: {e}")
            return predictions  # Return original predictions if adjustment fails
    
    async def _filter_and_sort_recommendations(
        self,
        predictions: List[Dict[str, Any]],
        vehicle_id: str,
        confidence_threshold: float
    ) -> List[Dict[str, Any]]:
        """Filter and sort recommendations by confidence score."""
        try:
            # Filter by confidence threshold
            filtered = [
                p for p in predictions 
                if p['final_confidence'] >= confidence_threshold
            ]
            
            # Sort by confidence score (descending)
            sorted_predictions = sorted(
                filtered,
                key=lambda x: x['final_confidence'],
                reverse=True
            )
            
            # Limit to top 10
            top_recommendations = sorted_predictions[:10]
            
            # Add ranking
            for i, rec in enumerate(top_recommendations, 1):
                rec['rank'] = i
            
            return top_recommendations
            
        except Exception as e:
            logger.error(f"Error filtering recommendations: {e}")
            return []
    
    async def _update_vehicle_odometer(self, vehicle_id: str, odometer: float) -> None:
        """Update vehicle odometer reading."""
        try:
            await self.vehicle_repository.update_odometer(vehicle_id, odometer)
        except Exception as e:
            logger.warning(f"Error updating odometer: {e}")
    
    async def _store_recommendations(
        self,
        vehicle_id: str,
        recommendations: List[Dict[str, Any]]
    ) -> None:
        """Store recommendations in database."""
        try:
            for rec in recommendations:
                recommendation = PartRecommendation(
                    vehicle_id=vehicle_id,
                    part_code=rec['part_code'],
                    part_name=rec['part_name'],
                    part_category=rec['category'],
                    confidence_score=rec['final_confidence'],
                    rank=rec['rank'],
                    estimated_cost=rec.get('estimated_cost', 0),
                    reasoning=rec.get('reasoning', {}),
                    model_version=self.config.get("model_version", "1.0.0"),
                    seasonal_adjustment=rec.get('seasonal_adjustment'),
                    terrain_adjustment=rec.get('terrain_adjustment'),
                    final_confidence=rec['final_confidence']
                )
                
                await self.recommendation_repository.create(recommendation)
                
        except Exception as e:
            logger.warning(f"Error storing recommendations: {e}")
    
    async def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get result from cache."""
        try:
            if self.redis_client:
                cached = self.redis_client.get(cache_key)
                if cached:
                    import json
                    return json.loads(cached)
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
        return None
    
    async def _cache_result(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Cache result."""
        try:
            if self.redis_client:
                import json
                ttl = self.config.get("cache_ttl", 1800)  # 30 minutes default
                self.redis_client.setex(
                    cache_key,
                    ttl,
                    json.dumps(result, default=str)
                )
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
