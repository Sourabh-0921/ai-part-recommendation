"""
Seasonal and terrain-based adjustments for parts recommendations.

This module implements business rules for adjusting confidence scores
based on seasonal conditions and terrain types as specified in the
.cursorrules file.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, date
import logging
from enum import Enum

from ..utils import (
    get_season,
    SEASON_SUMMER,
    SEASON_MONSOON,
    SEASON_SPRING,
    SEASON_WINTER,
    SEASON_MONTHS
)

logger = logging.getLogger(__name__)


class Season(Enum):
    """Season enumeration."""
    SPRING = "SPRING"
    SUMMER = "SUMMER"
    MONSOON = "MONSOON"
    WINTER = "WINTER"


class TerrainType(Enum):
    """Terrain type enumeration."""
    URBAN = "URBAN"
    HIGHWAY = "HIGHWAY"
    RURAL = "RURAL"
    HILLY = "HILLY"
    COASTAL = "COASTAL"


class SeasonalAdjustmentEngine:
    """
    Engine for applying seasonal and terrain-based adjustments to recommendations.
    
    This class implements business rules for adjusting confidence scores
    based on environmental factors that affect vehicle wear patterns.
    """
    
    def __init__(self):
        """Initialize the seasonal adjustment engine."""
        self.seasonal_rules = self._initialize_seasonal_rules()
        self.terrain_rules = self._initialize_terrain_rules()
        self.part_seasonal_factors = self._initialize_part_seasonal_factors()
        
        logger.info("Initialized SeasonalAdjustmentEngine")
    
    def _initialize_seasonal_rules(self) -> Dict[str, Dict[str, float]]:
        """
        Initialize seasonal adjustment rules.
        
        Returns:
            Dictionary of seasonal adjustment factors
        """
        return {
            "SPRING": {
                "base_adjustment": 0.0,
                "brake_adjustment": 0.05,  # More dust, higher brake wear
                "air_filter_adjustment": 0.10,  # Pollen season
                "tire_adjustment": 0.0,
                "battery_adjustment": 0.0
            },
            "SUMMER": {
                "base_adjustment": 0.0,
                "brake_adjustment": 0.0,
                "air_filter_adjustment": 0.0,
                "tire_adjustment": 0.15,  # Hot roads increase tire wear
                "battery_adjustment": 0.20,  # Heat affects battery life
                "cooling_system_adjustment": 0.25  # AC usage increases
            },
            "MONSOON": {
                "base_adjustment": 0.0,
                "brake_adjustment": 0.20,  # Wet conditions increase brake wear
                "air_filter_adjustment": 0.0,
                "tire_adjustment": 0.10,  # Wet roads affect tire wear
                "battery_adjustment": 0.0,
                "electrical_adjustment": 0.15  # Moisture affects electrical components
            },
            "WINTER": {
                "base_adjustment": 0.0,
                "brake_adjustment": 0.0,
                "air_filter_adjustment": 0.0,
                "tire_adjustment": 0.0,
                "battery_adjustment": 0.30,  # Cold weather affects battery
                "heating_system_adjustment": 0.20  # Heating system usage
            }
        }
    
    def _initialize_terrain_rules(self) -> Dict[str, Dict[str, float]]:
        """
        Initialize terrain-based adjustment rules.
        
        Returns:
            Dictionary of terrain adjustment factors
        """
        return {
            "URBAN": {
                "base_adjustment": 0.0,
                "brake_adjustment": 0.15,  # Frequent stops
                "clutch_adjustment": 0.20,  # Stop-and-go traffic
                "tire_adjustment": 0.05,  # City driving
                "suspension_adjustment": 0.10  # Potholes and rough roads
            },
            "HIGHWAY": {
                "base_adjustment": 0.0,
                "brake_adjustment": -0.05,  # Less frequent braking
                "clutch_adjustment": -0.10,  # Less clutch usage
                "tire_adjustment": 0.20,  # High-speed driving
                "engine_adjustment": 0.15,  # Sustained high RPM
                "transmission_adjustment": 0.10  # Sustained driving
            },
            "RURAL": {
                "base_adjustment": 0.0,
                "brake_adjustment": 0.0,
                "clutch_adjustment": 0.0,
                "tire_adjustment": 0.10,  # Rough roads
                "suspension_adjustment": 0.15,  # Uneven terrain
                "air_filter_adjustment": 0.20  # Dust and dirt
            },
            "HILLY": {
                "base_adjustment": 0.0,
                "brake_adjustment": 0.25,  # Downhill braking
                "clutch_adjustment": 0.30,  # Hill starts
                "tire_adjustment": 0.15,  # Steep inclines
                "engine_adjustment": 0.20,  # Uphill strain
                "transmission_adjustment": 0.15  # Gear changes
            },
            "COASTAL": {
                "base_adjustment": 0.0,
                "brake_adjustment": 0.0,
                "clutch_adjustment": 0.0,
                "tire_adjustment": 0.0,
                "battery_adjustment": 0.0,
                "electrical_adjustment": 0.25,  # Salt air corrosion
                "exhaust_adjustment": 0.20,  # Salt air corrosion
                "body_parts_adjustment": 0.30  # Rust and corrosion
            }
        }
    
    def _initialize_part_seasonal_factors(self) -> Dict[str, List[str]]:
        """
        Initialize part categories and their seasonal sensitivity.
        
        Returns:
            Dictionary mapping part categories to seasonal factors
        """
        return {
            "BRAKES": ["brake_adjustment"],
            "TIRES": ["tire_adjustment"],
            "BATTERY": ["battery_adjustment"],
            "AIR_FILTER": ["air_filter_adjustment"],
            "COOLING_SYSTEM": ["cooling_system_adjustment"],
            "HEATING_SYSTEM": ["heating_system_adjustment"],
            "ELECTRICAL": ["electrical_adjustment"],
            "ENGINE": ["engine_adjustment"],
            "TRANSMISSION": ["transmission_adjustment"],
            "CLUTCH": ["clutch_adjustment"],
            "SUSPENSION": ["suspension_adjustment"],
            "EXHAUST": ["exhaust_adjustment"],
            "BODY_PARTS": ["body_parts_adjustment"]
        }
    
    def get_current_season(self, date_obj: Optional[date] = None) -> Season:
        """
        Determine current season based on date.
        
        Uses utility function and constants for consistency.
        
        Args:
            date_obj: Date to determine season for (defaults to today)
            
        Returns:
            Current season
        """
        if date_obj is None:
            date_obj = date.today()
        
        # Use datetime_utils.get_season() which handles datetime objects
        dt_obj = datetime.combine(date_obj, datetime.min.time())
        season_str = get_season(dt_obj)
        
        # Map string to Season enum
        season_map = {
            SEASON_SPRING: Season.SPRING,
            SEASON_SUMMER: Season.SUMMER,
            SEASON_MONSOON: Season.MONSOON,
            SEASON_WINTER: Season.WINTER
        }
        
        return season_map.get(season_str, Season.SPRING)
    
    def calculate_seasonal_adjustment(
        self,
        part_category: str,
        season: Season,
        base_confidence: float
    ) -> float:
        """
        Calculate seasonal adjustment for a part.
        
        Args:
            part_category: Category of the part
            season: Current season
            base_confidence: Base confidence score
            
        Returns:
            Adjusted confidence score
        """
        try:
            seasonal_factors = self.seasonal_rules.get(season.value, {})
            part_factors = self.part_seasonal_factors.get(part_category, [])
            
            total_adjustment = 0.0
            
            for factor in part_factors:
                adjustment = seasonal_factors.get(factor, 0.0)
                total_adjustment += adjustment
            
            # Apply adjustment
            adjusted_confidence = base_confidence + (base_confidence * total_adjustment)
            
            # Ensure confidence stays within bounds (0-100)
            adjusted_confidence = max(0.0, min(100.0, adjusted_confidence))
            
            logger.debug(f"Seasonal adjustment for {part_category} in {season.value}: {total_adjustment:.2%}")
            
            return adjusted_confidence
            
        except Exception as e:
            logger.error(f"Error calculating seasonal adjustment: {e}")
            return base_confidence
    
    def calculate_terrain_adjustment(
        self,
        part_category: str,
        terrain_type: TerrainType,
        base_confidence: float
    ) -> float:
        """
        Calculate terrain-based adjustment for a part.
        
        Args:
            part_category: Category of the part
            terrain_type: Type of terrain
            base_confidence: Base confidence score
            
        Returns:
            Adjusted confidence score
        """
        try:
            terrain_factors = self.terrain_rules.get(terrain_type.value, {})
            part_factors = self.part_seasonal_factors.get(part_category, [])
            
            total_adjustment = 0.0
            
            for factor in part_factors:
                adjustment = terrain_factors.get(factor, 0.0)
                total_adjustment += adjustment
            
            # Apply adjustment
            adjusted_confidence = base_confidence + (base_confidence * total_adjustment)
            
            # Ensure confidence stays within bounds (0-100)
            adjusted_confidence = max(0.0, min(100.0, adjusted_confidence))
            
            logger.debug(f"Terrain adjustment for {part_category} in {terrain_type.value}: {total_adjustment:.2%}")
            
            return adjusted_confidence
            
        except Exception as e:
            logger.error(f"Error calculating terrain adjustment: {e}")
            return base_confidence
    
    def calculate_combined_adjustment(
        self,
        part_category: str,
        season: Season,
        terrain_type: TerrainType,
        base_confidence: float
    ) -> Dict[str, Any]:
        """
        Calculate combined seasonal and terrain adjustments.
        
        Args:
            part_category: Category of the part
            season: Current season
            terrain_type: Type of terrain
            base_confidence: Base confidence score
            
        Returns:
            Dictionary containing adjustment details
        """
        try:
            # Calculate individual adjustments
            seasonal_confidence = self.calculate_seasonal_adjustment(
                part_category, season, base_confidence
            )
            
            terrain_confidence = self.calculate_terrain_adjustment(
                part_category, terrain_type, base_confidence
            )
            
            # Calculate combined adjustment
            seasonal_adjustment = seasonal_confidence - base_confidence
            terrain_adjustment = terrain_confidence - base_confidence
            
            # Apply both adjustments
            final_confidence = base_confidence + seasonal_adjustment + terrain_adjustment
            final_confidence = max(0.0, min(100.0, final_confidence))
            
            return {
                'base_confidence': base_confidence,
                'seasonal_adjustment': seasonal_adjustment,
                'terrain_adjustment': terrain_adjustment,
                'final_confidence': final_confidence,
                'season': season.value,
                'terrain_type': terrain_type.value,
                'part_category': part_category,
                'total_adjustment_percentage': ((final_confidence - base_confidence) / base_confidence * 100) if base_confidence > 0 else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error calculating combined adjustment: {e}")
            return {
                'base_confidence': base_confidence,
                'seasonal_adjustment': 0.0,
                'terrain_adjustment': 0.0,
                'final_confidence': base_confidence,
                'season': season.value,
                'terrain_type': terrain_type.value,
                'part_category': part_category,
                'error': str(e)
            }
    
    def get_adjustment_reasoning(
        self,
        part_category: str,
        season: Season,
        terrain_type: TerrainType
    ) -> Dict[str, Any]:
        """
        Get reasoning for adjustments applied to a part.
        
        Args:
            part_category: Category of the part
            season: Current season
            terrain_type: Type of terrain
            
        Returns:
            Dictionary containing adjustment reasoning
        """
        seasonal_factors = self.seasonal_rules.get(season.value, {})
        terrain_factors = self.terrain_rules.get(terrain_type.value, {})
        part_factors = self.part_seasonal_factors.get(part_category, [])
        
        reasoning = {
            'part_category': part_category,
            'season': season.value,
            'terrain_type': terrain_type.value,
            'seasonal_factors': {},
            'terrain_factors': {},
            'applied_factors': []
        }
        
        # Get seasonal factors
        for factor in part_factors:
            if factor in seasonal_factors:
                reasoning['seasonal_factors'][factor] = seasonal_factors[factor]
                reasoning['applied_factors'].append(f"seasonal_{factor}")
        
        # Get terrain factors
        for factor in part_factors:
            if factor in terrain_factors:
                reasoning['terrain_factors'][factor] = terrain_factors[factor]
                reasoning['applied_factors'].append(f"terrain_{factor}")
        
        return reasoning
    
    def update_seasonal_rules(self, new_rules: Dict[str, Dict[str, float]]) -> None:
        """
        Update seasonal adjustment rules.
        
        Args:
            new_rules: New seasonal rules dictionary
        """
        self.seasonal_rules.update(new_rules)
        logger.info("Updated seasonal adjustment rules")
    
    def update_terrain_rules(self, new_rules: Dict[str, Dict[str, float]]) -> None:
        """
        Update terrain adjustment rules.
        
        Args:
            new_rules: New terrain rules dictionary
        """
        self.terrain_rules.update(new_rules)
        logger.info("Updated terrain adjustment rules")
    
    def get_adjustment_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about adjustment rules.
        
        Returns:
            Dictionary containing adjustment statistics
        """
        return {
            'seasons_covered': len(self.seasonal_rules),
            'terrain_types_covered': len(self.terrain_rules),
            'part_categories_covered': len(self.part_seasonal_factors),
            'total_seasonal_factors': sum(len(factors) for factors in self.seasonal_rules.values()),
            'total_terrain_factors': sum(len(factors) for factors in self.terrain_rules.values()),
            'available_seasons': list(self.seasonal_rules.keys()),
            'available_terrain_types': list(self.terrain_rules.keys()),
            'available_part_categories': list(self.part_seasonal_factors.keys())
        }
