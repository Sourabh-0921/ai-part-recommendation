#!/usr/bin/env python3
"""
Database initialization script for AI Parts Recommendation System.

This script initializes the database with the required schema and sample data.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import List, Dict

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.database import init_database, get_db_session
from src.data.models import (
    Vehicle, ServiceHistory, PartMaster, DealerMaster, 
    SeasonalConfig, TerrainConfig, BusinessRule, ModelVersion
)
from src.config.settings import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_dealers() -> List[Dict]:
    """Create sample dealer data."""
    return [
        {
            'dealer_code': 'DLR_MUM_01',
            'dealer_name': 'Mumbai Central Service Center',
            'region_code': 'MUM',
            'city': 'Mumbai',
            'state': 'Maharashtra',
            'pincode': '400001',
            'contact_person': 'Rajesh Kumar',
            'phone': '+91-22-12345678',
            'email': 'mumbai@service.com',
            'is_active': True
        },
        {
            'dealer_code': 'DLR_DEL_01',
            'dealer_name': 'Delhi North Service Center',
            'region_code': 'DEL',
            'city': 'New Delhi',
            'state': 'Delhi',
            'pincode': '110001',
            'contact_person': 'Amit Sharma',
            'phone': '+91-11-87654321',
            'email': 'delhi@service.com',
            'is_active': True
        },
        {
            'dealer_code': 'DLR_BLR_01',
            'dealer_name': 'Bangalore Tech Service Center',
            'region_code': 'BLR',
            'city': 'Bangalore',
            'state': 'Karnataka',
            'pincode': '560001',
            'contact_person': 'Priya Reddy',
            'phone': '+91-80-98765432',
            'email': 'bangalore@service.com',
            'is_active': True
        }
    ]


def create_sample_parts() -> List[Dict]:
    """Create sample parts data."""
    return [
        {
            'part_code': 'BP001',
            'part_name': 'Brake Pads - Front',
            'part_category': 'Brakes',
            'oem_part_number': 'OEM-BP-F-001',
            'estimated_cost': 2500.0,
            'replacement_interval_km': 30000,
            'replacement_interval_months': 18,
            'is_critical': True,
            'is_seasonal': False,
            'terrain_specific': False
        },
        {
            'part_code': 'BP002',
            'part_name': 'Brake Pads - Rear',
            'part_category': 'Brakes',
            'oem_part_number': 'OEM-BP-R-001',
            'estimated_cost': 2000.0,
            'replacement_interval_km': 40000,
            'replacement_interval_months': 24,
            'is_critical': True,
            'is_seasonal': False,
            'terrain_specific': False
        },
        {
            'part_code': 'OF001',
            'part_name': 'Engine Oil Filter',
            'part_category': 'Engine',
            'oem_part_number': 'OEM-OF-001',
            'estimated_cost': 800.0,
            'replacement_interval_km': 10000,
            'replacement_interval_months': 6,
            'is_critical': True,
            'is_seasonal': False,
            'terrain_specific': False
        },
        {
            'part_code': 'AF001',
            'part_name': 'Air Filter',
            'part_category': 'Engine',
            'oem_part_number': 'OEM-AF-001',
            'estimated_cost': 1200.0,
            'replacement_interval_km': 15000,
            'replacement_interval_months': 12,
            'is_critical': False,
            'is_seasonal': True,
            'terrain_specific': True
        },
        {
            'part_code': 'SP001',
            'part_name': 'Spark Plugs - Set of 4',
            'part_category': 'Engine',
            'oem_part_number': 'OEM-SP-001',
            'estimated_cost': 1500.0,
            'replacement_interval_km': 20000,
            'replacement_interval_months': 18,
            'is_critical': False,
            'is_seasonal': False,
            'terrain_specific': False
        }
    ]


def create_sample_vehicles() -> List[Dict]:
    """Create sample vehicle data."""
    return [
        {
            'vehicle_id': 'MH12AB1234',
            'vehicle_model': 'Pulsar 150',
            'invoice_date': datetime(2022, 1, 15),
            'current_odometer': 25000.0,
            'dealer_code': 'DLR_MUM_01',
            'region_code': 'MUM',
            'terrain_type': 'URBAN',
            'season_code': 'MONSOON',
            'ema_value': 650.0,
            'ema_category': 'HIGH_USAGE'
        },
        {
            'vehicle_id': 'DL01CD5678',
            'vehicle_model': 'Pulsar 150',
            'invoice_date': datetime(2021, 8, 20),
            'current_odometer': 45000.0,
            'dealer_code': 'DLR_DEL_01',
            'region_code': 'DEL',
            'terrain_type': 'URBAN',
            'season_code': 'WINTER',
            'ema_value': 420.0,
            'ema_category': 'MEDIUM_USAGE'
        },
        {
            'vehicle_id': 'KA03EF9012',
            'vehicle_model': 'Pulsar 150',
            'invoice_date': datetime(2023, 3, 10),
            'current_odometer': 12000.0,
            'dealer_code': 'DLR_BLR_01',
            'region_code': 'BLR',
            'terrain_type': 'HILLY',
            'season_code': 'SUMMER',
            'ema_value': 280.0,
            'ema_category': 'LOW_USAGE'
        }
    ]


def create_seasonal_configs() -> List[Dict]:
    """Create seasonal configuration data."""
    return [
        {
            'season_code': 'SUMMER',
            'season_name': 'Summer Season',
            'start_month': 3,
            'end_month': 6,
            'adjustment_factor': 1.2,
            'affected_parts': ['AF001', 'OF001'],
            'region_codes': ['MUM', 'BLR', 'DEL'],
            'is_active': True
        },
        {
            'season_code': 'MONSOON',
            'season_name': 'Monsoon Season',
            'start_month': 7,
            'end_month': 9,
            'adjustment_factor': 1.5,
            'affected_parts': ['BP001', 'BP002', 'AF001'],
            'region_codes': ['MUM', 'BLR'],
            'is_active': True
        },
        {
            'season_code': 'WINTER',
            'season_name': 'Winter Season',
            'start_month': 10,
            'end_month': 2,
            'adjustment_factor': 1.1,
            'affected_parts': ['SP001', 'OF001'],
            'region_codes': ['DEL', 'BLR'],
            'is_active': True
        }
    ]


def create_terrain_configs() -> List[Dict]:
    """Create terrain configuration data."""
    return [
        {
            'terrain_type': 'URBAN',
            'terrain_name': 'Urban Roads',
            'adjustment_factor': 1.0,
            'affected_parts': None,
            'region_codes': None,
            'usage_multiplier': 1.0,
            'is_active': True
        },
        {
            'terrain_type': 'HILLY',
            'terrain_name': 'Hilly Terrain',
            'adjustment_factor': 1.3,
            'affected_parts': ['BP001', 'BP002', 'SP001'],
            'region_codes': ['BLR'],
            'usage_multiplier': 1.2,
            'is_active': True
        },
        {
            'terrain_type': 'HIGHWAY',
            'terrain_name': 'Highway Driving',
            'adjustment_factor': 0.8,
            'affected_parts': ['OF001', 'AF001'],
            'region_codes': None,
            'usage_multiplier': 1.5,
            'is_active': True
        }
    ]


def create_business_rules() -> List[Dict]:
    """Create business rules data."""
    return [
        {
            'rule_name': 'High Usage Brake Replacement',
            'rule_type': 'USAGE',
            'rule_condition': {
                'ema_category': 'HIGH_USAGE',
                'part_category': 'Brakes'
            },
            'rule_action': {
                'adjustment_factor': 1.2,
                'priority_boost': 10
            },
            'priority': 1,
            'is_active': True,
            'created_by': 'system'
        },
        {
            'rule_name': 'Monsoon Brake Priority',
            'rule_type': 'SEASONAL',
            'rule_condition': {
                'season_code': 'MONSOON',
                'part_category': 'Brakes'
            },
            'rule_action': {
                'adjustment_factor': 1.3,
                'priority_boost': 15
            },
            'priority': 2,
            'is_active': True,
            'created_by': 'system'
        }
    ]


def create_model_version() -> Dict:
    """Create initial model version data."""
    return {
        'model_name': 'parts_recommendation_model',
        'version': '1.0.0',
        'model_path': 'models/parts_recommendation_v1.0.0.pkl',
        'model_type': 'LIGHTGBM',
        'training_date': datetime.now(),
        'accuracy_score': 0.85,
        'precision_score': 0.82,
        'recall_score': 0.88,
        'f1_score': 0.85,
        'auc_score': 0.90,
        'is_active': True,
        'metadata': {
            'training_samples': 10000,
            'features_count': 25,
            'categorical_features': ['vehicle_model', 'dealer_code', 'region_code', 'terrain_type', 'season_code']
        }
    }


def insert_sample_data():
    """Insert sample data into the database."""
    logger.info("Starting sample data insertion...")
    
    with get_db_session() as session:
        try:
            # Insert dealers
            logger.info("Inserting dealer data...")
            dealers_data = create_sample_dealers()
            for dealer_data in dealers_data:
                dealer = DealerMaster(**dealer_data)
                session.add(dealer)
            
            # Insert parts
            logger.info("Inserting parts data...")
            parts_data = create_sample_parts()
            for part_data in parts_data:
                part = PartMaster(**part_data)
                session.add(part)
            
            # Insert vehicles
            logger.info("Inserting vehicle data...")
            vehicles_data = create_sample_vehicles()
            for vehicle_data in vehicles_data:
                vehicle = Vehicle(**vehicle_data)
                session.add(vehicle)
            
            # Insert seasonal configs
            logger.info("Inserting seasonal configuration data...")
            seasonal_data = create_seasonal_configs()
            for config_data in seasonal_data:
                config = SeasonalConfig(**config_data)
                session.add(config)
            
            # Insert terrain configs
            logger.info("Inserting terrain configuration data...")
            terrain_data = create_terrain_configs()
            for config_data in terrain_data:
                config = TerrainConfig(**config_data)
                session.add(config)
            
            # Insert business rules
            logger.info("Inserting business rules data...")
            rules_data = create_business_rules()
            for rule_data in rules_data:
                rule = BusinessRule(**rule_data)
                session.add(rule)
            
            # Insert model version
            logger.info("Inserting model version data...")
            model_data = create_model_version()
            model_version = ModelVersion(**model_data)
            session.add(model_version)
            
            session.commit()
            logger.info("Sample data insertion completed successfully!")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error inserting sample data: {e}")
            raise


def main():
    """Main function to initialize the database."""
    logger.info("Starting database initialization...")
    
    try:
        # Initialize database
        logger.info("Initializing database connection...")
        init_database()
        
        # Insert sample data
        insert_sample_data()
        
        logger.info("Database initialization completed successfully!")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
