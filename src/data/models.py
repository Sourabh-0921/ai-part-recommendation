"""
Database models for AI Parts Recommendation System.

This module defines all SQLAlchemy models for the application.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Boolean, Text, ForeignKey, Index, UniqueConstraint, CheckConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from datetime import datetime
from typing import Optional, List
import uuid

Base = declarative_base()


class Vehicle(Base):
    """Vehicle master data model."""
    
    __tablename__ = 'vehicle_master'
    
    vehicle_id = Column(String(50), primary_key=True, index=True)
    vehicle_model = Column(String(100), nullable=False, index=True)
    invoice_date = Column(DateTime, nullable=False)
    current_odometer = Column(Float, nullable=False)
    dealer_code = Column(String(50), nullable=False, index=True)
    region_code = Column(String(50), nullable=False, index=True)
    terrain_type = Column(String(50), nullable=False)
    season_code = Column(String(20), nullable=False)
    ema_value = Column(Float, nullable=True)  # Exponential Moving Average
    ema_category = Column(String(20), nullable=True)  # HIGH_USAGE, MEDIUM_USAGE, LOW_USAGE
    last_updated = Column(DateTime, default=datetime.utcnow, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Additional indexes for performance
    __table_args__ = (
        Index('idx_vehicle_model_region', 'vehicle_model', 'region_code'),
        Index('idx_vehicle_terrain_season', 'terrain_type', 'season_code'),
        Index('idx_vehicle_ema_category', 'ema_category'),
        Index('idx_vehicle_dealer_updated', 'dealer_code', 'last_updated'),
    )
    
    # Relationships
    service_history = relationship("ServiceHistory", back_populates="vehicle")
    recommendations = relationship("PartRecommendation", back_populates="vehicle")
    
    def __repr__(self):
        return f"<Vehicle(vehicle_id={self.vehicle_id}, model={self.vehicle_model})>"
    
    def to_dict(self) -> dict:
        """Convert model to dictionary."""
        return {
            'vehicle_id': self.vehicle_id,
            'vehicle_model': self.vehicle_model,
            'invoice_date': self.invoice_date.isoformat() if self.invoice_date else None,
            'current_odometer': self.current_odometer,
            'dealer_code': self.dealer_code,
            'region_code': self.region_code,
            'terrain_type': self.terrain_type,
            'season_code': self.season_code,
            'ema_value': self.ema_value,
            'ema_category': self.ema_category,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class ServiceHistory(Base):
    """Service history model."""
    
    __tablename__ = 'service_history'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    vehicle_id = Column(String(50), ForeignKey('vehicle_master.vehicle_id'), nullable=False, index=True)
    service_date = Column(DateTime, nullable=False, index=True)
    odometer_reading = Column(Float, nullable=False)
    service_type = Column(String(50), nullable=False)  # REGULAR, MAJOR, EMERGENCY
    parts_replaced = Column(JSON, nullable=True)  # List of part codes
    labor_cost = Column(Float, nullable=True)
    parts_cost = Column(Float, nullable=True)
    total_cost = Column(Float, nullable=True)
    dealer_code = Column(String(50), nullable=False, index=True)
    technician_notes = Column(Text, nullable=True)
    customer_complaints = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Additional indexes for performance
    __table_args__ = (
        Index('idx_service_vehicle_date', 'vehicle_id', 'service_date'),
        Index('idx_service_dealer_type', 'dealer_code', 'service_type'),
        Index('idx_service_odometer', 'odometer_reading'),
        Index('idx_service_cost', 'total_cost'),
    )
    
    # Relationships
    vehicle = relationship("Vehicle", back_populates="service_history")
    
    def __repr__(self):
        return f"<ServiceHistory(vehicle_id={self.vehicle_id}, date={self.service_date})>"
    
    def to_dict(self) -> dict:
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'vehicle_id': self.vehicle_id,
            'service_date': self.service_date.isoformat() if self.service_date else None,
            'odometer_reading': self.odometer_reading,
            'service_type': self.service_type,
            'parts_replaced': self.parts_replaced,
            'labor_cost': self.labor_cost,
            'parts_cost': self.parts_cost,
            'total_cost': self.total_cost,
            'dealer_code': self.dealer_code,
            'technician_notes': self.technician_notes,
            'customer_complaints': self.customer_complaints,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class PartRecommendation(Base):
    """Part recommendation model."""
    
    __tablename__ = 'part_recommendations'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    vehicle_id = Column(String(50), ForeignKey('vehicle_master.vehicle_id'), nullable=False, index=True)
    part_code = Column(String(50), nullable=False, index=True)
    part_name = Column(String(200), nullable=False)
    part_category = Column(String(50), nullable=False, index=True)
    confidence_score = Column(Float, nullable=False)
    rank = Column(Integer, nullable=False)
    estimated_cost = Column(Float, nullable=True)
    reasoning = Column(JSON, nullable=True)  # Explanation of recommendation
    model_version = Column(String(20), nullable=False)
    seasonal_adjustment = Column(Float, nullable=True)
    terrain_adjustment = Column(Float, nullable=True)
    final_confidence = Column(Float, nullable=False)
    is_accepted = Column(Boolean, nullable=True)  # User feedback
    feedback_date = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Additional indexes for performance
    __table_args__ = (
        Index('idx_recommendation_vehicle_rank', 'vehicle_id', 'rank'),
        Index('idx_recommendation_confidence', 'final_confidence'),
        Index('idx_recommendation_category_confidence', 'part_category', 'final_confidence'),
        Index('idx_recommendation_model_version', 'model_version'),
        Index('idx_recommendation_feedback', 'is_accepted', 'feedback_date'),
    )
    
    # Relationships
    vehicle = relationship("Vehicle", back_populates="recommendations")
    feedback = relationship("UserFeedback", back_populates="recommendation")
    
    def __repr__(self):
        return f"<PartRecommendation(vehicle_id={self.vehicle_id}, part={self.part_code}, confidence={self.confidence_score})>"
    
    def to_dict(self) -> dict:
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'vehicle_id': self.vehicle_id,
            'part_code': self.part_code,
            'part_name': self.part_name,
            'part_category': self.part_category,
            'confidence_score': self.confidence_score,
            'rank': self.rank,
            'estimated_cost': self.estimated_cost,
            'reasoning': self.reasoning,
            'model_version': self.model_version,
            'seasonal_adjustment': self.seasonal_adjustment,
            'terrain_adjustment': self.terrain_adjustment,
            'final_confidence': self.final_confidence,
            'is_accepted': self.is_accepted,
            'feedback_date': self.feedback_date.isoformat() if self.feedback_date else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class UserFeedback(Base):
    """User feedback model."""
    
    __tablename__ = 'user_feedback'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    recommendation_id = Column(Integer, ForeignKey('part_recommendations.id'), nullable=False, index=True)
    user_id = Column(String(50), nullable=False, index=True)
    feedback_type = Column(String(20), nullable=False)  # ACCEPTED, REJECTED, MODIFIED
    feedback_reason = Column(Text, nullable=True)
    alternative_part = Column(String(50), nullable=True)
    actual_cost = Column(Float, nullable=True)
    service_date = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    recommendation = relationship("PartRecommendation", back_populates="feedback")
    
    def __repr__(self):
        return f"<UserFeedback(recommendation_id={self.recommendation_id}, type={self.feedback_type})>"
    
    def to_dict(self) -> dict:
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'recommendation_id': self.recommendation_id,
            'user_id': self.user_id,
            'feedback_type': self.feedback_type,
            'feedback_reason': self.feedback_reason,
            'alternative_part': self.alternative_part,
            'actual_cost': self.actual_cost,
            'service_date': self.service_date.isoformat() if self.service_date else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class PartMaster(Base):
    """Part master data model."""
    
    __tablename__ = 'part_master'
    
    part_code = Column(String(50), primary_key=True)
    part_name = Column(String(200), nullable=False)
    part_category = Column(String(50), nullable=False, index=True)
    oem_part_number = Column(String(100), nullable=True)
    estimated_cost = Column(Float, nullable=True)
    replacement_interval_km = Column(Integer, nullable=True)
    replacement_interval_months = Column(Integer, nullable=True)
    is_critical = Column(Boolean, default=False, nullable=False)
    is_seasonal = Column(Boolean, default=False, nullable=False)
    terrain_specific = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f"<PartMaster(part_code={self.part_code}, name={self.part_name})>"
    
    def to_dict(self) -> dict:
        """Convert model to dictionary."""
        return {
            'part_code': self.part_code,
            'part_name': self.part_name,
            'part_category': self.part_category,
            'oem_part_number': self.oem_part_number,
            'estimated_cost': self.estimated_cost,
            'replacement_interval_km': self.replacement_interval_km,
            'replacement_interval_months': self.replacement_interval_months,
            'is_critical': self.is_critical,
            'is_seasonal': self.is_seasonal,
            'terrain_specific': self.terrain_specific,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class PartsInventory(Base):
    """Parts inventory model for tracking stock levels."""
    
    __tablename__ = 'parts_inventory'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    part_code = Column(String(50), ForeignKey('part_master.part_code'), nullable=False, index=True)
    dealer_code = Column(String(50), nullable=False, index=True)
    current_stock = Column(Integer, nullable=False, default=0)
    minimum_stock = Column(Integer, nullable=False, default=0)
    maximum_stock = Column(Integer, nullable=True)
    reorder_point = Column(Integer, nullable=True)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    part = relationship("PartMaster")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('part_code', 'dealer_code', name='uq_part_dealer'),
        CheckConstraint('current_stock >= 0', name='ck_current_stock_positive'),
        CheckConstraint('minimum_stock >= 0', name='ck_minimum_stock_positive'),
    )
    
    def __repr__(self):
        return f"<PartsInventory(part_code={self.part_code}, dealer={self.dealer_code}, stock={self.current_stock})>"
    
    def to_dict(self) -> dict:
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'part_code': self.part_code,
            'dealer_code': self.dealer_code,
            'current_stock': self.current_stock,
            'minimum_stock': self.minimum_stock,
            'maximum_stock': self.maximum_stock,
            'reorder_point': self.reorder_point,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class SeasonalConfig(Base):
    """Seasonal configuration model for business rules."""
    
    __tablename__ = 'seasonal_config'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    season_code = Column(String(20), nullable=False, index=True)
    season_name = Column(String(50), nullable=False)
    start_month = Column(Integer, nullable=False)
    end_month = Column(Integer, nullable=False)
    adjustment_factor = Column(Float, nullable=False, default=1.0)
    affected_parts = Column(ARRAY(String), nullable=True)  # List of part codes
    region_codes = Column(ARRAY(String), nullable=True)  # List of region codes
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Constraints
    __table_args__ = (
        CheckConstraint('start_month >= 1 AND start_month <= 12', name='ck_start_month_valid'),
        CheckConstraint('end_month >= 1 AND end_month <= 12', name='ck_end_month_valid'),
        CheckConstraint('adjustment_factor > 0', name='ck_adjustment_factor_positive'),
    )
    
    def __repr__(self):
        return f"<SeasonalConfig(season={self.season_code}, factor={self.adjustment_factor})>"
    
    def to_dict(self) -> dict:
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'season_code': self.season_code,
            'season_name': self.season_name,
            'start_month': self.start_month,
            'end_month': self.end_month,
            'adjustment_factor': self.adjustment_factor,
            'affected_parts': self.affected_parts,
            'region_codes': self.region_codes,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class TerrainConfig(Base):
    """Terrain configuration model for business rules."""
    
    __tablename__ = 'terrain_config'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    terrain_type = Column(String(50), nullable=False, index=True)
    terrain_name = Column(String(100), nullable=False)
    adjustment_factor = Column(Float, nullable=False, default=1.0)
    affected_parts = Column(ARRAY(String), nullable=True)  # List of part codes
    region_codes = Column(ARRAY(String), nullable=True)  # List of region codes
    usage_multiplier = Column(Float, nullable=False, default=1.0)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Constraints
    __table_args__ = (
        CheckConstraint('adjustment_factor > 0', name='ck_terrain_adjustment_positive'),
        CheckConstraint('usage_multiplier > 0', name='ck_usage_multiplier_positive'),
    )
    
    def __repr__(self):
        return f"<TerrainConfig(terrain={self.terrain_type}, factor={self.adjustment_factor})>"
    
    def to_dict(self) -> dict:
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'terrain_type': self.terrain_type,
            'terrain_name': self.terrain_name,
            'adjustment_factor': self.adjustment_factor,
            'affected_parts': self.affected_parts,
            'region_codes': self.region_codes,
            'usage_multiplier': self.usage_multiplier,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class ModelVersion(Base):
    """Model version tracking for ML models."""
    
    __tablename__ = 'model_versions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(100), nullable=False, index=True)
    version = Column(String(20), nullable=False, index=True)
    model_path = Column(String(500), nullable=False)
    model_type = Column(String(50), nullable=False)  # LIGHTGBM, ENSEMBLE, etc.
    training_date = Column(DateTime, nullable=False)
    accuracy_score = Column(Float, nullable=True)
    precision_score = Column(Float, nullable=True)
    recall_score = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    auc_score = Column(Float, nullable=True)
    is_active = Column(Boolean, default=False, nullable=False)
    # Use attribute name different from SQLAlchemy's reserved 'metadata'
    model_metadata = Column('metadata', JSON, nullable=True)  # Additional model metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('model_name', 'version', name='uq_model_name_version'),
        CheckConstraint('accuracy_score >= 0 AND accuracy_score <= 1', name='ck_accuracy_range'),
        CheckConstraint('precision_score >= 0 AND precision_score <= 1', name='ck_precision_range'),
        CheckConstraint('recall_score >= 0 AND recall_score <= 1', name='ck_recall_range'),
        CheckConstraint('f1_score >= 0 AND f1_score <= 1', name='ck_f1_range'),
        CheckConstraint('auc_score >= 0 AND auc_score <= 1', name='ck_auc_range'),
    )
    
    def __repr__(self):
        return f"<ModelVersion(name={self.model_name}, version={self.version}, active={self.is_active})>"
    
    def to_dict(self) -> dict:
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'model_name': self.model_name,
            'version': self.version,
            'model_path': self.model_path,
            'model_type': self.model_type,
            'training_date': self.training_date.isoformat() if self.training_date else None,
            'accuracy_score': self.accuracy_score,
            'precision_score': self.precision_score,
            'recall_score': self.recall_score,
            'f1_score': self.f1_score,
            'auc_score': self.auc_score,
            'is_active': self.is_active,
            'metadata': self.model_metadata,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class BusinessRule(Base):
    """Business rules configuration model."""
    
    __tablename__ = 'business_rules'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    rule_name = Column(String(100), nullable=False, index=True)
    rule_type = Column(String(50), nullable=False)  # SEASONAL, TERRAIN, USAGE, COST
    rule_condition = Column(JSON, nullable=False)  # Rule conditions
    rule_action = Column(JSON, nullable=False)  # Rule actions
    priority = Column(Integer, nullable=False, default=0)
    is_active = Column(Boolean, default=True, nullable=False)
    effective_from = Column(DateTime, nullable=True)
    effective_until = Column(DateTime, nullable=True)
    created_by = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Constraints
    __table_args__ = (
        CheckConstraint('priority >= 0', name='ck_priority_positive'),
    )
    
    def __repr__(self):
        return f"<BusinessRule(name={self.rule_name}, type={self.rule_type}, active={self.is_active})>"
    
    def to_dict(self) -> dict:
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'rule_name': self.rule_name,
            'rule_type': self.rule_type,
            'rule_condition': self.rule_condition,
            'rule_action': self.rule_action,
            'priority': self.priority,
            'is_active': self.is_active,
            'effective_from': self.effective_from.isoformat() if self.effective_from else None,
            'effective_until': self.effective_until.isoformat() if self.effective_until else None,
            'created_by': self.created_by,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class DealerMaster(Base):
    """Dealer master data model."""
    
    __tablename__ = 'dealer_master'
    
    dealer_code = Column(String(50), primary_key=True)
    dealer_name = Column(String(200), nullable=False)
    region_code = Column(String(50), nullable=False, index=True)
    city = Column(String(100), nullable=False)
    state = Column(String(100), nullable=False)
    pincode = Column(String(10), nullable=True)
    contact_person = Column(String(100), nullable=True)
    phone = Column(String(20), nullable=True)
    email = Column(String(100), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f"<DealerMaster(code={self.dealer_code}, name={self.dealer_name})>"
    
    def to_dict(self) -> dict:
        """Convert model to dictionary."""
        return {
            'dealer_code': self.dealer_code,
            'dealer_name': self.dealer_name,
            'region_code': self.region_code,
            'city': self.city,
            'state': self.state,
            'pincode': self.pincode,
            'contact_person': self.contact_person,
            'phone': self.phone,
            'email': self.email,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class PredictionCache(Base):
    """Cache for model predictions to improve performance."""
    
    __tablename__ = 'prediction_cache'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    cache_key = Column(String(200), nullable=False, unique=True, index=True)
    vehicle_id = Column(String(50), nullable=False, index=True)
    model_version = Column(String(20), nullable=False)
    predictions = Column(JSON, nullable=False)
    confidence_scores = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=False, index=True)
    
    def __repr__(self):
        return f"<PredictionCache(vehicle_id={self.vehicle_id}, expires={self.expires_at})>"
    
    def to_dict(self) -> dict:
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'cache_key': self.cache_key,
            'vehicle_id': self.vehicle_id,
            'model_version': self.model_version,
            'predictions': self.predictions,
            'confidence_scores': self.confidence_scores,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None
        }


class ModelPrediction(Base):
    """Model prediction tracking for validation and monitoring."""
    
    __tablename__ = 'model_predictions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    vehicle_id = Column(String(50), ForeignKey('vehicle_master.vehicle_id'), nullable=False, index=True)
    job_card_number = Column(String(50), nullable=True, index=True)
    part_code = Column(String(50), nullable=False, index=True)
    confidence_score = Column(Float, nullable=False)
    model_version = Column(String(20), nullable=False, index=True)
    dealer_code = Column(String(50), nullable=False, index=True)
    ab_test_group = Column(String(20), nullable=True)  # 'control' or 'treatment' for A/B testing
    prediction_date = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    is_accepted = Column(Boolean, nullable=True)  # User feedback
    actual_replaced = Column(Boolean, nullable=True)  # Actual outcome
    feedback_date = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Additional indexes
    __table_args__ = (
        Index('idx_prediction_vehicle_part', 'vehicle_id', 'part_code'),
        Index('idx_prediction_model_date', 'model_version', 'prediction_date'),
        Index('idx_prediction_feedback', 'is_accepted', 'actual_replaced', 'feedback_date'),
    )
    
    def __repr__(self):
        return f"<ModelPrediction(vehicle_id={self.vehicle_id}, part={self.part_code}, confidence={self.confidence_score})>"
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'vehicle_id': self.vehicle_id,
            'job_card_number': self.job_card_number,
            'part_code': self.part_code,
            'confidence_score': self.confidence_score,
            'model_version': self.model_version,
            'dealer_code': self.dealer_code,
            'ab_test_group': self.ab_test_group,
            'prediction_date': self.prediction_date.isoformat() if self.prediction_date else None,
            'is_accepted': self.is_accepted,
            'actual_replaced': self.actual_replaced,
            'feedback_date': self.feedback_date.isoformat() if self.feedback_date else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class PilotFeedback(Base):
    """Pilot testing feedback tracking."""
    
    __tablename__ = 'pilot_feedback'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    job_card_number = Column(String(50), nullable=False, index=True)
    vehicle_id = Column(String(50), ForeignKey('vehicle_master.vehicle_id'), nullable=False, index=True)
    service_date = Column(DateTime, nullable=False, index=True)
    part_code = Column(String(50), nullable=False, index=True)
    confidence_score = Column(Float, nullable=False)
    action_taken = Column(String(20), nullable=False)  # ACCEPTED, REJECTED, IGNORED
    rejection_reason = Column(Text, nullable=True)
    service_advisor_id = Column(String(50), nullable=False)
    actual_replaced = Column(Boolean, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Additional indexes
    __table_args__ = (
        Index('idx_pilot_job_card', 'job_card_number', 'part_code'),
        Index('idx_pilot_service_date', 'service_date', 'action_taken'),
    )
    
    def __repr__(self):
        return f"<PilotFeedback(job_card={self.job_card_number}, part={self.part_code}, action={self.action_taken})>"
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'job_card_number': self.job_card_number,
            'vehicle_id': self.vehicle_id,
            'service_date': self.service_date.isoformat() if self.service_date else None,
            'part_code': self.part_code,
            'confidence_score': self.confidence_score,
            'action_taken': self.action_taken,
            'rejection_reason': self.rejection_reason,
            'service_advisor_id': self.service_advisor_id,
            'actual_replaced': self.actual_replaced,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class ModelValidation(Base):
    """Model validation results tracking."""
    
    __tablename__ = 'model_validations'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(100), nullable=False, index=True)
    model_version = Column(String(20), nullable=False, index=True)
    validation_type = Column(String(50), nullable=False)  # HISTORICAL, PILOT, PRODUCTION, AB_TEST
    validation_date = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    precision_score = Column(Float, nullable=True)
    recall_score = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    sample_size = Column(Integer, nullable=False)
    baseline_precision = Column(Float, nullable=True)
    baseline_recall = Column(Float, nullable=True)
    baseline_f1 = Column(Float, nullable=True)
    metrics = Column(JSON, nullable=True)  # Full metrics dictionary
    report_path = Column(String(500), nullable=True)  # Path to validation report
    validation_status = Column(String(20), nullable=False)  # PASS, WARNING, FAIL
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Constraints
    __table_args__ = (
        CheckConstraint('precision_score >= 0 AND precision_score <= 1', name='ck_val_precision_range'),
        CheckConstraint('recall_score >= 0 AND recall_score <= 1', name='ck_val_recall_range'),
        CheckConstraint('f1_score >= 0 AND f1_score <= 1', name='ck_val_f1_range'),
        Index('idx_validation_model_date', 'model_name', 'model_version', 'validation_date'),
    )
    
    def __repr__(self):
        return f"<ModelValidation(model={self.model_name}, version={self.model_version}, type={self.validation_type}, status={self.validation_status})>"
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'model_name': self.model_name,
            'model_version': self.model_version,
            'validation_type': self.validation_type,
            'validation_date': self.validation_date.isoformat() if self.validation_date else None,
            'precision_score': self.precision_score,
            'recall_score': self.recall_score,
            'f1_score': self.f1_score,
            'sample_size': self.sample_size,
            'baseline_precision': self.baseline_precision,
            'baseline_recall': self.baseline_recall,
            'baseline_f1': self.baseline_f1,
            'metrics': self.metrics,
            'report_path': self.report_path,
            'validation_status': self.validation_status,
            'notes': self.notes,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class ModelAlert(Base):
    """Model performance alerts tracking."""
    
    __tablename__ = 'model_alerts'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_level = Column(String(20), nullable=False, index=True)  # INFO, WARNING, CRITICAL
    alert_type = Column(String(50), nullable=False)  # DRIFT, ACCURACY_DROP, ERROR_RATE, etc.
    message = Column(Text, nullable=False)
    model_name = Column(String(100), nullable=True, index=True)
    model_version = Column(String(20), nullable=True)
    metrics = Column(JSON, nullable=True)  # Related metrics
    is_resolved = Column(Boolean, default=False, nullable=False, index=True)
    resolved_at = Column(DateTime, nullable=True)
    resolved_by = Column(String(50), nullable=True)
    resolution_notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    def __repr__(self):
        return f"<ModelAlert(level={self.alert_level}, type={self.alert_type}, resolved={self.is_resolved})>"
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'alert_level': self.alert_level,
            'alert_type': self.alert_type,
            'message': self.message,
            'model_name': self.model_name,
            'model_version': self.model_version,
            'metrics': self.metrics,
            'is_resolved': self.is_resolved,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'resolved_by': self.resolved_by,
            'resolution_notes': self.resolution_notes,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
