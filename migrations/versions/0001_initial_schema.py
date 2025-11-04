"""Initial database schema

Revision ID: 0001
Revises: 
Create Date: 2025-01-27 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '0001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create vehicle_master table
    op.create_table('vehicle_master',
        sa.Column('vehicle_id', sa.String(length=50), nullable=False),
        sa.Column('vehicle_model', sa.String(length=100), nullable=False),
        sa.Column('invoice_date', sa.DateTime(), nullable=False),
        sa.Column('current_odometer', sa.Float(), nullable=False),
        sa.Column('dealer_code', sa.String(length=50), nullable=False),
        sa.Column('region_code', sa.String(length=50), nullable=False),
        sa.Column('terrain_type', sa.String(length=50), nullable=False),
        sa.Column('season_code', sa.String(length=20), nullable=False),
        sa.Column('ema_value', sa.Float(), nullable=True),
        sa.Column('ema_category', sa.String(length=20), nullable=True),
        sa.Column('last_updated', sa.DateTime(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('vehicle_id')
    )
    op.create_index(op.f('ix_vehicle_master_dealer_code'), 'vehicle_master', ['dealer_code'], unique=False)
    op.create_index(op.f('ix_vehicle_master_region_code'), 'vehicle_master', ['region_code'], unique=False)
    op.create_index(op.f('ix_vehicle_master_vehicle_id'), 'vehicle_master', ['vehicle_id'], unique=False)
    op.create_index(op.f('ix_vehicle_master_vehicle_model'), 'vehicle_master', ['vehicle_model'], unique=False)
    op.create_index('idx_vehicle_dealer_updated', 'vehicle_master', ['dealer_code', 'last_updated'], unique=False)
    op.create_index('idx_vehicle_ema_category', 'vehicle_master', ['ema_category'], unique=False)
    op.create_index('idx_vehicle_model_region', 'vehicle_master', ['vehicle_model', 'region_code'], unique=False)
    op.create_index('idx_vehicle_terrain_season', 'vehicle_master', ['terrain_type', 'season_code'], unique=False)

    # Create dealer_master table
    op.create_table('dealer_master',
        sa.Column('dealer_code', sa.String(length=50), nullable=False),
        sa.Column('dealer_name', sa.String(length=200), nullable=False),
        sa.Column('region_code', sa.String(length=50), nullable=False),
        sa.Column('city', sa.String(length=100), nullable=False),
        sa.Column('state', sa.String(length=100), nullable=False),
        sa.Column('pincode', sa.String(length=10), nullable=True),
        sa.Column('contact_person', sa.String(length=100), nullable=True),
        sa.Column('phone', sa.String(length=20), nullable=True),
        sa.Column('email', sa.String(length=100), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('dealer_code')
    )
    op.create_index(op.f('ix_dealer_master_region_code'), 'dealer_master', ['region_code'], unique=False)

    # Create part_master table
    op.create_table('part_master',
        sa.Column('part_code', sa.String(length=50), nullable=False),
        sa.Column('part_name', sa.String(length=200), nullable=False),
        sa.Column('part_category', sa.String(length=50), nullable=False),
        sa.Column('oem_part_number', sa.String(length=100), nullable=True),
        sa.Column('estimated_cost', sa.Float(), nullable=True),
        sa.Column('replacement_interval_km', sa.Integer(), nullable=True),
        sa.Column('replacement_interval_months', sa.Integer(), nullable=True),
        sa.Column('is_critical', sa.Boolean(), nullable=False),
        sa.Column('is_seasonal', sa.Boolean(), nullable=False),
        sa.Column('terrain_specific', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('part_code')
    )
    op.create_index(op.f('ix_part_master_part_category'), 'part_master', ['part_category'], unique=False)

    # Create service_history table
    op.create_table('service_history',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('vehicle_id', sa.String(length=50), nullable=False),
        sa.Column('service_date', sa.DateTime(), nullable=False),
        sa.Column('odometer_reading', sa.Float(), nullable=False),
        sa.Column('service_type', sa.String(length=50), nullable=False),
        sa.Column('parts_replaced', sa.JSON(), nullable=True),
        sa.Column('labor_cost', sa.Float(), nullable=True),
        sa.Column('parts_cost', sa.Float(), nullable=True),
        sa.Column('total_cost', sa.Float(), nullable=True),
        sa.Column('dealer_code', sa.String(length=50), nullable=False),
        sa.Column('technician_notes', sa.Text(), nullable=True),
        sa.Column('customer_complaints', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['vehicle_id'], ['vehicle_master.vehicle_id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_service_history_dealer_code'), 'service_history', ['dealer_code'], unique=False)
    op.create_index(op.f('ix_service_history_service_date'), 'service_history', ['service_date'], unique=False)
    op.create_index(op.f('ix_service_history_vehicle_id'), 'service_history', ['vehicle_id'], unique=False)
    op.create_index('idx_service_cost', 'service_history', ['total_cost'], unique=False)
    op.create_index('idx_service_dealer_type', 'service_history', ['dealer_code', 'service_type'], unique=False)
    op.create_index('idx_service_odometer', 'service_history', ['odometer_reading'], unique=False)
    op.create_index('idx_service_vehicle_date', 'service_history', ['vehicle_id', 'service_date'], unique=False)

    # Create part_recommendations table
    op.create_table('part_recommendations',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('vehicle_id', sa.String(length=50), nullable=False),
        sa.Column('part_code', sa.String(length=50), nullable=False),
        sa.Column('part_name', sa.String(length=200), nullable=False),
        sa.Column('part_category', sa.String(length=50), nullable=False),
        sa.Column('confidence_score', sa.Float(), nullable=False),
        sa.Column('rank', sa.Integer(), nullable=False),
        sa.Column('estimated_cost', sa.Float(), nullable=True),
        sa.Column('reasoning', sa.JSON(), nullable=True),
        sa.Column('model_version', sa.String(length=20), nullable=False),
        sa.Column('seasonal_adjustment', sa.Float(), nullable=True),
        sa.Column('terrain_adjustment', sa.Float(), nullable=True),
        sa.Column('final_confidence', sa.Float(), nullable=False),
        sa.Column('is_accepted', sa.Boolean(), nullable=True),
        sa.Column('feedback_date', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['vehicle_id'], ['vehicle_master.vehicle_id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_part_recommendations_part_category'), 'part_recommendations', ['part_category'], unique=False)
    op.create_index(op.f('ix_part_recommendations_part_code'), 'part_recommendations', ['part_code'], unique=False)
    op.create_index(op.f('ix_part_recommendations_vehicle_id'), 'part_recommendations', ['vehicle_id'], unique=False)
    op.create_index('idx_recommendation_category_confidence', 'part_recommendations', ['part_category', 'final_confidence'], unique=False)
    op.create_index('idx_recommendation_confidence', 'part_recommendations', ['final_confidence'], unique=False)
    op.create_index('idx_recommendation_feedback', 'part_recommendations', ['is_accepted', 'feedback_date'], unique=False)
    op.create_index('idx_recommendation_model_version', 'part_recommendations', ['model_version'], unique=False)
    op.create_index('idx_recommendation_vehicle_rank', 'part_recommendations', ['vehicle_id', 'rank'], unique=False)

    # Create user_feedback table
    op.create_table('user_feedback',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('recommendation_id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.String(length=50), nullable=False),
        sa.Column('feedback_type', sa.String(length=20), nullable=False),
        sa.Column('feedback_reason', sa.Text(), nullable=True),
        sa.Column('alternative_part', sa.String(length=50), nullable=True),
        sa.Column('actual_cost', sa.Float(), nullable=True),
        sa.Column('service_date', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['recommendation_id'], ['part_recommendations.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_user_feedback_recommendation_id'), 'user_feedback', ['recommendation_id'], unique=False)
    op.create_index(op.f('ix_user_feedback_user_id'), 'user_feedback', ['user_id'], unique=False)

    # Create parts_inventory table
    op.create_table('parts_inventory',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('part_code', sa.String(length=50), nullable=False),
        sa.Column('dealer_code', sa.String(length=50), nullable=False),
        sa.Column('current_stock', sa.Integer(), nullable=False),
        sa.Column('minimum_stock', sa.Integer(), nullable=False),
        sa.Column('maximum_stock', sa.Integer(), nullable=True),
        sa.Column('reorder_point', sa.Integer(), nullable=True),
        sa.Column('last_updated', sa.DateTime(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['part_code'], ['part_master.part_code'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('part_code', 'dealer_code', name='uq_part_dealer')
    )
    op.create_index(op.f('ix_parts_inventory_dealer_code'), 'parts_inventory', ['dealer_code'], unique=False)
    op.create_index(op.f('ix_parts_inventory_part_code'), 'parts_inventory', ['part_code'], unique=False)

    # Create seasonal_config table
    op.create_table('seasonal_config',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('season_code', sa.String(length=20), nullable=False),
        sa.Column('season_name', sa.String(length=50), nullable=False),
        sa.Column('start_month', sa.Integer(), nullable=False),
        sa.Column('end_month', sa.Integer(), nullable=False),
        sa.Column('adjustment_factor', sa.Float(), nullable=False),
        sa.Column('affected_parts', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('region_codes', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_seasonal_config_season_code'), 'seasonal_config', ['season_code'], unique=False)

    # Create terrain_config table
    op.create_table('terrain_config',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('terrain_type', sa.String(length=50), nullable=False),
        sa.Column('terrain_name', sa.String(length=100), nullable=False),
        sa.Column('adjustment_factor', sa.Float(), nullable=False),
        sa.Column('affected_parts', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('region_codes', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('usage_multiplier', sa.Float(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_terrain_config_terrain_type'), 'terrain_config', ['terrain_type'], unique=False)

    # Create model_versions table
    op.create_table('model_versions',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('model_name', sa.String(length=100), nullable=False),
        sa.Column('version', sa.String(length=20), nullable=False),
        sa.Column('model_path', sa.String(length=500), nullable=False),
        sa.Column('model_type', sa.String(length=50), nullable=False),
        sa.Column('training_date', sa.DateTime(), nullable=False),
        sa.Column('accuracy_score', sa.Float(), nullable=True),
        sa.Column('precision_score', sa.Float(), nullable=True),
        sa.Column('recall_score', sa.Float(), nullable=True),
        sa.Column('f1_score', sa.Float(), nullable=True),
        sa.Column('auc_score', sa.Float(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('model_name', 'version', name='uq_model_name_version')
    )
    op.create_index(op.f('ix_model_versions_model_name'), 'model_versions', ['model_name'], unique=False)
    op.create_index(op.f('ix_model_versions_version'), 'model_versions', ['version'], unique=False)

    # Create business_rules table
    op.create_table('business_rules',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('rule_name', sa.String(length=100), nullable=False),
        sa.Column('rule_type', sa.String(length=50), nullable=False),
        sa.Column('rule_condition', sa.JSON(), nullable=False),
        sa.Column('rule_action', sa.JSON(), nullable=False),
        sa.Column('priority', sa.Integer(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('effective_from', sa.DateTime(), nullable=True),
        sa.Column('effective_until', sa.DateTime(), nullable=True),
        sa.Column('created_by', sa.String(length=50), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_business_rules_rule_name'), 'business_rules', ['rule_name'], unique=False)

    # Create prediction_cache table
    op.create_table('prediction_cache',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('cache_key', sa.String(length=200), nullable=False),
        sa.Column('vehicle_id', sa.String(length=50), nullable=False),
        sa.Column('model_version', sa.String(length=20), nullable=False),
        sa.Column('predictions', sa.JSON(), nullable=False),
        sa.Column('confidence_scores', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('expires_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('cache_key', name='uq_cache_key')
    )
    op.create_index(op.f('ix_prediction_cache_cache_key'), 'prediction_cache', ['cache_key'], unique=False)
    op.create_index(op.f('ix_prediction_cache_expires_at'), 'prediction_cache', ['expires_at'], unique=False)
    op.create_index(op.f('ix_prediction_cache_vehicle_id'), 'prediction_cache', ['vehicle_id'], unique=False)

    # Add check constraints
    op.create_check_constraint('ck_current_stock_positive', 'parts_inventory', 'current_stock >= 0')
    op.create_check_constraint('ck_minimum_stock_positive', 'parts_inventory', 'minimum_stock >= 0')
    op.create_check_constraint('ck_start_month_valid', 'seasonal_config', 'start_month >= 1 AND start_month <= 12')
    op.create_check_constraint('ck_end_month_valid', 'seasonal_config', 'end_month >= 1 AND end_month <= 12')
    op.create_check_constraint('ck_adjustment_factor_positive', 'seasonal_config', 'adjustment_factor > 0')
    op.create_check_constraint('ck_terrain_adjustment_positive', 'terrain_config', 'adjustment_factor > 0')
    op.create_check_constraint('ck_usage_multiplier_positive', 'terrain_config', 'usage_multiplier > 0')
    op.create_check_constraint('ck_accuracy_range', 'model_versions', 'accuracy_score >= 0 AND accuracy_score <= 1')
    op.create_check_constraint('ck_precision_range', 'model_versions', 'precision_score >= 0 AND precision_score <= 1')
    op.create_check_constraint('ck_recall_range', 'model_versions', 'recall_score >= 0 AND recall_score <= 1')
    op.create_check_constraint('ck_f1_range', 'model_versions', 'f1_score >= 0 AND f1_score <= 1')
    op.create_check_constraint('ck_auc_range', 'model_versions', 'auc_score >= 0 AND auc_score <= 1')
    op.create_check_constraint('ck_priority_positive', 'business_rules', 'priority >= 0')


def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_table('prediction_cache')
    op.drop_table('business_rules')
    op.drop_table('model_versions')
    op.drop_table('terrain_config')
    op.drop_table('seasonal_config')
    op.drop_table('parts_inventory')
    op.drop_table('user_feedback')
    op.drop_table('part_recommendations')
    op.drop_table('service_history')
    op.drop_table('part_master')
    op.drop_table('dealer_master')
    op.drop_table('vehicle_master')
