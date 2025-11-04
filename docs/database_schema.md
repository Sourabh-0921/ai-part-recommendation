# Database Schema Documentation

## Overview

The AI Parts Recommendation System uses PostgreSQL as the primary database with a comprehensive schema designed to support machine learning-based parts recommendations, business rules, and performance optimization.

## Database Architecture

### Core Tables

#### 1. Vehicle Master (`vehicle_master`)
Primary table storing vehicle information and usage patterns.

| Column | Type | Description |
|--------|------|-------------|
| `vehicle_id` | VARCHAR(50) | Primary key, unique vehicle identifier |
| `vehicle_model` | VARCHAR(100) | Vehicle model name |
| `invoice_date` | TIMESTAMP | Vehicle purchase date |
| `current_odometer` | FLOAT | Current odometer reading in km |
| `dealer_code` | VARCHAR(50) | Associated dealer code |
| `region_code` | VARCHAR(50) | Geographic region |
| `terrain_type` | VARCHAR(50) | Terrain type (URBAN, HILLY, HIGHWAY) |
| `season_code` | VARCHAR(20) | Current season (SUMMER, MONSOON, WINTER) |
| `ema_value` | FLOAT | Exponential Moving Average usage |
| `ema_category` | VARCHAR(20) | Usage category (HIGH_USAGE, MEDIUM_USAGE, LOW_USAGE) |
| `last_updated` | TIMESTAMP | Last update timestamp |
| `created_at` | TIMESTAMP | Record creation timestamp |

**Indexes:**
- Primary key on `vehicle_id`
- Index on `vehicle_model`
- Index on `dealer_code`
- Index on `region_code`
- Composite index on `vehicle_model, region_code`
- Composite index on `terrain_type, season_code`
- Index on `ema_category`
- Composite index on `dealer_code, last_updated`

#### 2. Service History (`service_history`)
Tracks all service records for vehicles.

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary key, auto-increment |
| `vehicle_id` | VARCHAR(50) | Foreign key to vehicle_master |
| `service_date` | TIMESTAMP | Service date |
| `odometer_reading` | FLOAT | Odometer reading at service |
| `service_type` | VARCHAR(50) | Service type (REGULAR, MAJOR, EMERGENCY) |
| `parts_replaced` | JSON | Array of replaced part codes |
| `labor_cost` | FLOAT | Labor cost |
| `parts_cost` | FLOAT | Parts cost |
| `total_cost` | FLOAT | Total service cost |
| `dealer_code` | VARCHAR(50) | Service dealer |
| `technician_notes` | TEXT | Technician observations |
| `customer_complaints` | TEXT | Customer reported issues |
| `created_at` | TIMESTAMP | Record creation timestamp |

**Indexes:**
- Primary key on `id`
- Foreign key index on `vehicle_id`
- Index on `service_date`
- Index on `dealer_code`
- Composite index on `vehicle_id, service_date`
- Composite index on `dealer_code, service_type`
- Index on `odometer_reading`
- Index on `total_cost`

#### 3. Parts Master (`part_master`)
Catalog of all available parts.

| Column | Type | Description |
|--------|------|-------------|
| `part_code` | VARCHAR(50) | Primary key, unique part identifier |
| `part_name` | VARCHAR(200) | Human-readable part name |
| `part_category` | VARCHAR(50) | Part category (Brakes, Engine, etc.) |
| `oem_part_number` | VARCHAR(100) | OEM part number |
| `estimated_cost` | FLOAT | Estimated cost in INR |
| `replacement_interval_km` | INTEGER | Replacement interval in km |
| `replacement_interval_months` | INTEGER | Replacement interval in months |
| `is_critical` | BOOLEAN | Critical part flag |
| `is_seasonal` | BOOLEAN | Seasonal part flag |
| `terrain_specific` | BOOLEAN | Terrain-specific part flag |
| `created_at` | TIMESTAMP | Record creation timestamp |
| `updated_at` | TIMESTAMP | Last update timestamp |

**Indexes:**
- Primary key on `part_code`
- Index on `part_category`

#### 4. Parts Inventory (`parts_inventory`)
Tracks inventory levels at each dealer.

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary key, auto-increment |
| `part_code` | VARCHAR(50) | Foreign key to part_master |
| `dealer_code` | VARCHAR(50) | Dealer identifier |
| `current_stock` | INTEGER | Current stock level |
| `minimum_stock` | INTEGER | Minimum stock threshold |
| `maximum_stock` | INTEGER | Maximum stock level |
| `reorder_point` | INTEGER | Reorder threshold |
| `last_updated` | TIMESTAMP | Last update timestamp |
| `created_at` | TIMESTAMP | Record creation timestamp |

**Constraints:**
- Unique constraint on `part_code, dealer_code`
- Check constraint: `current_stock >= 0`
- Check constraint: `minimum_stock >= 0`

#### 5. Part Recommendations (`part_recommendations`)
ML-generated part recommendations.

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary key, auto-increment |
| `vehicle_id` | VARCHAR(50) | Foreign key to vehicle_master |
| `part_code` | VARCHAR(50) | Recommended part code |
| `part_name` | VARCHAR(200) | Part name |
| `part_category` | VARCHAR(50) | Part category |
| `confidence_score` | FLOAT | Base confidence score (0-100) |
| `rank` | INTEGER | Recommendation rank (1-10) |
| `estimated_cost` | FLOAT | Estimated cost |
| `reasoning` | JSON | Explanation of recommendation |
| `model_version` | VARCHAR(20) | ML model version |
| `seasonal_adjustment` | FLOAT | Seasonal adjustment factor |
| `terrain_adjustment` | FLOAT | Terrain adjustment factor |
| `final_confidence` | FLOAT | Final confidence after adjustments |
| `is_accepted` | BOOLEAN | User acceptance status |
| `feedback_date` | TIMESTAMP | Feedback timestamp |
| `created_at` | TIMESTAMP | Record creation timestamp |

**Indexes:**
- Primary key on `id`
- Foreign key index on `vehicle_id`
- Index on `part_code`
- Index on `part_category`
- Index on `model_version`
- Composite index on `vehicle_id, rank`
- Index on `final_confidence`
- Composite index on `part_category, final_confidence`
- Composite index on `is_accepted, feedback_date`

### Configuration Tables

#### 6. Dealer Master (`dealer_master`)
Dealer information and locations.

| Column | Type | Description |
|--------|------|-------------|
| `dealer_code` | VARCHAR(50) | Primary key, dealer identifier |
| `dealer_name` | VARCHAR(200) | Dealer name |
| `region_code` | VARCHAR(50) | Geographic region |
| `city` | VARCHAR(100) | City name |
| `state` | VARCHAR(100) | State name |
| `pincode` | VARCHAR(10) | Postal code |
| `contact_person` | VARCHAR(100) | Contact person name |
| `phone` | VARCHAR(20) | Phone number |
| `email` | VARCHAR(100) | Email address |
| `is_active` | BOOLEAN | Active status |
| `created_at` | TIMESTAMP | Record creation timestamp |
| `updated_at` | TIMESTAMP | Last update timestamp |

#### 7. Seasonal Configuration (`seasonal_config`)
Seasonal adjustment rules.

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary key, auto-increment |
| `season_code` | VARCHAR(20) | Season identifier |
| `season_name` | VARCHAR(50) | Season name |
| `start_month` | INTEGER | Start month (1-12) |
| `end_month` | INTEGER | End month (1-12) |
| `adjustment_factor` | FLOAT | Adjustment multiplier |
| `affected_parts` | ARRAY | List of affected part codes |
| `region_codes` | ARRAY | List of affected regions |
| `is_active` | BOOLEAN | Active status |
| `created_at` | TIMESTAMP | Record creation timestamp |
| `updated_at` | TIMESTAMP | Last update timestamp |

**Constraints:**
- Check constraint: `start_month >= 1 AND start_month <= 12`
- Check constraint: `end_month >= 1 AND end_month <= 12`
- Check constraint: `adjustment_factor > 0`

#### 8. Terrain Configuration (`terrain_config`)
Terrain-based adjustment rules.

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary key, auto-increment |
| `terrain_type` | VARCHAR(50) | Terrain identifier |
| `terrain_name` | VARCHAR(100) | Terrain description |
| `adjustment_factor` | FLOAT | Adjustment multiplier |
| `affected_parts` | ARRAY | List of affected part codes |
| `region_codes` | ARRAY | List of affected regions |
| `usage_multiplier` | FLOAT | Usage pattern multiplier |
| `is_active` | BOOLEAN | Active status |
| `created_at` | TIMESTAMP | Record creation timestamp |
| `updated_at` | TIMESTAMP | Last update timestamp |

**Constraints:**
- Check constraint: `adjustment_factor > 0`
- Check constraint: `usage_multiplier > 0`

#### 9. Business Rules (`business_rules`)
Configurable business logic rules.

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary key, auto-increment |
| `rule_name` | VARCHAR(100) | Rule name |
| `rule_type` | VARCHAR(50) | Rule type (SEASONAL, TERRAIN, USAGE, COST) |
| `rule_condition` | JSON | Rule conditions |
| `rule_action` | JSON | Rule actions |
| `priority` | INTEGER | Rule priority |
| `is_active` | BOOLEAN | Active status |
| `effective_from` | TIMESTAMP | Effective start date |
| `effective_until` | TIMESTAMP | Effective end date |
| `created_by` | VARCHAR(50) | Creator identifier |
| `created_at` | TIMESTAMP | Record creation timestamp |
| `updated_at` | TIMESTAMP | Last update timestamp |

**Constraints:**
- Check constraint: `priority >= 0`

### ML and Performance Tables

#### 10. Model Versions (`model_versions`)
ML model version tracking.

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary key, auto-increment |
| `model_name` | VARCHAR(100) | Model name |
| `version` | VARCHAR(20) | Model version |
| `model_path` | VARCHAR(500) | Model file path |
| `model_type` | VARCHAR(50) | Model type (LIGHTGBM, ENSEMBLE) |
| `training_date` | TIMESTAMP | Training date |
| `accuracy_score` | FLOAT | Accuracy score (0-1) |
| `precision_score` | FLOAT | Precision score (0-1) |
| `recall_score` | FLOAT | Recall score (0-1) |
| `f1_score` | FLOAT | F1 score (0-1) |
| `auc_score` | FLOAT | AUC score (0-1) |
| `is_active` | BOOLEAN | Active model flag |
| `metadata` | JSON | Additional model metadata |
| `created_at` | TIMESTAMP | Record creation timestamp |

**Constraints:**
- Unique constraint on `model_name, version`
- Check constraints for all score ranges (0-1)

#### 11. User Feedback (`user_feedback`)
User interaction tracking.

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary key, auto-increment |
| `recommendation_id` | INTEGER | Foreign key to part_recommendations |
| `user_id` | VARCHAR(50) | User identifier |
| `feedback_type` | VARCHAR(20) | Feedback type (ACCEPTED, REJECTED, MODIFIED) |
| `feedback_reason` | TEXT | Feedback reason |
| `alternative_part` | VARCHAR(50) | Alternative part code |
| `actual_cost` | FLOAT | Actual service cost |
| `service_date` | TIMESTAMP | Service date |
| `created_at` | TIMESTAMP | Record creation timestamp |

#### 12. Prediction Cache (`prediction_cache`)
Performance optimization cache.

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary key, auto-increment |
| `cache_key` | VARCHAR(200) | Unique cache key |
| `vehicle_id` | VARCHAR(50) | Vehicle identifier |
| `model_version` | VARCHAR(20) | Model version used |
| `predictions` | JSON | Cached predictions |
| `confidence_scores` | JSON | Cached confidence scores |
| `created_at` | TIMESTAMP | Cache creation timestamp |
| `expires_at` | TIMESTAMP | Cache expiration timestamp |

**Constraints:**
- Unique constraint on `cache_key`

## Database Relationships

### Primary Relationships
- `vehicle_master` → `service_history` (1:N)
- `vehicle_master` → `part_recommendations` (1:N)
- `part_master` → `parts_inventory` (1:N)
- `part_recommendations` → `user_feedback` (1:N)

### Foreign Key Constraints
- `service_history.vehicle_id` → `vehicle_master.vehicle_id`
- `part_recommendations.vehicle_id` → `vehicle_master.vehicle_id`
- `parts_inventory.part_code` → `part_master.part_code`
- `user_feedback.recommendation_id` → `part_recommendations.id`

## Performance Optimizations

### Indexing Strategy
1. **Primary Keys**: All tables have primary key indexes
2. **Foreign Keys**: All foreign keys are indexed
3. **Composite Indexes**: Multi-column indexes for common query patterns
4. **Covering Indexes**: Indexes that include frequently accessed columns

### Query Optimization
1. **Partitioning**: Consider partitioning `service_history` by date
2. **Materialized Views**: For complex analytics queries
3. **Connection Pooling**: Configured for high concurrency
4. **Caching**: Redis integration for frequently accessed data

## Data Types and Constraints

### JSON Columns
- `service_history.parts_replaced`: Array of part codes
- `part_recommendations.reasoning`: Recommendation explanation
- `business_rules.rule_condition`: Rule conditions
- `business_rules.rule_action`: Rule actions
- `model_versions.metadata`: Model metadata
- `prediction_cache.predictions`: Cached predictions
- `prediction_cache.confidence_scores`: Cached confidence scores

### Array Columns (PostgreSQL)
- `seasonal_config.affected_parts`: Array of part codes
- `seasonal_config.region_codes`: Array of region codes
- `terrain_config.affected_parts`: Array of part codes
- `terrain_config.region_codes`: Array of region codes

### Check Constraints
- Stock levels must be non-negative
- Month values must be 1-12
- Adjustment factors must be positive
- Score values must be 0-1 range
- Priority values must be non-negative

## Migration Management

### Alembic Configuration
- **Location**: `migrations/` directory
- **Configuration**: `alembic.ini`
- **Environment**: `migrations/env.py`
- **Template**: `migrations/script.py.mako`

### Migration Commands
```bash
# Initialize Alembic
alembic init migrations

# Create new migration
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1

# Check current version
alembic current
```

## Sample Data

The database includes sample data for:
- **3 Dealers**: Mumbai, Delhi, Bangalore
- **5 Parts**: Brake pads, oil filter, air filter, spark plugs
- **3 Vehicles**: Different models and usage patterns
- **3 Seasons**: Summer, Monsoon, Winter configurations
- **3 Terrain Types**: Urban, Hilly, Highway
- **2 Business Rules**: High usage and monsoon rules
- **1 Model Version**: Initial ML model

## Security Considerations

### Data Protection
- **Encryption**: Sensitive data encrypted at rest
- **Access Control**: Role-based access control
- **Audit Trail**: All changes tracked with timestamps
- **Input Validation**: All inputs validated and sanitized

### Backup Strategy
- **Daily Backups**: Automated daily backups
- **Point-in-time Recovery**: WAL-based recovery
- **Cross-region Replication**: For disaster recovery
- **Data Retention**: Configurable retention policies

## Monitoring and Maintenance

### Performance Monitoring
- **Query Performance**: Slow query identification
- **Index Usage**: Index effectiveness monitoring
- **Connection Pooling**: Connection usage tracking
- **Cache Hit Rates**: Cache performance metrics

### Maintenance Tasks
- **VACUUM**: Regular table maintenance
- **ANALYZE**: Statistics updates
- **REINDEX**: Index rebuilding
- **Archival**: Old data archival strategy

## Troubleshooting

### Common Issues
1. **Connection Timeouts**: Check connection pool settings
2. **Slow Queries**: Review query plans and indexes
3. **Lock Contention**: Monitor transaction durations
4. **Disk Space**: Monitor database growth

### Diagnostic Queries
```sql
-- Check table sizes
SELECT schemaname, tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Check index usage
SELECT schemaname, tablename, indexname, idx_scan, idx_tup_read, idx_tup_fetch
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;

-- Check slow queries
SELECT query, mean_time, calls, total_time
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 10;
```

This comprehensive database schema supports the AI Parts Recommendation System with proper indexing, constraints, and relationships optimized for machine learning workloads and high-performance operations.
