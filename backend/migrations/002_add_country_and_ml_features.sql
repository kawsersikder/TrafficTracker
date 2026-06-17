-- 002_add_country_and_ml_features.sql
-- Adds country (categorical), geospatial, and ML/DL feature columns.
-- Safe to run once on an existing database created by 001_initial_schema.sql.

-- Categorical country enum (stable, low-cardinality feature for ML encoding).
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'Country') THEN
    CREATE TYPE "Country" AS ENUM (
      'BANGLADESH', 'INDIA', 'THAILAND', 'PHILIPPINES', 'MALAYSIA'
    );
  END IF;
END$$;

-- Intersection: country + geospatial + arm_count features.
ALTER TABLE intersections
  ADD COLUMN IF NOT EXISTS country   "Country",
  ADD COLUMN IF NOT EXISTS latitude  double precision,
  ADD COLUMN IF NOT EXISTS longitude double precision,
  ADD COLUMN IF NOT EXISTS arm_count integer;

CREATE INDEX IF NOT EXISTS idx_intersections_country ON intersections(country);

-- Traffic observations: denormalized country + engineered temporal features
-- so the training table can be consumed without joins.
ALTER TABLE traffic_observations
  ADD COLUMN IF NOT EXISTS country     "Country",
  ADD COLUMN IF NOT EXISTS hour_of_day integer,
  ADD COLUMN IF NOT EXISTS day_of_week integer,
  ADD COLUMN IF NOT EXISTS is_weekend  boolean;

CREATE INDEX IF NOT EXISTS idx_traffic_observations_country_time
  ON traffic_observations(country, observed_at DESC);
