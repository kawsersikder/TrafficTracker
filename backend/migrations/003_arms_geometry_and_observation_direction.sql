-- 003_arms_geometry_and_observation_direction.sql
-- Persist arm geometry (the drawn polylines) and per-direction observations
-- so the whole intersection can be saved/reloaded and analyses stored per arm.
-- Additive and idempotent.

-- Arm polyline geometry + ordering.
ALTER TABLE intersection_arms
  ADD COLUMN IF NOT EXISTS points    jsonb,
  ADD COLUMN IF NOT EXISTS arm_order integer;

-- One arm label per intersection (lets us upsert arms by label).
CREATE UNIQUE INDEX IF NOT EXISTS uq_intersection_arms_label
  ON intersection_arms(intersection_id, arm_label);

-- Direction of a traffic reading: 'incoming' (toward the signal) or 'outgoing'.
ALTER TABLE traffic_observations
  ADD COLUMN IF NOT EXISTS direction text;

CREATE INDEX IF NOT EXISTS idx_traffic_observations_arm_dir_time
  ON traffic_observations(arm_id, direction, observed_at DESC);
