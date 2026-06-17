-- 005_signal_stopline.sql
-- The stop-line reading (peak congestion of the last two bins) — the
-- end-prioritized signal value, stored separately from the broad end zone.
-- Additive and idempotent.

ALTER TABLE traffic_observations
  ADD COLUMN IF NOT EXISTS signal_dominant_color   text,
  ADD COLUMN IF NOT EXISTS signal_congestion_score numeric;
