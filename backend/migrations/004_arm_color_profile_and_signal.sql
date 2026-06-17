-- 004_arm_color_profile_and_signal.sql
-- Richer per-arm traffic features for ML/DL research: an end-prioritized
-- "signal" reading, a red-queue length, and the spatial colour profile along
-- the line (start → signal). Additive and idempotent.

ALTER TABLE traffic_observations
  ADD COLUMN IF NOT EXISTS weighted_congestion_score numeric,
  ADD COLUMN IF NOT EXISTS end_dominant_color        text,
  ADD COLUMN IF NOT EXISTS end_green_pct             numeric,
  ADD COLUMN IF NOT EXISTS end_yellow_pct            numeric,
  ADD COLUMN IF NOT EXISTS end_red_pct               numeric,
  ADD COLUMN IF NOT EXISTS end_dark_red_pct          numeric,
  ADD COLUMN IF NOT EXISTS end_congestion_score      numeric,
  ADD COLUMN IF NOT EXISTS red_queue_px              double precision,
  ADD COLUMN IF NOT EXISTS red_queue_frac            numeric,
  ADD COLUMN IF NOT EXISTS arm_length_px             double precision,
  ADD COLUMN IF NOT EXISTS profile                   jsonb;
