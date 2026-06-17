create table intersections (
  id text primary key,
  name text not null,
  city text,
  google_maps_url text not null,
  expected_type text,
  main_road_label text,
  notes text,
  created_at timestamptz default now()
);

create table intersection_arms (
  id bigserial primary key,
  intersection_id text not null references intersections(id) on delete cascade,
  arm_label text not null,
  angle_start numeric,
  angle_end numeric,
  is_main boolean default false,
  expected_lane_groups integer,
  notes text
);

create table traffic_observations (
  id bigserial primary key,
  intersection_id text not null references intersections(id) on delete cascade,
  arm_id bigint references intersection_arms(id) on delete set null,
  observed_at timestamptz not null,
  dominant_color text,
  green_pct numeric,
  yellow_pct numeric,
  red_pct numeric,
  dark_red_pct numeric,
  congestion_score numeric,
  estimated_lane_groups integer,
  confidence numeric,
  extractor_version text,
  created_at timestamptz default now()
);

create table processing_runs (
  id bigserial primary key,
  intersection_id text not null references intersections(id) on delete cascade,
  started_at timestamptz not null,
  finished_at timestamptz,
  status text not null,
  message text,
  extractor_version text
);

create index idx_traffic_observations_time on traffic_observations(observed_at);
create index idx_traffic_observations_intersection_time on traffic_observations(intersection_id, observed_at desc);
create index idx_traffic_observations_arm_time on traffic_observations(arm_id, observed_at desc);
