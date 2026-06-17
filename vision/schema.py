from pydantic import BaseModel
from typing import List

class ColorStats(BaseModel):
    """Colour breakdown of a set of sampled pixels."""
    dominant_color: str
    green_pct: float
    yellow_pct: float
    red_pct: float
    dark_red_pct: float
    # Congestion in [0,1]: green=free-flow, darker reds weigh heavier.
    congestion_score: float = 0.0

class ArmBin(BaseModel):
    """One slice of an arm, ordered start (A) → signal end (B)."""
    index: int
    pos: float            # bin centre, 0.0 = start, 1.0 = signal end
    length_px: float      # physical length of this slice along the line
    dominant_color: str
    green_pct: float
    yellow_pct: float
    red_pct: float
    dark_red_pct: float
    congestion_score: float

class ArmObservation(BaseModel):
    arm_label: str
    # ── Whole-line aggregate (kept for convenience) ──
    dominant_color: str
    green_pct: float
    yellow_pct: float
    red_pct: float
    dark_red_pct: float
    congestion_score: float
    # ── End-prioritized signal (the queue right before the signal) ──
    weighted_congestion_score: float = 0.0   # whole line, weighted toward the end
    end: ColorStats                           # last END_FRAC of the line near the signal
    signal: ColorStats                        # the stop-line itself (peak of the last 2 bins)
    red_queue_px: float = 0.0                 # contiguous congested length back from signal
    red_queue_frac: float = 0.0               # …as a fraction of the line length
    arm_length_px: float = 0.0
    # ── Spatial colour profile, start → signal end ──
    profile: List[ArmBin] = []
    estimated_lane_groups: int
    confidence: float

class IntersectionObservation(BaseModel):
    intersection_id: str
    observed_at: str
    intersection_type: str
    arms: List[ArmObservation]
    run_confidence: float
    extractor_version: str
