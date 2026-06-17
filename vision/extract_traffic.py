import sys
import json
import datetime
from preprocess import create_color_masks
from schema import IntersectionObservation, ArmObservation, ColorStats, ArmBin
import math
import numpy as np
import cv2

BIN_COUNT = 8          # slices per arm (start → signal end), fixed for ML feature vectors
END_FRAC = 0.30        # last 30% near the signal = the broad "end" zone
QUEUE_CONG = 0.45      # a bin counts toward the red queue at/above this congestion


def calculate_traffic_stats(counts: dict) -> dict:
    """Colour breakdown for the pixels sampled along one drawn line.

    Dominant colour is SEVERITY-AWARE: red + dark_red are treated as one
    "red family" so a slice that is e.g. 0.29 red + 0.27 dark_red reads as red,
    not yellow (single-bucket argmax would mislabel it).
    """
    total = sum(counts.values())
    if total == 0:
        return {
            "dominant_color": "none",
            "green_pct": 0.0, "yellow_pct": 0.0, "red_pct": 0.0, "dark_red_pct": 0.0,
            "congestion_score": 0.0,
        }
    green_pct = round(counts['green'] / total, 3)
    yellow_pct = round(counts['yellow'] / total, 3)
    red_pct = round(counts['red'] / total, 3)
    dark_red_pct = round(counts['dark_red'] / total, 3)

    red_family = red_pct + dark_red_pct
    cands = {'green': green_pct, 'yellow': yellow_pct, 'red_family': red_family}
    top = max(cands, key=cands.get)
    if top == 'red_family':
        dominant = 'dark_red' if dark_red_pct >= red_pct else 'red'
    else:
        dominant = top

    # Congestion in [0,1]: green is free-flow, darker reds weigh heavier.
    congestion = min(1.0, 0.4 * yellow_pct + 0.85 * red_pct + 1.0 * dark_red_pct)
    return {
        "dominant_color": dominant,
        "green_pct": green_pct,
        "yellow_pct": yellow_pct,
        "red_pct": red_pct,
        "dark_red_pct": dark_red_pct,
        "congestion_score": round(congestion, 3),
    }

def _normalize_arm_points(arm: dict, legacy_center: dict | None) -> list:
    """Return a list of {x, y} points for an arm across all config formats.

    New format: arm['points'] = [{x, y}, ...] (a directed polyline / road).
    Legacy format: a shared center + arm['end'] → a 2-point line center→end.
    """
    raw = arm.get('points')
    if isinstance(raw, list) and len(raw) >= 2:
        return [{'x': float(p['x']), 'y': float(p['y'])} for p in raw]
    if arm.get('end') is not None and legacy_center is not None:
        return [
            {'x': float(legacy_center['x']), 'y': float(legacy_center['y'])},
            {'x': float(arm['end']['x']), 'y': float(arm['end']['y'])},
        ]
    return []


def _closest_segment(px: float, py: float, points: list):
    """Closest polyline segment to (px, py).

    Returns (min_dist, seg_index, t) where seg_index is the index of the
    segment start vertex and t is the clamped projection parameter [0,1] along
    that segment. Used for corridor membership, local travel direction, and
    the pixel's arc-length position along the arm.
    """
    best_dist = float('inf')
    best_i = 0
    best_t = 0.0
    for i in range(len(points) - 1):
        ax, ay = points[i]['x'], points[i]['y']
        bx, by = points[i + 1]['x'], points[i + 1]['y']
        abx, aby = bx - ax, by - ay
        seg_len2 = abx * abx + aby * aby
        if seg_len2 == 0:
            continue
        t = ((px - ax) * abx + (py - ay) * aby) / seg_len2
        t = max(0.0, min(1.0, t))
        cx_, cy_ = ax + abx * t, ay + aby * t
        d = math.hypot(px - cx_, py - cy_)
        if d < best_dist:
            best_dist = d
            best_i = i
            best_t = t
    return best_dist, best_i, best_t


def extract_traffic(intersection_id: str, image_path: str, expected_type: str = "4-arm", config_str: str = "none", annotated_path: str = None) -> str:
    import base64
    config = {}
    if config_str and config_str != 'none':
        try:
            config = json.loads(base64.b64decode(config_str).decode('utf-8'))
        except Exception as e:
            print("Failed to parse intersection config:", e, file=sys.stderr)

    # ── ARM MODEL: each arm is a directed multi-point polyline (a road) ──
    # New format: arms[{ name, points: [{x,y}, ...] }] — no shared center.
    # Legacy formats still supported:
    #   • center + arms[{ end }]            → 2-point line center→end
    #   • arms_config[{ angle, width }]     → radial endpoints from a center
    legacy_center = config.get('center', None)
    raw_arms = config.get('arms', [])

    legacy_arms_config = config.get('arms_config', [])
    if legacy_arms_config and not raw_arms:
        cx0 = config.get('center_x', 640)
        cy0 = config.get('center_y', 400)
        outer_radius = config.get('outer_radius', 200)
        legacy_center = {'x': cx0, 'y': cy0}
        for arm_cfg in legacy_arms_config:
            angle_rad = math.radians(arm_cfg['angle'])
            raw_arms.append({
                'name': arm_cfg['name'],
                'end': {
                    'x': int(cx0 + outer_radius * math.cos(angle_rad)),
                    'y': int(cy0 + outer_radius * math.sin(angle_rad)),
                },
            })

    masks = create_color_masks(image_path)
    observations = []

    # Pre-compute all colored pixel coordinates
    color_coords = {}
    for color in ['green', 'yellow', 'red', 'dark_red']:
        y_coords, x_coords = np.nonzero(masks[color])
        color_coords[color] = list(zip(x_coords, y_coords))

    img_draw = None
    if annotated_path:
        img_draw = cv2.imread(image_path)

    # The read corridor hugs the drawn line tightly and is controlled by each
    # arm's on-map "width" — so the sampler reads only the road the line sits
    # on, not the U-turn / slip-lane connectors that crowd the junction.
    DEFAULT_ARM_WIDTH = 3

    def half_width_for(arm):
        w = arm.get('width', DEFAULT_ARM_WIDTH) or DEFAULT_ARM_WIDTH
        return min(30.0, max(8.0, w * 3.0))   # default w=3 → 9px each side

    # ── 1) Pre-compute geometry for every arm (the line the user drew) ──
    def new_counts():
        return {'green': 0, 'yellow': 0, 'red': 0, 'dark_red': 0}

    geoms = []
    for arm in raw_arms:
        name = arm.get('name', 'arm')
        points = _normalize_arm_points(arm, legacy_center)
        if len(points) < 2:
            continue
        seg_lens = [0.0] * (len(points) - 1)
        cum = [0.0] * len(points)
        for i in range(len(points) - 1):
            seg_lens[i] = math.hypot(points[i + 1]['x'] - points[i]['x'],
                                     points[i + 1]['y'] - points[i]['y'])
            cum[i + 1] = cum[i] + seg_lens[i]
        total_len = cum[-1]
        if total_len < 5:
            continue
        geoms.append({
            'name': name, 'points': points, 'seg_lens': seg_lens, 'cum': cum,
            'total_len': total_len,
            'half_width': half_width_for(arm),
            'counts': new_counts(),                              # whole line
            'end_counts': new_counts(),                          # last END_FRAC near signal
            'bins': [new_counts() for _ in range(BIN_COUNT)],    # start → signal profile
        })

    # Widest corridor across all arms — a quick reject before the exact check.
    max_half = max((g['half_width'] for g in geoms), default=0.0)

    # ── 2) Read the colours along each drawn line ──
    # Each coloured pixel is assigned to its NEAREST arm only (exclusive), kept
    # only if inside that arm's tight corridor, then placed into a bin by its
    # position ALONG the line so we capture the start→signal colour profile.
    for color, coords in color_coords.items():
        for px, py in coords:
            best = None  # (dist, gi, seg_i, t)
            for gi, g in enumerate(geoms):
                d, seg_i, t = _closest_segment(px, py, g['points'])
                if best is None or d < best[0]:
                    best = (d, gi, seg_i, t)
            if best is None or best[0] > max_half:
                continue
            d, gi, seg_i, t = best
            g = geoms[gi]
            if d > g['half_width']:
                continue
            frac = (g['cum'][seg_i] + t * g['seg_lens'][seg_i]) / g['total_len']
            frac = min(0.999999, max(0.0, frac))
            g['counts'][color] += 1
            g['bins'][int(frac * BIN_COUNT)][color] += 1
            if frac >= 1.0 - END_FRAC:
                g['end_counts'][color] += 1
            if img_draw is not None:
                # End/signal pixels in magenta, the rest in yellow.
                dot = (200, 0, 255) if frac >= 1.0 - END_FRAC else (0, 255, 255)
                cv2.circle(img_draw, (px, py), 1, dot, -1)

    # ── 3) Build per-arm reading (profile + end-priority) + annotation ──
    for g in geoms:
        overall = calculate_traffic_stats(g['counts'])
        end = calculate_traffic_stats(g['end_counts'])
        bin_len = g['total_len'] / BIN_COUNT

        profile = []
        for i, bc in enumerate(g['bins']):
            s = calculate_traffic_stats(bc)
            profile.append(ArmBin(
                index=i, pos=round((i + 0.5) / BIN_COUNT, 4), length_px=round(bin_len, 2),
                dominant_color=s['dominant_color'], green_pct=s['green_pct'],
                yellow_pct=s['yellow_pct'], red_pct=s['red_pct'], dark_red_pct=s['dark_red_pct'],
                congestion_score=s['congestion_score'],
            ))

        # End-weighted congestion: bins near the signal weigh more (w = pos + 0.15).
        num = den = 0.0
        for b in profile:
            n = sum(g['bins'][b.index].values())
            if n == 0:
                continue
            w = (b.pos + 0.15) * n
            num += b.congestion_score * w
            den += w
        weighted = round(num / den, 3) if den else overall['congestion_score']

        # Red queue: contiguous CONGESTED bins (by congestion score, so heavy
        # red+dark-red mixes count even if no single colour is the plurality)
        # measured back from the signal end.
        q = 0
        for b in reversed(profile):
            if b.congestion_score >= QUEUE_CONG:
                q += 1
            else:
                break
        red_queue_px = round(q * bin_len, 2)
        red_queue_frac = round(q / BIN_COUNT, 3)

        # Signal = the stop-line itself: the worst (peak-congestion) of the last
        # two bins, so the reading isn't diluted by the broad END_FRAC window or
        # by a tip that overshoots into a calmer patch.
        tail = profile[-2:] if len(profile) >= 2 else profile
        sb = max(tail, key=lambda b: b.congestion_score) if tail else None
        signal = ColorStats(
            dominant_color=sb.dominant_color if sb else 'none',
            green_pct=sb.green_pct if sb else 0.0,
            yellow_pct=sb.yellow_pct if sb else 0.0,
            red_pct=sb.red_pct if sb else 0.0,
            dark_red_pct=sb.dark_red_pct if sb else 0.0,
            congestion_score=sb.congestion_score if sb else 0.0,
        )

        pts = g['points']
        if img_draw is not None:
            poly = np.array([(int(p['x']), int(p['y'])) for p in pts])
            cv2.polylines(img_draw, [poly], False, (255, 255, 255), 2)
            a_pt = (int(pts[-2]['x']), int(pts[-2]['y']))
            b_pt = (int(pts[-1]['x']), int(pts[-1]['y']))
            cv2.arrowedLine(img_draw, a_pt, b_pt, (255, 255, 255), 2, tipLength=0.3)
            cv2.circle(img_draw, b_pt, 7, (60, 60, 220), -1)
            cv2.circle(img_draw, b_pt, 7, (255, 255, 255), 2)
            cv2.putText(img_draw, g['name'], (b_pt[0] + 6, b_pt[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(img_draw, g['name'], (b_pt[0] + 6, b_pt[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        observations.append(ArmObservation(
            arm_label=g['name'],
            **overall,
            weighted_congestion_score=weighted,
            end=ColorStats(**end),
            signal=signal,
            red_queue_px=red_queue_px,
            red_queue_frac=red_queue_frac,
            arm_length_px=round(g['total_len'], 2),
            profile=profile,
            estimated_lane_groups=2,
            confidence=0.85,
        ))

    if img_draw is not None and annotated_path:
        cv2.imwrite(annotated_path, img_draw)
        print(f"Annotated image saved to {annotated_path}", file=sys.stderr)

    result = IntersectionObservation(
        intersection_id=intersection_id,
        observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        intersection_type=expected_type,
        arms=observations,
        run_confidence=0.9,
        extractor_version="v3.1.0"
    )
    
    return result.model_dump_json(indent=2)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extract_traffic.py <intersection_id> <image_path> [expected_type] [base64_config] [annotated_path]")
        sys.exit(1)
    
    intersection_id = sys.argv[1]
    image_path = sys.argv[2]
    expected_type = sys.argv[3] if len(sys.argv) > 3 else "4-arm"
    config_str = sys.argv[4] if len(sys.argv) > 4 else "none"
    annotated_path = sys.argv[5] if len(sys.argv) > 5 else None
    
    print(extract_traffic(intersection_id, image_path, expected_type, config_str, annotated_path))
