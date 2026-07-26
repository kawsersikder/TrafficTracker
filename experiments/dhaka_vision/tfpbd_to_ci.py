"""Derive a congestion index for Dhaka from TFP-BD imagery.

Dhaka has no public congestion time series -- every open Dhaka dataset is
imagery. But TFP-BD's annotated frames are *consecutive video frames* (609 per
zip, gap-free, ~5.9 s apart), so a per-frame occupancy signal is itself a time
series.

We define, per frame:

    visual occupancy = (sum of vehicle bounding-box area) / (frame area)

which is the camera analogue of loop-detector occupancy -- the quantity that
already defines CI for Bangkok Sathorn (CI = occupancy/100). Because camera
perspective inflates near-field vehicles, absolute occupancy is not comparable
across locations, so we robust-scale per series (q05..q95) exactly as the
`volume` mode of prepare_sathorn.py does for Bangkok CCTV lane counts.

Pedestrians are excluded from occupancy (they are not roadway congestion in the
same sense) but counted separately, since Dhaka's pedestrian share is a
distinguishing feature of the city.

Output: experiments/out/dhaka_ci.parquet in the canonical long format
(ts, series_id, y, cls), one series per (location, lane, time window).
"""
from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "dataset" / "dhaka_tfp-bd" / "raw" / "Annotated Images"
OUT = REPO / "experiments" / "out"

PEDESTRIAN = {"people", "person", "pedestrian"}
FRAME_SECONDS = 3600.0 / 609.0  # one zip covers a one-hour window


def build(frame_seconds: float) -> pd.DataFrame:
    zips = sorted(SRC.rglob("*.zip"))
    if not zips:
        raise SystemExit(f"no zips under {SRC}")
    rows = []
    for zp in zips:
        rel = zp.relative_to(SRC)
        location = rel.parts[0].split("(")[-1].rstrip(")").strip() or rel.parts[0]
        lane = "DL" if "double" in rel.parts[1].lower() else "SL"
        m = re.search(r"LOC(\d+)-(\d{4})-(\d{4})", zp.stem, re.I)
        if not m:
            continue
        start = m.group(2)
        base = pd.Timestamp("2024-01-01") + pd.Timedelta(hours=int(start[:2]), minutes=int(start[2:]))
        sid = f"dhaka|{location}|{lane}|{start}"

        zf = zipfile.ZipFile(zp)
        per_frame = {}
        for n in zf.namelist():
            if not n.lower().endswith(".xml"):
                continue
            fm = re.search(r"frame(\d+)", n)
            if not fm:
                continue
            try:
                root = ET.fromstring(zf.read(n))
            except ET.ParseError:
                continue
            W = float(root.findtext("size/width") or 0)
            H = float(root.findtext("size/height") or 0)
            if not W or not H:
                continue
            veh_area = 0.0
            n_veh = n_ped = 0
            for o in root.iter("object"):
                nm = (o.findtext("name") or "").strip().lower()
                b = o.find("bndbox")
                if b is None:
                    continue
                w = float(b.findtext("xmax")) - float(b.findtext("xmin"))
                h = float(b.findtext("ymax")) - float(b.findtext("ymin"))
                if w <= 0 or h <= 0:
                    continue
                if nm in PEDESTRIAN:
                    n_ped += 1
                else:
                    veh_area += w * h
                    n_veh += 1
            per_frame[int(fm.group(1))] = (veh_area / (W * H), n_veh, n_ped)

        for f in sorted(per_frame):
            occ, nv, np_ = per_frame[f]
            rows.append({"ts": base + pd.Timedelta(seconds=f * frame_seconds),
                         "series_id": sid, "occ": occ,
                         "n_veh": nv, "n_ped": np_})
        print(f"  {rel.parent}/{zp.stem}: {len(per_frame)} frames")

    df = pd.DataFrame(rows).sort_values(["series_id", "ts"]).reset_index(drop=True)

    # robust per-series scaling -> CI in [0,1]; perspective makes raw occupancy
    # incomparable across cameras, so each series is scaled to its own q05..q95
    lo = df.groupby("series_id")["occ"].transform(lambda s: s.quantile(0.05))
    hi = df.groupby("series_id")["occ"].transform(lambda s: s.quantile(0.95))
    df["y"] = np.clip((df["occ"] - lo) / (hi - lo).replace(0, np.nan), 0, 1).fillna(0).astype(np.float32)
    df["cls"] = np.int8(-1)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame-seconds", type=float, default=FRAME_SECONDS)
    ap.add_argument("--out", default=str(OUT / "dhaka_ci.parquet"))
    a = ap.parse_args()

    df = build(a.frame_seconds)
    OUT.mkdir(parents=True, exist_ok=True)
    df[["ts", "series_id", "y", "cls"]].to_parquet(a.out, index=False)
    df.to_parquet(OUT / "dhaka_ci_full.parquet", index=False)

    print(f"\nwrote {a.out}: {len(df)} rows, {df.series_id.nunique()} series")
    print(f"time span per series: {df.groupby('series_id').size().median():.0f} steps "
          f"at {a.frame_seconds:.1f}s")
    print(f"CI mean {df.y.mean():.3f}  raw occupancy mean {df.occ.mean():.3f}")
    print(f"vehicles/frame mean {df.n_veh.mean():.1f}  pedestrians/frame mean {df.n_ped.mean():.1f}")


if __name__ == "__main__":
    main()
