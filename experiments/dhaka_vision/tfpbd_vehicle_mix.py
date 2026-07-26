"""Vehicle-mix characterisation of Dhaka from TFP-BD, and comparison to DhakaAI.

TFP-BD ships 40 annotated zips laid out as
    Annotated Images/Location N (Name)/{Single,Double} Lane/K-LOCN-HHMM-HHMM-XL.v1i.voc.zip
so location, lane type and time-of-day window are all recoverable from the path.
We read the VOC XML directly out of each zip -- the 1.6 GB of images is never
extracted, which keeps this cheap enough to run while the GPU is busy.

Outputs (experiments/out/dhaka/):
    tfpbd_objects.csv        one row per (location, lane, window, class) with counts
    tfpbd_mix_by_window.csv  class share per time-of-day window
    tfpbd_mix_by_location.csv
    dhaka_class_comparison.csv   TFP-BD vs DhakaAI class shares on a common vocabulary
"""
from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
TFPBD = REPO / "dataset" / "dhaka_tfp-bd" / "raw" / "Annotated Images"
OUT = REPO / "experiments" / "out" / "dhaka"

# TFP-BD and DhakaAI name the same vehicles differently; map both onto one
# vocabulary so the two independent Dhaka sources can be compared directly.
CANON = {
    "cng": "three-wheeler (CNG)", "three wheelers (cng)": "three-wheeler (CNG)",
    "auto rickshaw": "three-wheeler (CNG)",
    "rickshaw": "rickshaw",
    "bike": "motorbike", "motorbike": "motorbike", "motorcycle": "motorbike",
    "scooter": "motorbike",
    "car": "car", "private car": "car", "taxi": "car", "suv": "car",
    "minivan": "car", "van": "car", "pickup": "car", "policecar": "car",
    "bus": "bus", "minibus": "bus",
    "truck": "truck", "garbagevan": "truck", "covered van": "truck",
    "human hauler": "human hauler", "leguna": "human hauler",
    "bicycle": "bicycle", "cycle": "bicycle",
    "person": "pedestrian", "pedestrian": "pedestrian", "people": "pedestrian",
    "van (pickup)": "car", "wheelbarrow": "other", "ambulance": "other",
    "army vehicle": "other", "boat": "other", "train": "other",
}


def canon(name: str) -> str:
    return CANON.get(name.strip().lower(), name.strip().lower())


def parse_tfpbd() -> pd.DataFrame:
    zips = sorted(TFPBD.rglob("*.zip"))
    if not zips:
        sys.exit(f"no zips under {TFPBD} - extract the Mendeley download there first")
    rows = []
    for zp in zips:
        rel = zp.relative_to(TFPBD)
        location = rel.parts[0]
        lane = rel.parts[1]
        m = re.search(r"LOC\d+-(\d{4})-(\d{4})", zp.stem, re.I)
        window = f"{m.group(1)[:2]}:{m.group(1)[2:]}-{m.group(2)[:2]}:{m.group(2)[2:]}" if m else "unknown"
        counts: Counter = Counter()
        frames = 0
        try:
            zf = zipfile.ZipFile(zp)
        except Exception as e:  # noqa: BLE001
            print(f"  skip unreadable {zp.name}: {e}")
            continue
        for n in zf.namelist():
            if not n.lower().endswith(".xml"):
                continue
            frames += 1
            try:
                root = ET.fromstring(zf.read(n))
            except ET.ParseError:
                continue
            for obj in root.iter("object"):
                nm = obj.findtext("name") or ""
                if nm:
                    counts[canon(nm)] += 1
        for cls, c in counts.items():
            rows.append({"location": location, "lane": lane, "window": window,
                         "cls": cls, "count": c, "frames": frames})
        print(f"  {rel}: {frames} frames, {sum(counts.values())} objects")
    return pd.DataFrame(rows)


def dhakaai_counts() -> pd.DataFrame:
    """Class counts from the already-converted DhakaAI YOLO labels."""
    import yaml
    ds = REPO / "experiments" / "out" / "dhakaai_yolo"
    names = yaml.safe_load((ds / "dataset.yaml").read_text())["names"]
    counts: Counter = Counter()
    for lb in (ds / "labels").rglob("*.txt"):
        for line in lb.read_text().splitlines():
            if line.strip():
                counts[canon(names[int(line.split()[0])])] += 1
    return (pd.DataFrame({"cls": list(counts), "dhakaai": list(counts.values())})
            .sort_values("dhakaai", ascending=False))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("reading TFP-BD annotations from zips ...")
    df = parse_tfpbd()
    df.to_csv(OUT / "tfpbd_objects.csv", index=False)

    tot = df["count"].sum()
    print(f"\nTFP-BD: {tot} objects across {df.frames.groupby([df.location, df.lane, df.window]).first().sum()} frames")

    by_win = (df.groupby(["window", "cls"])["count"].sum().unstack(fill_value=0))
    by_win = by_win.div(by_win.sum(axis=1), axis=0).round(4)
    by_win.to_csv(OUT / "tfpbd_mix_by_window.csv")

    by_loc = (df.groupby(["location", "cls"])["count"].sum().unstack(fill_value=0))
    by_loc = by_loc.div(by_loc.sum(axis=1), axis=0).round(4)
    by_loc.to_csv(OUT / "tfpbd_mix_by_location.csv")

    overall = df.groupby("cls")["count"].sum().rename("tfpbd").reset_index()
    try:
        cmp = overall.merge(dhakaai_counts(), on="cls", how="outer").fillna(0)
    except Exception as e:  # noqa: BLE001
        print(f"(DhakaAI comparison skipped: {e})")
        cmp = overall
    for c in [x for x in ("tfpbd", "dhakaai") if x in cmp.columns]:
        cmp[c + "_share"] = (cmp[c] / cmp[c].sum()).round(4)
    cmp = cmp.sort_values("tfpbd", ascending=False)
    cmp.to_csv(OUT / "dhaka_class_comparison.csv", index=False)

    print("\n=== Dhaka vehicle mix: TFP-BD vs DhakaAI (share of annotated objects) ===")
    print(cmp.to_string(index=False))
    print("\n=== TFP-BD mix by time-of-day window ===")
    print(by_win.to_string())
    print(f"\nwrote 4 CSVs to {OUT}")


if __name__ == "__main__":
    main()
