"""Convert TFP-BD (VOC-in-zips) into a YOLO dataset, and build a shared-class
DhakaAI test set so the two Dhaka sources can be cross-evaluated.

TFP-BD frames come from continuous video, so consecutive frames are highly
redundant. We subsample every Nth frame (default 4). Splitting is by LOCATION,
not randomly: training on Locations 1-3 and validating on Location 4 measures
generalisation to an unseen intersection rather than to an adjacent video frame.

Usage:
    python experiments/dhaka_vision/tfpbd_to_yolo.py --stride 4
    python experiments/dhaka_vision/tfpbd_to_yolo.py --dhakaai-remap   # shared-class DhakaAI copy
"""
from __future__ import annotations

import argparse
import re
import shutil
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "dataset" / "dhaka_tfp-bd" / "raw" / "Annotated Images"
OUT = REPO / "experiments" / "out" / "tfpbd_yolo"
DHAKAAI = REPO / "experiments" / "out" / "dhakaai_yolo"
SHARED_OUT = REPO / "experiments" / "out" / "dhakaai_shared_yolo"

# Shared vocabulary between the two Dhaka sources. Anything outside it is
# dropped from BOTH datasets so cross-evaluation is a like-for-like comparison.
SHARED = ["rickshaw", "cng", "car", "bus", "bike", "person"]
IDX = {c: i for i, c in enumerate(SHARED)}

TFPBD_MAP = {"rickshaw": "rickshaw", "cng": "cng", "car": "car", "bus": "bus",
             "bike": "bike", "people": "person", "mini-truck": None,
             "cycle": None}
DHAKAAI_MAP = {
    "rickshaw": "rickshaw", "three wheelers (cng)": "cng", "auto rickshaw": "cng",
    "car": "car", "taxi": "car", "suv": "car", "minivan": "car", "van": "car",
    "bus": "bus", "minibus": "bus",
    "motorbike": "bike", "scooter": "bike",
}


def convert_tfpbd(stride: int) -> None:
    zips = sorted(SRC.rglob("*.zip"))
    if not zips:
        raise SystemExit(f"no zips under {SRC}")
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            (OUT / sub / split).mkdir(parents=True, exist_ok=True)

    counts: Counter = Counter()
    kept = dropped = 0
    for zp in zips:
        location = zp.relative_to(SRC).parts[0]
        # hold out Location 4 entirely -> unseen-intersection validation
        split = "val" if location.startswith("Location 4") else "train"
        m = re.search(r"LOC\d+-(\d{4})-(\d{4})", zp.stem, re.I)
        window = m.group(1) if m else "0000"
        zf = zipfile.ZipFile(zp)
        xmls = sorted(n for n in zf.namelist() if n.lower().endswith(".xml"))
        for i, n in enumerate(xmls):
            if i % stride:
                continue
            try:
                root = ET.fromstring(zf.read(n))
            except ET.ParseError:
                continue
            W = float(root.findtext("size/width") or 0)
            H = float(root.findtext("size/height") or 0)
            if not W or not H:
                continue
            lines = []
            for o in root.iter("object"):
                raw = (o.findtext("name") or "").strip().lower()
                cls = TFPBD_MAP.get(raw)
                if cls is None:
                    continue
                b = o.find("bndbox")
                x1, x2 = float(b.findtext("xmin")), float(b.findtext("xmax"))
                y1, y2 = float(b.findtext("ymin")), float(b.findtext("ymax"))
                x1, x2 = sorted((max(0, x1), min(W, x2)))
                y1, y2 = sorted((max(0, y1), min(H, y2)))
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                lines.append(f"{IDX[cls]} {((x1+x2)/2)/W:.6f} {((y1+y2)/2)/H:.6f} "
                             f"{(x2-x1)/W:.6f} {(y2-y1)/H:.6f}")
                counts[cls] += 1
            if not lines:
                dropped += 1
                continue
            img = n[:-4] + ".jpg"
            if img not in zf.namelist():
                cand = [c for c in zf.namelist()
                        if c.rsplit(".", 1)[0] == n[:-4] and not c.endswith(".xml")]
                if not cand:
                    continue
                img = cand[0]
            stem = f"{location[:10].replace(' ','')}_{window}_{Path(n).stem[:40]}"
            (OUT / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))
            with zf.open(img) as fsrc, open(OUT / "images" / split / f"{stem}.jpg", "wb") as fdst:
                shutil.copyfileobj(fsrc, fdst)
            kept += 1
        print(f"  {zp.relative_to(SRC)} -> {split}")

    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(SHARED))
    (OUT / "dataset.yaml").write_text(
        f"path: {OUT.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n")
    ntr = len(list((OUT / "images" / "train").glob("*.jpg")))
    nva = len(list((OUT / "images" / "val").glob("*.jpg")))
    print(f"\nTFP-BD -> YOLO: {kept} images kept ({ntr} train / {nva} val), "
          f"{dropped} empty dropped, stride={stride}")
    print("objects per class:", dict(counts))


def remap_dhakaai() -> None:
    """Re-label DhakaAI onto the shared vocabulary so it can be cross-evaluated."""
    yml = (DHAKAAI / "dataset.yaml").read_text()
    names = {int(a): b.strip() for a, b in re.findall(r"^\s*(\d+):\s*(.+)$", yml, re.M)}
    counts: Counter = Counter()
    n_img = 0
    for split in ("train", "val"):
        (SHARED_OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (SHARED_OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
        for lb in sorted((DHAKAAI / "labels" / split).glob("*.txt")):
            out = []
            for line in lb.read_text().splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                cls = DHAKAAI_MAP.get(names[int(parts[0])].lower())
                if cls is None:
                    continue
                out.append(" ".join([str(IDX[cls])] + parts[1:]))
                counts[cls] += 1
            if not out:
                continue
            img = next((p for p in (DHAKAAI / "images" / split).glob(lb.stem + ".*")), None)
            if img is None:
                continue
            (SHARED_OUT / "labels" / split / lb.name).write_text("\n".join(out))
            shutil.copy2(img, SHARED_OUT / "images" / split / img.name)
            n_img += 1
    names_y = "\n".join(f"  {i}: {c}" for i, c in enumerate(SHARED))
    (SHARED_OUT / "dataset.yaml").write_text(
        f"path: {SHARED_OUT.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n{names_y}\n")
    print(f"DhakaAI shared-class copy: {n_img} images, objects {dict(counts)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--dhakaai-remap", action="store_true")
    a = ap.parse_args()
    if a.dhakaai_remap:
        remap_dhakaai()
    else:
        convert_tfpbd(a.stride)
