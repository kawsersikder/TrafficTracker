"""Convert the DhakaAI Pascal-VOC annotations to YOLO layout for ultralytics.

Reads dataset/dhaka_dhakaai/raw/train/Final Train Dataset (JPG + VOC XML in one
folder), writes experiments/out/dhakaai_yolo/{images,labels}/{train,val} plus
dataset.yaml, with a deterministic hash-based 90/10 split.

Usage:
    python experiments/dhaka_vision/voc_to_yolo.py
Then train (see TRAINING.md):
    yolo detect train data=experiments/out/dhakaai_yolo/dataset.yaml model=yolov8n.pt ^
        epochs=50 imgsz=640 batch=16 workers=2 device=0
"""
from __future__ import annotations

import hashlib
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "dataset" / "dhaka_dhakaai" / "raw" / "train" / "Final Train Dataset"
OUT = REPO / "experiments" / "out" / "dhakaai_yolo"
VAL_FRAC = 0.10


def parse_voc(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w, h = int(size.findtext("width")), int(size.findtext("height"))
    boxes = []
    for obj in root.findall("object"):
        name = obj.findtext("name").strip().lower()
        bb = obj.find("bndbox")
        x1, y1 = float(bb.findtext("xmin")), float(bb.findtext("ymin"))
        x2, y2 = float(bb.findtext("xmax")), float(bb.findtext("ymax"))
        x1, x2 = max(0.0, min(x1, w)), max(0.0, min(x2, w))
        y1, y2 = max(0.0, min(y1, h)), max(0.0, min(y2, h))
        if x2 > x1 and y2 > y1:
            boxes.append((name, x1, y1, x2, y2))
    return w, h, boxes


def main() -> None:
    xmls = sorted(SRC.glob("*.xml"))
    if not xmls:
        raise SystemExit(f"no XML annotations under {SRC}")

    # first pass: fixed class vocabulary
    class_counts: Counter[str] = Counter()
    parsed = {}
    for x in xmls:
        try:
            parsed[x] = parse_voc(x)
            class_counts.update(name for name, *_ in parsed[x][2])
        except Exception as e:  # noqa: BLE001 — a broken XML shouldn't kill the export
            print(f"skip {x.name}: {e}")
    classes = sorted(class_counts)
    cls_idx = {c: i for i, c in enumerate(classes)}
    print(f"{len(parsed)} annotated images, {len(classes)} classes")
    for c, n in class_counts.most_common():
        print(f"  {c:20s} {n}")

    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    n_written = 0
    for xml_path, (w, h, boxes) in parsed.items():
        img = None
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
            cand = xml_path.with_suffix(ext)
            if cand.exists():
                img = cand
                break
        if img is None or not boxes:
            continue
        digest = hashlib.md5(xml_path.stem.encode()).hexdigest()
        split = "val" if int(digest, 16) % 100 < VAL_FRAC * 100 else "train"
        lines = []
        for name, x1, y1, x2, y2 in boxes:
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            lines.append(f"{cls_idx[name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (OUT / "labels" / split / f"{xml_path.stem}.txt").write_text("\n".join(lines))
        dst = OUT / "images" / split / img.name
        if not dst.exists():
            shutil.copy2(img, dst)
        n_written += 1

    yaml_lines = [f"path: {OUT.as_posix()}", "train: images/train", "val: images/val",
                  "names:"] + [f"  {i}: {c}" for i, c in enumerate(classes)]
    (OUT / "dataset.yaml").write_text("\n".join(yaml_lines))
    print(f"\nwrote {n_written} images -> {OUT}\\dataset.yaml")


if __name__ == "__main__":
    main()
