"""Extract the per-class detector table from the Ultralytics validation output.

Ultralytics prints a per-class block at the end of training. We parse it from the
orchestrator logs (UTF-16 via Tee-Object) rather than re-running validation, so
this costs no GPU time.

Writes experiments/out/dhaka/yolo_per_class.csv  (+ a markdown table for the paper)
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "experiments" / "out" / "dhaka"
LOGS = {"YOLOv8s": REPO / "logs" / "11_yolov8s_finish.log",
        "YOLOv8m": REPO / "logs" / "12_yolov8m.log"}

# "  three wheelers (cng)   107   292   0.843   0.784   0.857   0.595"
ROW = re.compile(r"^\s{2,}([A-Za-z][A-Za-z ()\-]*?)\s{2,}"
                 r"(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$")


def read(p: Path) -> str:
    raw = p.read_bytes()
    for enc in ("utf-16", "utf-16-le", "utf-8"):
        try:
            t = raw.decode(enc)
            if t.count("\x00") < len(t) // 4:
                return t.replace("\r", "")
        except (UnicodeDecodeError, UnicodeError):
            continue
    return raw.decode("utf-8", "ignore").replace("\r", "")


def parse(p: Path) -> pd.DataFrame:
    rows = []
    for line in read(p).split("\n"):
        line = re.sub(r"\x1b\[[0-9;]*m", "", line)
        m = ROW.match(line)
        if not m:
            continue
        name = m.group(1).strip()
        if name.lower() in {"all", "class"}:
            continue
        rows.append({"cls": name, "images": int(m.group(2)), "instances": int(m.group(3)),
                     "P": float(m.group(4)), "R": float(m.group(5)),
                     "mAP50": float(m.group(6)), "mAP50_95": float(m.group(7))})
    df = pd.DataFrame(rows)
    # a class can appear once per validation pass; keep the last (final epoch)
    return df.drop_duplicates("cls", keep="last") if len(df) else df


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames = {}
    for model, p in LOGS.items():
        if not p.exists():
            print(f"missing log: {p}")
            continue
        d = parse(p)
        if d.empty:
            print(f"no per-class rows parsed from {p.name}")
            continue
        frames[model] = d.set_index("cls")
        print(f"{model}: parsed {len(d)} classes")

    if not frames:
        raise SystemExit("nothing parsed")

    base = next(iter(frames.values()))[["instances"]]
    tbl = base.copy()
    for model, d in frames.items():
        tbl[f"{model} mAP50"] = d["mAP50"]
        tbl[f"{model} mAP50-95"] = d["mAP50_95"]
    tbl = tbl.sort_values("instances", ascending=False)
    tbl.to_csv(OUT / "yolo_per_class.csv")

    md = tbl.round(3).to_markdown()
    (OUT / "yolo_per_class.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"\nwrote {OUT/'yolo_per_class.csv'} and .md")


if __name__ == "__main__":
    main()
