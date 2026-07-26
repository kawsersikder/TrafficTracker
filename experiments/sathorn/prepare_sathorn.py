"""Prepare the Bangkok Sathorn sensor archives into the canonical long format.

The figshare archives are .rar — extract them first (needs 7-Zip, see
TRAINING.md), e.g. into dataset/bangkok_sathorn-intersection/extracted/.

Because the internal CSV schemas differ per modality (CCTV volumes, loop-coil
occupancy, thermal), this script has two modes:

1) Inspect what was extracted (prints every CSV with its columns):
     python experiments/sathorn/prepare_sathorn.py --inspect

2) Build the parquet once you know the columns:
     python experiments/sathorn/prepare_sathorn.py ^
       --csv "dataset/bangkok_sathorn-intersection/extracted/loopcoil/*.csv" ^
       --time-col timestamp --value-col occupancy --series-col detector_id ^
       --mode occupancy --resample 15min --out experiments/out/sathorn_loopcoil.parquet

   CCTV lane volumes (date lives in the filename, lanes are separate columns):
     python experiments/sathorn/prepare_sathorn.py ^
       --csv "dataset/.../cctv/cctv-camera/Link*/*_volume_*.csv" ^
       --time-col Time --date-from-filename --series-from-parent ^
       --value-col E1,E2,E3,S1,S2,S3,W1,W2,W3,W4,N1,N2,N3 ^
       --mode volume --out experiments/out/sathorn_cctv.parquet

Normalization to CI in [0, 1]:
    occupancy  CI = value / 100 (clipped)
    volume     CI = per-series robust scale (q05..q95) — volume is only a
               congestion *proxy*; state this as a limitation in the paper
    speed      CI = 1 - robust-scaled value (inverted)
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO / "dataset" / "bangkok_sathorn-intersection" / "extracted"


def inspect(root: Path) -> None:
    csvs = sorted(root.rglob("*.csv"))
    if not csvs:
        sys.exit(f"no CSVs under {root} — extract the .rar archives first (see TRAINING.md)")
    for p in csvs:
        try:
            head = pd.read_csv(p, nrows=3)
            print(f"\n{p.relative_to(root)}  ({p.stat().st_size/1e6:.1f} MB)")
            print("  columns:", list(head.columns))
            print(head.to_string(max_colwidth=25))
        except Exception as e:  # noqa: BLE001 — inventory must not die on one bad file
            print(f"\n{p}: unreadable ({e})")


def build(args: argparse.Namespace) -> None:
    files = sorted(glob.glob(args.csv, recursive=True))
    if not files:
        sys.exit(f"no files match {args.csv}")

    value_cols = [c.strip() for c in args.value_col.split(",") if c.strip()]
    multi_lane = len(value_cols) > 1
    frames = []
    skipped = 0

    for f in tqdm(files, desc="reading CSVs", unit="file", dynamic_ncols=True):
        try:
            df = pd.read_csv(f)
        except Exception:  # noqa: BLE001 — one corrupt day must not kill a 3-year build
            skipped += 1
            continue

        if args.date_from_filename:
            m = re.search(r"(\d{8})", Path(f).stem)
            if not m:
                skipped += 1
                continue
            df["ts"] = pd.to_datetime(
                m.group(1) + " " + df[args.time_col].astype(str),
                format="%Y%m%d %H:%M:%S", errors="coerce")
        elif args.date_col:
            df["ts"] = pd.to_datetime(df[args.date_col].astype(str) + " "
                                      + df[args.time_col].astype(str), format="mixed")
        else:
            df["ts"] = pd.to_datetime(df[args.time_col], format="mixed")

        present = [c for c in value_cols if c in df.columns]
        if not present or df["ts"].isna().all():
            skipped += 1
            continue

        if args.series_from_parent:
            sub = df[["ts"] + present].copy()
            sub["__base"] = Path(f).parent.name
        elif args.series_col:
            sub = df[["ts", args.series_col] + present].copy()
            sub["__base"] = sub[args.series_col].astype(str)
            sub = sub.drop(columns=[args.series_col])
        else:
            sub = df[["ts"] + present].copy()
            sub["__base"] = Path(f).stem

        # Melt lane columns (E1/E2/E3, W1..W4, ...) into one series each.
        long = sub.melt(id_vars=["ts", "__base"], value_vars=present,
                        var_name="__lane", value_name="val")
        if multi_lane:
            long["series_id"] = "sathorn|" + long["__base"] + "|" + long["__lane"]
        else:
            long["series_id"] = "sathorn|" + long["__base"]
        long["val"] = pd.to_numeric(long["val"], errors="coerce")
        long = long[["ts", "series_id", "val"]].dropna()

        # Resample inside the loop when each file is one day: a 15-min bin can
        # never span two files, so this is exactly equivalent to resampling
        # after the concat -- but it keeps peak memory ~200x lower. Concatenating
        # 4356 raw CCTV days first is ~240M rows and thrashes a 24GB machine.
        if args.per_file_resample and len(long):
            long = (long.set_index("ts")
                        .groupby("series_id")["val"]
                        .resample(args.resample).mean()
                        .dropna().reset_index())
        frames.append(long)

    if not frames:
        sys.exit("every input file was skipped — check --time-col / --value-col")
    if skipped:
        print(f"note: skipped {skipped} unusable file(s) of {len(files)}")

    df = pd.concat(frames, ignore_index=True).dropna()

    df = (
        df.set_index("ts")
        .groupby("series_id")["val"]
        .resample(args.resample).mean()
        .dropna()
        .reset_index()
    )

    if args.mode == "occupancy":
        df["y"] = np.clip(df["val"] / 100.0, 0, 1)
    else:
        lo = df.groupby("series_id")["val"].transform(lambda s: s.quantile(0.05))
        hi = df.groupby("series_id")["val"].transform(lambda s: s.quantile(0.95))
        scaled = np.clip((df["val"] - lo) / (hi - lo).replace(0, np.nan), 0, 1).fillna(0)
        df["y"] = 1 - scaled if args.mode == "speed" else scaled

    df["y"] = df["y"].astype(np.float32)
    df["cls"] = np.int8(-1)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df[["ts", "series_id", "y", "cls"]].to_parquet(out, index=False)
    print(f"wrote {out}: {len(df)} rows, {df.series_id.nunique()} series, "
          f"{df.ts.min()} -> {df.ts.max()}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--csv", help="glob of source CSVs")
    ap.add_argument("--time-col")
    ap.add_argument("--date-col", default=None,
                    help="separate date column to combine with --time-col")
    ap.add_argument("--date-from-filename", action="store_true",
                    help="take the date from the first YYYYMMDD in each filename "
                         "(CCTV layout: 20160913_volume_l1.csv holds only HH:MM:SS)")
    ap.add_argument("--value-col",
                    help="value column, or comma-separated lane columns "
                         "(e.g. E1,E2,E3) which each become their own series")
    ap.add_argument("--series-col", default=None)
    ap.add_argument("--series-from-parent", action="store_true",
                    help="use each CSV's parent folder name as the series id "
                         "(Sathorn layout: Link1/, Link3/, ...)")
    ap.add_argument("--mode", choices=["occupancy", "volume", "speed"], default="volume")
    ap.add_argument("--resample", default="15min")
    ap.add_argument("--per-file-resample", action="store_true",
                    help="resample each file before concatenating (safe and much "
                         "cheaper when one file == one day, as in the CCTV archive)")
    ap.add_argument("--out", default=str(REPO / "experiments" / "out" / "sathorn.parquet"))
    args = ap.parse_args()

    if args.inspect:
        inspect(Path(args.root))
    else:
        if not (args.csv and args.time_col and args.value_col):
            ap.error("--csv, --time-col and --value-col are required to build")
        build(args)


if __name__ == "__main__":
    main()
