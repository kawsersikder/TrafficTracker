"""Reconstruct segment-level series from the flattened benjiao Manila logs.

The MMDA feed listed N named segments per road+direction; the flattened CSV
dropped the segment names but preserved row order.  Verified on the raw data:
the per-snapshot row count is constant per (road, direction) — EDSA=39, C5=19,
QUEZON_AVE=26, ... — so segment identity is recoverable positionally.  We keep
only snapshots whose row count equals the modal count (drops logging glitches)
and assign each row a stable segment index s00..sNN.

Output: experiments/out/manila_segments.parquet   (canonical long format)
        experiments/out/manila_series_meta.csv    (per-series inventory)
        experiments/out/manila_reconstruction_report.json

Usage:
    python experiments/manila/reconstruct_segments.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments"))

from common.data import CLS_TO_CI, MANILA_STATUS_TO_CLS  # noqa: E402

RAW = REPO / "dataset" / "manila_benjiao-traffic-logs" / "raw" / "manila-traffic.csv"
OUT_DIR = REPO / "experiments" / "out"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"reading {RAW} ...")
    df = pd.read_csv(RAW, index_col=0)
    report: dict = {"rows_raw": len(df)}

    # normalize the mojibake road name (ESPA�A -> ESPANA) and any stray chars
    df["road"] = df["road"].str.replace(r"[^A-Z0-9_.]", "N", regex=True)

    # timestamps are all +08:00 — convert to tz-naive Manila local time
    ts = pd.to_datetime(df["date_extracted"], format="mixed", utc=True)
    df["ts"] = ts.dt.tz_convert("Asia/Manila").dt.tz_localize(None)

    n_ni = int((df["status"] == "NI").sum())
    df = df[df["status"].isin(MANILA_STATUS_TO_CLS)].copy()
    report["rows_dropped_no_info_status"] = n_ni

    # preserve original file order — it encodes segment position within a snapshot
    df["orig_ord"] = np.arange(len(df))

    counts = df.groupby(["road", "direction", "ts"], sort=False).transform("size")
    modal = (
        df.assign(cnt=counts)
        .groupby(["road", "direction"])["cnt"]
        .transform(lambda s: s.mode().iat[0])
    )
    keep = counts == modal
    report["rows_dropped_incomplete_snapshots"] = int((~keep).sum())
    df = df[keep].copy()

    df = df.sort_values(["road", "direction", "ts", "orig_ord"], kind="stable")
    df["seg"] = df.groupby(["road", "direction", "ts"], sort=False).cumcount()
    df["series_id"] = (
        df["road"] + "|" + df["direction"] + "|s" + df["seg"].astype(str).str.zfill(2)
    )

    df["cls"] = df["status"].map(MANILA_STATUS_TO_CLS).astype(np.int8)
    df["y"] = CLS_TO_CI[df["cls"].to_numpy()]

    out = (
        df[["ts", "series_id", "y", "cls"]]
        .drop_duplicates(["series_id", "ts"])
        .sort_values(["series_id", "ts"], kind="stable")
        .reset_index(drop=True)
    )

    meta = (
        out.groupby("series_id")
        .agg(n_obs=("ts", "size"), first=("ts", "min"), last=("ts", "max"),
             mean_ci=("y", "mean"))
        .reset_index()
    )
    report.update(
        rows_kept=len(out),
        n_series=int(meta["series_id"].nunique()),
        date_range=[str(out["ts"].min()), str(out["ts"].max())],
        class_share={
            k: round(float((out["cls"] == v).mean()), 5)
            for k, v in MANILA_STATUS_TO_CLS.items()
        },
    )

    out.to_parquet(OUT_DIR / "manila_segments.parquet", index=False)
    meta.to_csv(OUT_DIR / "manila_series_meta.csv", index=False)
    (OUT_DIR / "manila_reconstruction_report.json").write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print(f"\nwrote {OUT_DIR / 'manila_segments.parquet'}")


if __name__ == "__main__":
    main()
