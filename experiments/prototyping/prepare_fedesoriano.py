"""Prepare the fedesoriano 4-junction Kaggle CSV as a tiny smoke-test set.

Strictly for debugging the training loop (city unknown — never cite it as a
compared city).  Hourly vehicle counts are scaled per junction to [0, 1] by the
99th percentile to act as a stand-in Congestion Index; cls = -1 (regression).

Output: experiments/out/fedesoriano.parquet

Usage:
    python experiments/prototyping/prepare_fedesoriano.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "dataset" / "prototyping_fedesoriano-junctions" / "raw" / "traffic.csv"
OUT = REPO / "experiments" / "out" / "fedesoriano.parquet"


def main() -> None:
    df = pd.read_csv(RAW)
    df["ts"] = pd.to_datetime(df["DateTime"])
    df["series_id"] = "junction_" + df["Junction"].astype(str)
    q99 = df.groupby("series_id")["Vehicles"].transform(lambda s: s.quantile(0.99))
    df["y"] = np.clip(df["Vehicles"] / q99, 0, 1).astype(np.float32)
    df["cls"] = np.int8(-1)

    out = df[["ts", "series_id", "y", "cls"]].sort_values(["series_id", "ts"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"wrote {OUT}: {len(out)} rows, {out.series_id.nunique()} series, "
          f"{out.ts.min()} -> {out.ts.max()}")


if __name__ == "__main__":
    main()
