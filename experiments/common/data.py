"""Shared data utilities for the forecasting experiments.

Canonical long format (parquet), produced by each dataset's prepare script:
    ts        datetime64[ns]  observation timestamp (tz-naive, city-local time)
    series_id str             unique series key, e.g. "EDSA|NB|s07"
    y         float32         Congestion Index (CI) in [0, 1]
    cls       int8            ordinal congestion class 0..4, or -1 when the
                              source has no discrete classes (regression task)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Manila MMDA ordinal states (verified against the raw CSV: L/ML/M/MH/H + rare NI)
MANILA_STATUS_TO_CLS = {"L": 0, "ML": 1, "M": 2, "MH": 3, "H": 4}
CLS_TO_CI = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float32)
# CI -> class bin edges (inverse of CLS_TO_CI midpoints)
CI_CLASS_BINS = np.array([0.2, 0.4, 0.6, 0.8])

# "Congestion onset" = jump into MH-or-worse (CI >= 0.6)
ONSET_CLS_THRESHOLD = 3
ONSET_CI_THRESHOLD = 0.6

N_FEATURES = 5  # y + sin/cos time-of-day + sin/cos day-of-week


def ci_to_cls(ci: np.ndarray) -> np.ndarray:
    return np.digitize(ci, CI_CLASS_BINS).astype(np.int8)


def load_long_parquet(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    missing = {"ts", "series_id", "y"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}")
    if "cls" not in df.columns:
        df["cls"] = np.int8(-1)
    df["ts"] = pd.to_datetime(df["ts"])
    df["y"] = df["y"].astype(np.float32)
    df["cls"] = df["cls"].astype(np.int8)
    return df.sort_values(["series_id", "ts"], kind="stable").reset_index(drop=True)


def _time_features(ts: pd.Series) -> np.ndarray:
    tod = (ts.dt.hour * 3600 + ts.dt.minute * 60 + ts.dt.second).to_numpy() / 86400.0
    dow = ts.dt.dayofweek.to_numpy() / 7.0
    return np.stack(
        [
            np.sin(2 * np.pi * tod),
            np.cos(2 * np.pi * tod),
            np.sin(2 * np.pi * dow),
            np.cos(2 * np.pi * dow),
        ],
        axis=1,
    ).astype(np.float32)


def build_windows(
    df: pd.DataFrame,
    history: int = 24,
    horizon: int = 1,
    gap_factor: float = 1.9,
) -> dict:
    """Slice each series into (history -> horizon-step-ahead) supervised windows.

    A window is kept only if every sampling interval it spans is at most
    ``gap_factor`` x the series' median interval, so windows never bridge
    logging outages.

    Returns dict of arrays:
        X          (N, history, 5) float32
        y          (N,)  float32   CI at target step
        cls        (N,)  int8      class at target step (-1 for regression sets)
        prev_cls   (N,)  int8      ground-truth class one step before target
        prev_y     (N,)  float32   ground-truth CI one step before target
        target_ts  (N,)  datetime64
        series     (N,)  int32     index into ``series_ids``
        series_ids list[str]
    """
    xs, ys, cs, pcs, pys, tss, sids = [], [], [], [], [], [], []
    series_ids: list[str] = []

    groups = df.groupby("series_id", sort=True)
    for sid, g in tqdm(groups, total=groups.ngroups, desc="building windows",
                       unit="series", leave=False, dynamic_ncols=True):
        n = len(g)
        span = history + horizon
        if n < span + 1:
            continue
        ts = g["ts"]
        feats = np.concatenate(
            [g["y"].to_numpy(np.float32)[:, None], _time_features(ts)], axis=1
        )
        diffs = ts.diff().dt.total_seconds().to_numpy()[1:]
        step = np.nanmedian(diffs)
        bad = np.concatenate([[0], (diffs > gap_factor * step).astype(np.int64)])
        bad_cum = np.cumsum(bad)

        starts = np.arange(0, n - span + 1)
        tgt = starts + span - 1
        # window valid iff no oversized gap between index `start` and `tgt`
        valid = (bad_cum[tgt] - bad_cum[starts]) == 0
        starts, tgt = starts[valid], tgt[valid]
        if len(starts) == 0:
            continue

        idx = starts[:, None] + np.arange(history)[None, :]
        xs.append(feats[idx])
        ys.append(g["y"].to_numpy(np.float32)[tgt])
        cs.append(g["cls"].to_numpy(np.int8)[tgt])
        pcs.append(g["cls"].to_numpy(np.int8)[tgt - 1])
        pys.append(g["y"].to_numpy(np.float32)[tgt - 1])
        tss.append(ts.to_numpy()[tgt])
        sids.append(np.full(len(tgt), len(series_ids), dtype=np.int32))
        series_ids.append(sid)

    if not xs:
        raise ValueError("no valid windows produced — check history/horizon vs series length")

    return {
        "X": np.concatenate(xs),
        "y": np.concatenate(ys),
        "cls": np.concatenate(cs),
        "prev_cls": np.concatenate(pcs),
        "prev_y": np.concatenate(pys),
        "target_ts": np.concatenate(tss),
        "series": np.concatenate(sids),
        "series_ids": series_ids,
    }


def time_split(target_ts: np.ndarray, train=0.70, val=0.15) -> dict[str, np.ndarray]:
    """Chronological split on the target timestamp (no leakage across the cut)."""
    q_train = np.quantile(target_ts.astype("int64"), train)
    q_val = np.quantile(target_ts.astype("int64"), train + val)
    t = target_ts.astype("int64")
    return {
        "train": t <= q_train,
        "val": (t > q_train) & (t <= q_val),
        "test": t > q_val,
    }
