"""Baselines on a canonical long-format parquet, evaluated on the same
chronological test split as the deep models.

    persistence      predict the last observed value
    seasonal-naive   predict the value one week earlier (same series)
    hist-avg         per-series x day-of-week x time-slot mean (train only)
    xgboost          gradient boosting on lag + calendar features (CPU)
    arima            optional SARIMAX on a sample of series (slow; --arima N)

Example:
    python experiments/baselines.py --data experiments/out/manila_segments.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.data import build_windows, ci_to_cls, load_long_parquet, time_split  # noqa: E402
from common.metrics import classification_report, regression_report  # noqa: E402


def evaluate(name, task, pred_ci, W, ti) -> dict:
    if task == "cls":
        rep = classification_report(W["cls"][ti].astype(np.int64),
                                    ci_to_cls(np.clip(pred_ci, 0, 1)).astype(np.int64),
                                    W["prev_cls"][ti])
    else:
        rep = regression_report(W["y"][ti], pred_ci, W["prev_y"][ti])
    print(f"\n== {name} ==\n{json.dumps(rep, indent=2)}")
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--task", choices=["cls", "reg"], default=None)
    ap.add_argument("--history", type=int, default=24)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--arima", type=int, default=0,
                    help="run SARIMAX on N sampled series (slow, ~30-60s each)")
    args = ap.parse_args()

    t_start = time.time()
    print(f"loading {args.data} ...")
    df = load_long_parquet(args.data)
    task = args.task or ("cls" if (df["cls"] >= 0).all() else "reg")
    W = build_windows(df, history=args.history, horizon=args.horizon)
    split = time_split(W["target_ts"])
    ti = np.where(split["test"])[0]
    print(f"task: {task} | {len(W['series_ids'])} series | "
          f"train {split['train'].sum():,} / test {len(ti):,} windows | "
          f"prep took {time.time() - t_start:.0f}s")
    results: dict[str, dict] = {}

    # --- persistence: last observed CI in the window ---
    print("\n[1/4] persistence ...")
    results["persistence"] = evaluate("persistence", task, W["X"][ti, -1, 0], W, ti)

    # --- seasonal naive: same time-slot one week before (falls back to persistence) ---
    print("\n[2/4] seasonal-naive (1 week lookback) ...")
    step_s = float(np.median(np.diff(np.sort(df[df.series_id == df.series_id.iloc[0]]
                                             ["ts"].to_numpy())).astype("timedelta64[s]")
                             .astype(np.int64)))
    lookup = {(s, t): y for s, t, y in zip(W["series"], W["target_ts"].astype("int64"),
                                           W["y"])}
    week_ns = 7 * 24 * 3600 * 10**9
    sn = np.array([lookup.get((W["series"][i], int(W["target_ts"][i].astype("int64"))
                               - week_ns), W["X"][i, -1, 0])
                   for i in tqdm(ti, desc="seasonal-naive", unit="win",
                                 leave=False, dynamic_ncols=True)],
                  dtype=np.float32)
    results["seasonal_naive"] = evaluate("seasonal-naive (1 week)", task, sn, W, ti)

    # --- historical average per (series, dow, slot), fit on train targets only ---
    print("\n[3/4] historical average ...")
    tsr = pd.DatetimeIndex(W["target_ts"])
    slot = (tsr.hour * 3600 + tsr.minute * 60) // max(int(step_s), 1)
    key = pd.DataFrame({"series": W["series"], "dow": tsr.dayofweek, "slot": slot,
                        "y": W["y"]})
    ha = key[split["train"]].groupby(["series", "dow", "slot"])["y"].mean()
    merged = key.iloc[ti].merge(ha.rename("pred"), how="left",
                                left_on=["series", "dow", "slot"], right_index=True)
    pred_ha = merged["pred"].fillna(key[split["train"]]["y"].mean()).to_numpy(np.float32)
    results["hist_avg"] = evaluate("historical average", task, pred_ha, W, ti)

    # --- XGBoost on flattened lags + calendar features ---
    import xgboost as xgb

    feats = np.concatenate([W["X"][:, :, 0],            # CI lags
                            W["X"][:, -1, 1:]], axis=1)  # target-adjacent time feats
    vi = np.where(split["val"])[0]
    print(f"\n[4/4] xgboost: fitting 400 trees on {int(split['train'].sum()):,} windows "
          f"(all CPU threads; prints val loss every 50 trees) ...")
    t0 = time.time()
    if task == "cls":
        m = xgb.XGBClassifier(n_estimators=400, max_depth=8, learning_rate=0.1,
                              tree_method="hist", n_jobs=-1)
        m.fit(feats[split["train"]], W["cls"][split["train"]].astype(np.int64),
              eval_set=[(feats[vi], W["cls"][vi].astype(np.int64))], verbose=50)
        pred = m.predict(feats[ti]).astype(np.int64)
        rep = classification_report(W["cls"][ti].astype(np.int64), pred, W["prev_cls"][ti])
    else:
        m = xgb.XGBRegressor(n_estimators=400, max_depth=8, learning_rate=0.1,
                             tree_method="hist", n_jobs=-1)
        m.fit(feats[split["train"]], W["y"][split["train"]],
              eval_set=[(feats[vi], W["y"][vi])], verbose=50)
        rep = regression_report(W["y"][ti], m.predict(feats[ti]), W["prev_y"][ti])
    rep["fit_seconds"] = round(time.time() - t0, 1)
    print(f"\n== xgboost ==\n{json.dumps(rep, indent=2)}")
    results["xgboost"] = rep

    # --- optional ARIMA on a sample (reference numbers for the paper) ---
    if args.arima:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        rng = np.random.default_rng(0)
        sample = rng.choice(df["series_id"].unique(),
                            min(args.arima, df["series_id"].nunique()), replace=False)
        maes = []
        for sid in sample:
            s = df[df.series_id == sid].set_index("ts")["y"].asfreq(
                pd.Timedelta(seconds=step_s)).interpolate(limit=4).dropna()
            cut = int(len(s) * 0.85)
            try:
                fit = SARIMAX(s.iloc[:cut], order=(2, 0, 2)).fit(disp=False)
                fc = fit.forecast(len(s) - cut)
                maes.append(float(np.abs(fc.to_numpy() - s.iloc[cut:].to_numpy()).mean()))
            except Exception as e:  # noqa: BLE001 — a failed series shouldn't kill the sweep
                print(f"  arima failed on {sid}: {e}")
        results["arima_sample"] = {"series": len(maes),
                                   "mae_mean": float(np.mean(maes)) if maes else None}
        print(f"\n== arima ({len(maes)} series) == mae {results['arima_sample']['mae_mean']}")

    out = Path(__file__).parent / "runs" / f"{Path(args.data).stem}_baselines_h{args.horizon}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nall baselines done in {time.time() - t_start:.0f}s\nwrote {out}")


if __name__ == "__main__":
    main()
