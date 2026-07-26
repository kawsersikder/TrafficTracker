"""Aggregate every experiment artifact into paper-ready tables and figures.

Scans experiments/runs/ for:
    <data>_<model>_<task>_h<H>/metrics.json      deep model results
    <data>_baselines_h<H>.json                   baseline results
    transfer_<data>_<target>.csv                 k-day transfer curves

Writes to experiments/out/paper/:
    results_all.csv          every model x dataset x horizon, one row each
    results_table.md         markdown table grouped by dataset+horizon (paste into paper)
    fig_transfer_curve.png   headline figure: macro-F1 / onset-F1 vs k days
    fig_horizon_decay.png    metric decay with forecast horizon

Usage:
    python experiments/analysis/aggregate_results.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

RUNS = Path(__file__).resolve().parents[1] / "runs"
OUT = Path(__file__).resolve().parents[1] / "out" / "paper"


def collect() -> pd.DataFrame:
    rows = []
    for mj in sorted(RUNS.glob("*/metrics.json")):
        name = mj.parent.name
        m = re.match(r"(.+)_(gru|lstm|tcn)_(cls|reg)_h(\d+)$", name)
        if not m:
            print(f"skip unrecognized run dir: {name}")
            continue
        rows.append({"dataset": m.group(1), "model": m.group(2).upper(),
                     "task": m.group(3), "horizon": int(m.group(4)),
                     **json.loads(mj.read_text())})
    for bj in sorted(RUNS.glob("*_baselines_h*.json")):
        m = re.match(r"(.+)_baselines_h(\d+)$", bj.stem)
        for base_name, rep in json.loads(bj.read_text()).items():
            rows.append({"dataset": m.group(1), "model": base_name,
                         "task": "baseline", "horizon": int(m.group(2)), **rep})
    return pd.DataFrame(rows)


def results_markdown(df: pd.DataFrame) -> str:
    cols = ["model", "accuracy", "macro_f1", "f1_class_4", "onset_f1",
            "onset_precision", "onset_recall", "mae", "rmse"]
    lines = ["# Results tables (auto-generated — do not edit)\n"]
    for (ds, h), g in df.groupby(["dataset", "horizon"]):
        mins = {1: 30, 2: 60, 4: 120}.get(h, h * 30)
        lines.append(f"\n## {ds} — {mins} min ahead (h{h})\n")
        sub = g[[c for c in cols if c in g.columns]].copy()
        num = sub.select_dtypes("number").columns
        sub[num] = sub[num].round(4)
        lines.append(sub.sort_values("onset_f1", ascending=False)
                     .to_markdown(index=False))
    return "\n".join(lines)


def plot_transfer() -> None:
    for csv in sorted(RUNS.glob("transfer_*.csv")):
        t = pd.read_csv(csv)
        metric_cols = [("macro_f1", "Macro-F1"), ("onset_f1", "Congestion-onset F1")]
        metric_cols = [(c, l) for c, l in metric_cols if c in t.columns]
        fig, axes = plt.subplots(1, len(metric_cols), figsize=(11, 4))
        for ax, (col, label) in zip([axes] if len(metric_cols) == 1 else axes,
                                    metric_cols):
            zs = t[t["mode"] == "zero_shot"][col]
            if len(zs):
                ax.axhline(zs.iloc[0], ls="--", c="gray",
                           label=f"zero-shot ({zs.iloc[0]:.3f})")
            for mode, style in (("finetune", "o-"), ("scratch", "s-")):
                g = t[t["mode"] == mode].sort_values("k_days")
                ax.plot(g["k_days"], g[col], style, label=mode)
            ax.set_xlabel("days of local (target) training data, k")
            ax.set_ylabel(label)
            ax.set_xscale("symlog", linthresh=1)
            ax.set_xticks([1, 3, 7, 14, 28], [1, 3, 7, 14, 28])
            ax.grid(alpha=0.3)
            ax.legend()
        fig.suptitle(f"Transfer vs from-scratch — {csv.stem.replace('transfer_', '')}")
        fig.tight_layout()
        out = OUT / f"fig_{csv.stem}.png"
        fig.savefig(out, dpi=150)
        print(f"wrote {out}")


def plot_horizon_decay(df: pd.DataFrame) -> None:
    man = df[df.dataset.str.startswith("manila")]
    if man.horizon.nunique() < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, label in ((axes[0], "macro_f1", "Macro-F1"),
                           (axes[1], "onset_f1", "Congestion-onset F1")):
        for model, g in man.groupby("model"):
            g = g.sort_values("horizon")
            if len(g) and col in g:
                ax.plot(g["horizon"] * 30, g[col], "o-", label=model)
        ax.set_xlabel("forecast horizon (minutes)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Forecast quality vs horizon — Manila segments")
    fig.tight_layout()
    out = OUT / "fig_horizon_decay.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = collect()
    if df.empty:
        raise SystemExit("no run artifacts found under experiments/runs")
    df.to_csv(OUT / "results_all.csv", index=False)
    (OUT / "results_table.md").write_text(results_markdown(df), encoding="utf-8")
    print(f"wrote {OUT / 'results_all.csv'} ({len(df)} rows)")
    print(f"wrote {OUT / 'results_table.md'}")
    plot_transfer()
    plot_horizon_decay(df)


if __name__ == "__main__":
    main()
