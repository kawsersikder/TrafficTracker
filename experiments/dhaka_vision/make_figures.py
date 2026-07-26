"""Figures for the Dhaka heterogeneity section.

    fig_dhaka_mix_by_window.png   stacked share of vehicle classes by time of day
    fig_dhaka_two_sources.png     TFP-BD vs DhakaAI, vehicles only
    fig_dhaka_yolo_scaling.png    per-class mAP50, v8s vs v8m
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "experiments" / "out" / "dhaka"
ORDER = ["rickshaw", "three-wheeler (CNG)", "motorbike", "bicycle",
         "car", "bus", "truck", "mini-truck", "pedestrian"]
# heterogeneous (non-lane-disciplined) modes get warm colours, standard traffic cool
COLORS = {"rickshaw": "#d1495b", "three-wheeler (CNG)": "#edae49",
          "motorbike": "#f79256", "bicycle": "#c05761",
          "car": "#00798c", "bus": "#30638e", "truck": "#003d5b",
          "mini-truck": "#4a7c94", "pedestrian": "#8d99ae"}


def fig_by_window() -> None:
    w = pd.read_csv(OUT / "tfpbd_mix_by_window.csv", index_col=0)
    cols = [c for c in ORDER if c in w.columns]
    w = w[cols] * 100
    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = pd.Series(0.0, index=w.index)
    for c in cols:
        ax.bar(w.index, w[c], bottom=bottom, label=c, color=COLORS.get(c), width=0.62)
        bottom += w[c]
    ax.set_ylabel("share of annotated objects (%)")
    ax.set_xlabel("time-of-day window")
    ax.set_title("Dhaka road-user mix by time of day (TFP-BD, 265,698 objects, 4 intersections)")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False, fontsize=9)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "fig_dhaka_mix_by_window.png", dpi=300)
    plt.close(fig)


def fig_two_sources() -> None:
    c = pd.read_csv(OUT / "dhaka_class_comparison_vehicles_only.csv", index_col=0)
    c = c[(c["tfpbd_%"] > 0) | (c["dhakaai_%"] > 0)].sort_values("tfpbd_%", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    y = range(len(c))
    ax.barh([i + 0.2 for i in y], c["tfpbd_%"], height=0.38, label="TFP-BD (continuous frames)", color="#d1495b")
    ax.barh([i - 0.2 for i in y], c["dhakaai_%"], height=0.38, label="DhakaAI (curated set)", color="#00798c")
    ax.set_yticks(list(y))
    ax.set_yticklabels(c.index)
    ax.set_xlabel("share of annotated vehicles (%)")
    ax.set_title("Dhaka vehicle mix: two independent sources")
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "fig_dhaka_two_sources.png", dpi=300)
    plt.close(fig)


def fig_yolo() -> None:
    p = OUT / "yolo_per_class.csv"
    if not p.exists():
        print("skip yolo figure: run per_class_table.py first")
        return
    t = pd.read_csv(p, index_col=0)
    t = t[t["instances"] >= 20].sort_values("instances", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    y = range(len(t))
    ax.barh([i + 0.2 for i in y], t["YOLOv8m mAP50"], height=0.38, label="YOLOv8m", color="#00798c")
    ax.barh([i - 0.2 for i in y], t["YOLOv8s mAP50"], height=0.38, label="YOLOv8s", color="#8d99ae")
    ax.set_yticks(list(y))
    ax.set_yticklabels([f"{i}  (n={n})" for i, n in zip(t.index, t["instances"])], fontsize=9)
    ax.set_xlabel("mAP50")
    ax.set_title("DhakaAI detection by class (classes with >=20 val instances)")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "fig_dhaka_yolo_scaling.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    fig_by_window()
    fig_two_sources()
    fig_yolo()
    for f in sorted(OUT.glob("fig_*.png")):
        print(f"wrote {f.name}  ({f.stat().st_size/1024:.0f} KB)")
