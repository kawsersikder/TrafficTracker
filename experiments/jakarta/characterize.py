"""Characterize the Jakarta / Bandung / Semarang HERE jam-factor GeoPackages.

The Zenodo v2 export is aggregated to 8 time-of-day periods (no raw series),
so this dataset supports spatial characterization (RQ1), not forecasting.
Produces the cross-city comparison table and figures for the paper.

Output (experiments/out/jakarta/):
    city_period_summary.csv    mean/std/p90 jam factor, % heavy, per city x period
    heatmap_city_period.png    city x period mean jam factor
    peak_offpeak.png           morning/evening peak vs off-peak deltas per city

Usage:
    python experiments/jakarta/characterize.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "dataset" / "jakarta-bandung-semarang_zenodo" / "raw" / "zenodo_export"
OUT = REPO / "experiments" / "out" / "jakarta"

CITIES = {"jkt": "Jakarta", "bdg": "Bandung", "smg": "Semarang"}
PERIOD_ORDER = ["night", "morning_peak", "morning_offpeak", "lunch_hours",
                "afternoon_offpeak", "evening_peak", "evening_offpeak", "late_night"]


def main() -> None:
    import geopandas as gpd

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for gpkg in sorted(RAW.glob("*.gpkg")):
        stem = gpkg.stem  # e.g. "evening_peak_jkt"
        city = stem.rsplit("_", 1)[-1]
        period = stem[: -(len(city) + 1)]
        gdf = gpd.read_file(gpkg)
        jf = gdf["jam_factor_mean"].dropna()
        rows.append({
            "city": CITIES.get(city, city), "period": period,
            "segments": len(gdf),
            "jf_mean": jf.mean(), "jf_std": jf.std(),
            "jf_p50": jf.quantile(0.50), "jf_p90": jf.quantile(0.90),
            "pct_heavy_gt6": (jf > 6).mean() * 100,
            "obs_per_segment": gdf["jam_factor_count"].median(),
        })
        print(f"{stem}: {len(gdf)} segments, mean JF {jf.mean():.2f}")

    df = pd.DataFrame(rows)
    df["period"] = pd.Categorical(df["period"], PERIOD_ORDER, ordered=True)
    df = df.sort_values(["city", "period"])
    df.to_csv(OUT / "city_period_summary.csv", index=False)

    pivot = df.pivot(index="city", columns="period", values="jf_mean")
    fig, ax = plt.subplots(figsize=(10, 3))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.iat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, label="mean jam factor")
    fig.suptitle("Mean HERE jam factor by city and time-of-day period")
    fig.tight_layout()
    fig.savefig(OUT / "heatmap_city_period.png", dpi=150)

    fig, ax = plt.subplots(figsize=(7, 4))
    for city, g in df.groupby("city", observed=True):
        ax.plot(g["period"].astype(str), g["jf_mean"], marker="o", label=city)
    ax.set_ylabel("mean jam factor")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    ax.set_title("Daily congestion profile (does city size matter?)")
    fig.tight_layout()
    fig.savefig(OUT / "peak_offpeak.png", dpi=150)

    print(f"\nwrote {OUT}\\city_period_summary.csv and 2 figures")


if __name__ == "__main__":
    main()
