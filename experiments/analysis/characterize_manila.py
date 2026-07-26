"""RQ1 characterization of the reconstructed Manila segment series, plus the
cross-city comparison table against Jakarta/Bandung/Semarang.

Writes to experiments/out/paper/:
    manila_markov_transitions.csv   5x5 row-normalized transition matrix (30-min steps)
    fig_manila_markov.png           transition heatmap + dwell times
    fig_manila_daily_profile.png    mean CI by time of day, weekday vs weekend
    fig_manila_week_heatmap.png     mean CI, day-of-week x hour
    manila_road_ranking.csv         per-road mean CI, % time congested, coverage
    manila_episodes.csv             congestion episode duration stats per road
    cross_city_periods.csv          Manila vs Jakarta/Bandung/Semarang, aligned on
                                    the Zenodo dataset's 8 time-of-day periods

Usage:
    python experiments/analysis/characterize_manila.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments"))
from common.data import ONSET_CLS_THRESHOLD  # noqa: E402

OUT = REPO / "experiments" / "out" / "paper"
PARQUET = REPO / "experiments" / "out" / "manila_segments.parquet"
JAKARTA_CSV = REPO / "experiments" / "out" / "jakarta" / "city_period_summary.csv"

STATES = ["L", "ML", "M", "MH", "H"]
STEP_MIN = 30
# the Zenodo dataset's 8 time-of-day periods (hour ranges, end-exclusive)
PERIODS = [("night", 0, 6), ("morning_peak", 6, 9), ("morning_offpeak", 9, 12),
           ("lunch_hours", 12, 14), ("afternoon_offpeak", 14, 16),
           ("evening_peak", 16, 19), ("evening_offpeak", 19, 22),
           ("late_night", 22, 24)]


def hour_to_period(hour: pd.Series) -> pd.Series:
    bins = [0, 6, 9, 12, 14, 16, 19, 22, 24]
    return pd.cut(hour, bins=bins, right=False, labels=[p for p, _, _ in PERIODS])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PARQUET).sort_values(["series_id", "ts"])
    df["road"] = df["series_id"].str.split("|").str[0]
    print(f"{len(df):,} observations, {df.series_id.nunique()} series, "
          f"{df.road.nunique()} roads")

    # ---- Markov transition matrix over consecutive 30-min observations ----
    nxt_cls = df.groupby("series_id")["cls"].shift(-1)
    gap_ok = (df.groupby("series_id")["ts"].shift(-1) - df["ts"]
              ) <= pd.Timedelta(minutes=STEP_MIN * 1.5)
    pairs = df[gap_ok & nxt_cls.notna()]
    trans = np.zeros((5, 5))
    np.add.at(trans, (pairs["cls"].to_numpy(int),
                      nxt_cls[gap_ok & nxt_cls.notna()].to_numpy(int)), 1)
    tm = trans / trans.sum(1, keepdims=True)
    pd.DataFrame(tm, index=STATES, columns=STATES).round(4).to_csv(
        OUT / "manila_markov_transitions.csv")
    dwell = STEP_MIN / (1 - np.diag(tm))  # expected dwell time per state, minutes

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    im = axes[0].imshow(tm, cmap="YlOrRd", vmin=0, vmax=1)
    axes[0].set_xticks(range(5), STATES)
    axes[0].set_yticks(range(5), STATES)
    axes[0].set_xlabel("state at t+30 min")
    axes[0].set_ylabel("state at t")
    for i in range(5):
        for j in range(5):
            axes[0].text(j, i, f"{tm[i, j]:.2f}", ha="center", va="center",
                         fontsize=8, color="black" if tm[i, j] < 0.6 else "white")
    fig.colorbar(im, ax=axes[0])
    axes[0].set_title("Markov transitions (30-min steps)")
    axes[1].bar(STATES, dwell, color="steelblue")
    axes[1].set_ylabel("expected dwell time (minutes)")
    axes[1].set_title("State persistence")
    fig.tight_layout()
    fig.savefig(OUT / "fig_manila_markov.png", dpi=150)
    print("transition diag:", np.round(np.diag(tm), 3),
          "| dwell (min):", np.round(dwell, 0))

    # ---- daily profile: weekday vs weekend ----
    df["slot"] = df["ts"].dt.hour + df["ts"].dt.minute / 60
    df["weekend"] = df["ts"].dt.dayofweek >= 5
    prof = df.groupby(["weekend", "slot"])["y"].mean().unstack(0)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(prof.index, prof[False], label="weekday")
    ax.plot(prof.index, prof[True], label="weekend")
    ax.set_xlabel("hour of day")
    ax.set_ylabel("mean Congestion Index")
    ax.set_xticks(range(0, 25, 2))
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Manila mean CI by time of day (Nov 2015 – Jun 2016)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_manila_daily_profile.png", dpi=150)

    # ---- day-of-week x hour heatmap ----
    hm = df.groupby([df["ts"].dt.dayofweek, df["ts"].dt.hour])["y"].mean().unstack()
    fig, ax = plt.subplots(figsize=(10, 3.2))
    im = ax.imshow(hm.to_numpy(), aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_xlabel("hour of day")
    fig.colorbar(im, label="mean CI")
    ax.set_title("Manila weekly congestion rhythm")
    fig.tight_layout()
    fig.savefig(OUT / "fig_manila_week_heatmap.png", dpi=150)

    # ---- road ranking + honest coverage stats ----
    expected = (df.groupby("series_id")["ts"].agg(lambda s: (s.max() - s.min())
                / pd.Timedelta(minutes=STEP_MIN)) + 1)
    coverage = (df.groupby("series_id").size() / expected).rename("coverage")
    road = df.groupby("road").agg(mean_ci=("y", "mean"),
                                  pct_congested=("cls", lambda c: (c >= ONSET_CLS_THRESHOLD).mean() * 100),
                                  n_series=("series_id", "nunique"),
                                  n_obs=("ts", "size"))
    road["median_coverage"] = df[["road", "series_id"]].drop_duplicates() \
        .join(coverage, on="series_id").groupby("road")["coverage"].median()
    road.round(4).sort_values("pct_congested", ascending=False) \
        .to_csv(OUT / "manila_road_ranking.csv")
    print("\nroad ranking (% time in MH/H):")
    print(road.sort_values("pct_congested", ascending=False)
          [["pct_congested", "mean_ci", "median_coverage"]].round(3).to_string())

    # ---- congestion episodes: duration of contiguous MH/H spells ----
    df["hot"] = df["cls"] >= ONSET_CLS_THRESHOLD
    grp = df.groupby("series_id", group_keys=False)
    new_spell = (df["hot"] != grp["hot"].shift()) | \
                ((df["ts"] - grp["ts"].shift()) > pd.Timedelta(minutes=STEP_MIN * 1.5))
    df["spell"] = new_spell.cumsum()
    ep = (df[df["hot"]].groupby(["road", "spell"]).size() * STEP_MIN).rename("minutes")
    stats = ep.groupby("road").agg(episodes="size", median_min="median",
                                   p90_min=lambda s: s.quantile(0.9)).round(1)
    stats.to_csv(OUT / "manila_episodes.csv")
    print("\ncongestion episode durations (min):")
    print(stats.to_string())

    # ---- cross-city table aligned on the Zenodo 8 periods ----
    df["period"] = hour_to_period(df["ts"].dt.hour)
    manila = df.groupby("period", observed=True).agg(
        mean_ci=("y", "mean"),
        pct_congested=("cls", lambda c: (c >= ONSET_CLS_THRESHOLD).mean() * 100)).reset_index()
    manila.insert(0, "city", "Manila (2015-16)")
    if JAKARTA_CSV.exists():
        jk = pd.read_csv(JAKARTA_CSV)
        jk = pd.DataFrame({"city": jk["city"] + " (2025-26)", "period": jk["period"],
                           "mean_ci": jk["jf_mean"] / 10,
                           "pct_congested": jk["pct_heavy_gt6"]})
        cross = pd.concat([manila, jk], ignore_index=True)
    else:
        print("! jakarta summary missing — run experiments/jakarta/characterize.py first")
        cross = manila
    cross["period"] = pd.Categorical(cross["period"], [p for p, _, _ in PERIODS],
                                     ordered=True)
    cross = cross.sort_values(["city", "period"]).round(4)
    cross.to_csv(OUT / "cross_city_periods.csv", index=False)
    print(f"\nwrote cross-city table + all figures to {OUT}")


if __name__ == "__main__":
    main()
