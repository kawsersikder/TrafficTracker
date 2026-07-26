"""Aggregate the transfer curves across seeds into mean +/- std.

The headline claim is that fine-tuning beats from-scratch at every k. With one
seed per curve that is suggestive; with three it is a result. This collapses
transfer_<target>[_hN][_sK].csv files into a per-(target, k, mode) summary and
reports how many (target, k) cells fine-tuning wins on every seed.

Writes experiments/out/paper/transfer_variance.csv and .md
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "experiments" / "runs"
OUT = REPO / "experiments" / "out" / "paper"

NAME = re.compile(r"^transfer_(?P<target>.+?)(?:_h(?P<h>\d+))?(?:_s(?P<s>\d+))?$")


def main() -> None:
    rows = []
    for csv in sorted(RUNS.glob("transfer_*.csv")):
        m = NAME.match(csv.stem)
        if not m:
            print(f"skip unparsed: {csv.name}")
            continue
        t = pd.read_csv(csv)
        t["target"] = m.group("target")
        t["horizon"] = int(m.group("h") or 1)
        t["seed"] = int(m.group("s") or 42)
        rows.append(t)

    if not rows:
        raise SystemExit("no transfer CSVs found")
    df = pd.concat(rows, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)

    g = (df.groupby(["target", "horizon", "k_days", "mode"])["macro_f1"]
           .agg(["mean", "std", "count"]).reset_index())
    g.to_csv(OUT / "transfer_variance.csv", index=False)

    # how often does finetune beat scratch, per seed, at each (target, horizon, k)?
    ft = df[df["mode"] == "finetune"].set_index(["target", "horizon", "k_days", "seed"])["macro_f1"]
    sc = df[df["mode"] == "scratch"].set_index(["target", "horizon", "k_days", "seed"])["macro_f1"]
    both = pd.concat([ft.rename("finetune"), sc.rename("scratch")], axis=1).dropna()
    both["finetune_wins"] = both["finetune"] > both["scratch"]
    wins, total = int(both["finetune_wins"].sum()), len(both)

    multi = g[(g["count"] > 1) & (g["mode"].isin(["finetune", "scratch"]))]
    lines = [
        "# Transfer curves across seeds", "",
        f"Fine-tuning beats from-scratch in **{wins} of {total}** "
        f"(target, horizon, k, seed) comparisons.", "",
        f"Seeds available per curve: "
        f"{sorted(df.groupby(['target','horizon'])['seed'].nunique().unique().tolist())}", "",
    ]
    if len(multi):
        piv = multi.pivot_table(index=["target", "horizon", "k_days"],
                                columns="mode", values=["mean", "std"])
        lines += ["## Mean +/- std where more than one seed exists", "", piv.round(4).to_markdown(), ""]
    else:
        lines += ["_Only one seed per curve so far - run run_phase2.ps1 for error bars._", ""]

    (OUT / "transfer_variance.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:6]))
    print(f"\nwrote {OUT/'transfer_variance.csv'} and .md")


if __name__ == "__main__":
    main()
