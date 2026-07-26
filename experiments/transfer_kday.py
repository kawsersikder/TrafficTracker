"""The headline experiment: k-day fine-tuning curves.

Pre-train a model on a data-rich source (e.g. most Manila corridors, or
Sathorn Bangkok), then fine-tune on only the LAST k days of the target's train
period and evaluate on the target's fixed test split.  For every k we also
train an identical model from scratch on the same k days — the crossover point
("how many days of local data before local training beats transfer?") is the
paper's headline figure.

When a classification checkpoint is applied to a regression-only dataset
(e.g. Manila cls model -> Sathorn CI series), the target CI is binned onto the
same 5-class ordinal scale so source and target stay comparable.

Examples:
    # source = all Manila roads except EDSA, target = EDSA
    python experiments/transfer_kday.py --data experiments/out/manila_segments.parquet ^
        --target-prefix "EDSA|" --k-days 1 3 7 14 28

    # source = Manila checkpoint, target = Sathorn Bangkok
    python experiments/transfer_kday.py --data experiments/out/sathorn.parquet ^
        --pretrained experiments/runs/manila_segments_gru_cls_h1/best.pt --k-days 1 3 7 14 28
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.data import N_FEATURES, build_windows, ci_to_cls, load_long_parquet, time_split  # noqa: E402
from common.metrics import classification_report, regression_report  # noqa: E402
from common.models import build_model  # noqa: E402


def fmt_dur(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def train_loop(model, X, t, idx, task, device, epochs, lr, desc, batch=512):
    if len(idx) == 0:
        print(f"  {desc}: 0 windows — skipping training")
        return model
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss() if task == "cls" else nn.SmoothL1Loss()
    Xt = torch.from_numpy(X[idx]).to(device)
    tt = torch.from_numpy(t[idx]).to(device)
    model.train()
    bar = tqdm(range(epochs), desc=desc, unit="epoch", leave=False, dynamic_ncols=True)
    for _ in bar:
        perm = torch.randperm(len(idx), device=device)
        total, n = 0.0, 0
        for i in range(0, len(idx), batch):
            sel = perm[i:i + batch]
            out = model(Xt[sel])
            loss = crit(out.squeeze(-1) if task == "reg" else out, tt[sel])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(sel)
            n += len(sel)
        bar.set_postfix(loss=f"{total / n:.4f}")
    return model


def evaluate(model, W, ti, task, device, batch=2048):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(ti), batch):
            xb = torch.from_numpy(W["X"][ti[i:i + batch]]).to(device)
            preds.append(model(xb).float().cpu().numpy())
    preds = np.concatenate(preds)
    if task == "cls":
        return classification_report(W["cls"][ti].astype(np.int64), preds.argmax(1),
                                     W["prev_cls"][ti])
    return regression_report(W["y"][ti], preds.squeeze(-1), W["prev_y"][ti])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="target-city parquet")
    ap.add_argument("--pretrained", default=None,
                    help="checkpoint from train_forecaster.py; omit with --target-prefix "
                         "to pre-train on the complement within the same parquet")
    ap.add_argument("--target-prefix", default=None,
                    help="series_id prefix defining the target; the rest is the source")
    ap.add_argument("--model", default="gru")
    ap.add_argument("--k-days", type=int, nargs="+", default=[1, 3, 7, 14, 28])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--pretrain-epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--history", type=int, default=24)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.pretrained and not args.target_prefix:
        ap.error("need --pretrained and/or --target-prefix")

    t_start = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}"
          + ("" if device == "cuda" else "  << no GPU found — this will be SLOW"))

    df = load_long_parquet(args.data)

    # task comes from the checkpoint when one is given, else from the data
    ck = None
    if args.pretrained:
        ck = torch.load(args.pretrained, weights_only=True)
        task = ck["config"]["task"]
        args.model = ck["config"]["model"]
        args.history = ck["config"]["history"]
        args.horizon = ck["config"]["horizon"]
        print(f"loaded pretrained: {args.pretrained} "
              f"({args.model}, task={task}, history={args.history}, horizon={args.horizon})")
    else:
        task = "cls" if (df["cls"] >= 0).all() else "reg"
    out_dim = 5 if task == "cls" else 1

    # cls model on a regression-only dataset: bin CI onto the shared 5-class scale
    if task == "cls" and (df["cls"] < 0).any():
        df["cls"] = ci_to_cls(df["y"].to_numpy())
        print("target has no ordinal classes — derived cls from CI bins")

    if args.target_prefix:
        tgt_df = df[df.series_id.str.startswith(args.target_prefix)]
        src_df = df[~df.series_id.str.startswith(args.target_prefix)]
        if tgt_df.empty:
            sys.exit(f"no series match prefix {args.target_prefix!r}")
    else:
        tgt_df, src_df = df, None

    W = build_windows(tgt_df, history=args.history, horizon=args.horizon)
    split = time_split(W["target_ts"])
    ti = np.where(split["test"])[0]
    targets = W["cls"].astype(np.int64) if task == "cls" else W["y"].astype(np.float32)
    train_end = W["target_ts"][split["train"]].max()
    n_runs = 1 + 2 * len(args.k_days) + (0 if args.pretrained else 1)
    run_i = 0
    print(f"target: {len(W['series_ids'])} series | test windows: {len(ti):,} "
          f"| task: {task} | total runs: {n_runs}")

    # ---- source model ----
    base = build_model(args.model, N_FEATURES, out_dim).to(device)
    if ck is not None:
        base.load_state_dict(ck["model_state"])
    else:
        run_i += 1
        Ws = build_windows(src_df, history=args.history, horizon=args.horizon)
        ts_ = Ws["cls"].astype(np.int64) if task == "cls" else Ws["y"].astype(np.float32)
        src_train = np.where(time_split(Ws["target_ts"])["train"])[0]
        print(f"\n[run {run_i}/{n_runs}] pre-training on {len(src_train):,} source windows "
              f"({len(Ws['series_ids'])} series, {args.pretrain_epochs} epochs) ...")
        base = train_loop(base, Ws["X"], ts_, src_train, task, device,
                          args.pretrain_epochs, 1e-3, "pretrain")
        print(f"  pre-training done | elapsed {fmt_dur(time.time() - t_start)}")

    run_i += 1
    print(f"\n[run {run_i}/{n_runs}] zero-shot evaluation ...")
    rows = [{"k_days": 0, "mode": "zero_shot", **evaluate(base, W, ti, task, device)}]
    key = "macro_f1" if task == "cls" else "mae"
    print(f"  zero-shot {key}={rows[0][key]:.4f} onset_f1={rows[0]['onset_f1']:.4f}")

    for k in args.k_days:
        cutoff = train_end - np.timedelta64(k, "D")
        idx = np.where(split["train"] & (W["target_ts"] > cutoff))[0]
        print(f"\n--- k={k} days ({len(idx):,} fine-tune windows) "
              f"| elapsed {fmt_dur(time.time() - t_start)} ---")
        for mode in ("finetune", "scratch"):
            run_i += 1
            model = copy.deepcopy(base) if mode == "finetune" else \
                build_model(args.model, N_FEATURES, out_dim).to(device)
            model = train_loop(model, W["X"], targets, idx, task, device,
                               args.epochs, args.lr, f"[run {run_i}/{n_runs}] k={k} {mode}")
            rep = evaluate(model, W, ti, task, device)
            rows.append({"k_days": k, "mode": mode, **rep})
            print(f"  [run {run_i}/{n_runs}] {mode:9s} {key}={rep[key]:.4f} "
                  f"onset_f1={rep['onset_f1']:.4f}")

    # h1 keeps its historical filename so existing artifacts stay valid;
    # other horizons get an explicit suffix instead of overwriting them.
    hsuf = "" if args.horizon == 1 else f"_h{args.horizon}"
    out = Path(__file__).parent / "runs" / (
        f"transfer_{Path(args.data).stem}_"
        f"{(args.target_prefix or 'all').replace('|', '_')}{hsuf}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nall {n_runs} runs done in {fmt_dur(time.time() - t_start)}\nwrote {out}")


if __name__ == "__main__":
    main()
