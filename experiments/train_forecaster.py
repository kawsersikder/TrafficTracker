"""Train a deep forecaster (LSTM / GRU / TCN) on a canonical long-format parquet.

Task is inferred from the data (cls >= 0 -> 5-class ordinal classification,
else CI regression) and can be overridden with --task.

Examples (from the repo root):
    python experiments/train_forecaster.py --data experiments/out/manila_segments.parquet --model gru
    python experiments/train_forecaster.py --data experiments/out/manila_segments.parquet --model tcn --horizon 2
    python experiments/train_forecaster.py --data experiments/out/fedesoriano.parquet --model lstm --epochs 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.data import N_FEATURES, build_windows, load_long_parquet, time_split  # noqa: E402
from common.metrics import classification_report, regression_report  # noqa: E402
from common.models import build_model  # noqa: E402


def fmt_dur(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def make_loader(X, t, idx, batch, shuffle):
    ds = TensorDataset(torch.from_numpy(X[idx]), torch.from_numpy(t[idx]))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=0,
                      pin_memory=torch.cuda.is_available())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", choices=["gru", "lstm", "tcn"], default="gru")
    ap.add_argument("--task", choices=["cls", "reg"], default=None)
    ap.add_argument("--history", type=int, default=24)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="run dir (default: experiments/runs/<auto>)")
    args = ap.parse_args()

    t_start = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[1/4] device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda"
             else "  << no GPU found — this will be SLOW, check your torch install"))

    print(f"[2/4] loading {args.data} ...")
    df = load_long_parquet(args.data)
    task = args.task or ("cls" if (df["cls"] >= 0).all() else "reg")
    print(f"      task: {task} | series: {df.series_id.nunique()} | rows: {len(df):,}")

    t0 = time.time()
    W = build_windows(df, history=args.history, horizon=args.horizon)
    split = time_split(W["target_ts"])
    print(f"[3/4] windows built in {fmt_dur(time.time() - t0)}: "
          f"train {split['train'].sum():,} | val {split['val'].sum():,} "
          f"| test {split['test'].sum():,}")

    if task == "cls":
        targets, out_dim = W["cls"].astype(np.int64), 5
        counts = np.bincount(targets[split["train"]], minlength=5).astype(np.float64)
        weights = np.sqrt(counts.sum() / np.maximum(counts, 1))  # tempered inverse-freq
        weights /= weights.mean()
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32, device=device))
        print(f"      class weights: {np.round(weights, 2)}")
    else:
        targets, out_dim = W["y"].astype(np.float32), 1
        criterion = nn.SmoothL1Loss()

    loaders = {k: make_loader(W["X"], targets, np.where(m)[0], args.batch, k == "train")
               for k, m in split.items()}

    model = build_model(args.model, N_FEATURES, out_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      model: {args.model} ({n_params:,} params) | batch {args.batch} "
          f"| max {args.epochs} epochs, early-stop patience {args.patience}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    scaler = torch.amp.GradScaler(device, enabled=device == "cuda")

    name = f"{Path(args.data).stem}_{args.model}_{task}_h{args.horizon}"
    run_dir = Path(args.out) if args.out else Path(__file__).parent / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "best.pt"

    def run_epoch(loader, train: bool, desc: str) -> float:
        model.train(train)
        total, n = 0.0, 0
        bar = tqdm(loader, desc=desc, unit="batch", leave=False, dynamic_ncols=True)
        with torch.set_grad_enabled(train):
            for xb, tb in bar:
                xb, tb = xb.to(device, non_blocking=True), tb.to(device, non_blocking=True)
                with torch.autocast(device, enabled=device == "cuda"):
                    out = model(xb)
                    loss = criterion(out.squeeze(-1) if task == "reg" else out, tb)
                if train:
                    opt.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                total += loss.item() * len(xb)
                n += len(xb)
                bar.set_postfix(loss=f"{total / n:.4f}")
        return total / n

    print(f"[4/4] training — run dir: {run_dir}\n")
    best_val, best_epoch = float("inf"), -1
    epoch_times: list[float] = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(loaders["train"], True, f"epoch {epoch}/{args.epochs} [train]")
        va = run_epoch(loaders["val"], False, f"epoch {epoch}/{args.epochs} [val]  ")
        sched.step(va)
        epoch_times.append(time.time() - t0)
        avg = float(np.mean(epoch_times))
        eta = avg * min(args.epochs - epoch, best_epoch + args.patience - epoch
                        if best_epoch > 0 else args.epochs - epoch)
        marker = ""
        if va < best_val:
            best_val, best_epoch = va, epoch
            torch.save({"model_state": model.state_dict(),
                        "config": {"model": args.model, "task": task, "out_dim": out_dim,
                                   "history": args.history, "horizon": args.horizon,
                                   "in_feats": N_FEATURES}}, ckpt_path)
            marker = "  *best saved"
        print(f"epoch {epoch:3d}/{args.epochs} | train {tr:.4f} | val {va:.4f} "
              f"| {fmt_dur(epoch_times[-1])}/epoch | elapsed {fmt_dur(time.time() - t_start)} "
              f"| ETA <= {fmt_dur(max(eta, 0))}{marker}")
        if epoch - best_epoch >= args.patience:
            print(f"early stop: no val improvement for {args.patience} epochs "
                  f"(best was epoch {best_epoch})")
            break

    # ---- test evaluation with the best checkpoint ----
    print("\nevaluating best checkpoint on the test split ...")
    model.load_state_dict(torch.load(ckpt_path, weights_only=True)["model_state"])
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in tqdm(loaders["test"], desc="test", unit="batch",
                          leave=False, dynamic_ncols=True):
            with torch.autocast(device, enabled=device == "cuda"):
                preds.append(model(xb.to(device)).float().cpu().numpy())
    preds = np.concatenate(preds)
    ti = np.where(split["test"])[0]

    if task == "cls":
        report = classification_report(W["cls"][ti].astype(np.int64),
                                       preds.argmax(1), W["prev_cls"][ti])
    else:
        report = regression_report(W["y"][ti], preds.squeeze(-1), W["prev_y"][ti])

    report.update(best_epoch=best_epoch, best_val_loss=best_val,
                  n_test_windows=int(len(ti)),
                  total_runtime_seconds=round(time.time() - t_start, 1))
    (run_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    print("\nTEST:", json.dumps(report, indent=2))
    print(f"\ndone in {fmt_dur(time.time() - t_start)} | checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
