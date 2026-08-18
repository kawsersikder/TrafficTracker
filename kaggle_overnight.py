"""
OVERNIGHT RUN — Single self-contained cell for Kaggle.
Paste this ENTIRE thing into ONE cell, hit Run All, go to sleep.
"""

# ============================================================================
# IMPORTS
# ============================================================================
import os, sys, json, time, copy, math, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_ROOT = Path("/kaggle/input/datasets/kawsersikder/traffictracker-data")
OUT_ROOT  = Path("/kaggle/working/results")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

MANILA_PARQUET = DATA_ROOT / "manila_segments.parquet"
SATHORN_PARQUET = DATA_ROOT / "sathorn.parquet"
CCTV_PARQUET   = DATA_ROOT / "sathorn_cctv.parquet"
SERIES_META    = DATA_ROOT / "manila_series_meta.csv"

CKPT_H1 = DATA_ROOT / "ckpt_manila_gru_h1.pt"
CKPT_H2 = DATA_ROOT / "ckpt_manila_gru_h2.pt"
CKPT_H4 = DATA_ROOT / "ckpt_manila_gru_h4.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else ""))

# ============================================================================
# DATA UTILITIES
# ============================================================================
CI_CLASS_BINS = np.array([0.2, 0.4, 0.6, 0.8])
ONSET_CLS_THRESHOLD = 3
N_FEATURES = 5

def ci_to_cls(ci):
    return np.digitize(ci, CI_CLASS_BINS).astype(np.int8)

def load_long_parquet(path):
    df = pd.read_parquet(path)
    if "cls" not in df.columns:
        df["cls"] = np.int8(-1)
    df["ts"] = pd.to_datetime(df["ts"])
    df["y"] = df["y"].astype(np.float32)
    df["cls"] = df["cls"].astype(np.int8)
    return df.sort_values(["series_id", "ts"], kind="stable").reset_index(drop=True)

def _time_features(ts):
    tod = (ts.dt.hour * 3600 + ts.dt.minute * 60 + ts.dt.second).to_numpy() / 86400.0
    dow = ts.dt.dayofweek.to_numpy() / 7.0
    return np.stack([
        np.sin(2 * np.pi * tod), np.cos(2 * np.pi * tod),
        np.sin(2 * np.pi * dow), np.cos(2 * np.pi * dow),
    ], axis=1).astype(np.float32)

def build_windows(df, history=24, horizon=1, gap_factor=1.9):
    xs, ys, cs, pcs, pys, tss, sids = [], [], [], [], [], [], []
    series_ids = []
    groups = df.groupby("series_id", sort=True)
    for sid, g in tqdm(groups, total=groups.ngroups, desc="building windows",
                       unit="series", leave=False):
        n = len(g)
        span = history + horizon
        if n < span + 1:
            continue
        ts = g["ts"]
        feats = np.concatenate(
            [g["y"].to_numpy(np.float32)[:, None], _time_features(ts)], axis=1)
        diffs = ts.diff().dt.total_seconds().to_numpy()[1:]
        step = np.nanmedian(diffs)
        bad = np.concatenate([[0], (diffs > gap_factor * step).astype(np.int64)])
        bad_cum = np.cumsum(bad)
        starts = np.arange(0, n - span + 1)
        tgt = starts + span - 1
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
    return {
        "X": np.concatenate(xs), "y": np.concatenate(ys),
        "cls": np.concatenate(cs), "prev_cls": np.concatenate(pcs),
        "prev_y": np.concatenate(pys), "target_ts": np.concatenate(tss),
        "series": np.concatenate(sids), "series_ids": series_ids,
    }

def time_split(target_ts, train=0.70, val=0.15):
    q_train = np.quantile(target_ts.astype("int64"), train)
    q_val = np.quantile(target_ts.astype("int64"), train + val)
    t = target_ts.astype("int64")
    return {"train": t <= q_train, "val": (t > q_train) & (t <= q_val), "test": t > q_val}

# ============================================================================
# METRICS
# ============================================================================
def onset_f1(true_cls, pred_cls, prev_true_cls, thr=ONSET_CLS_THRESHOLD):
    eligible = prev_true_cls < thr
    t = (true_cls >= thr) & eligible
    p = (pred_cls >= thr) & eligible
    tp = int((t & p).sum()); fp = int((~t & p).sum()); fn = int((t & ~p).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"onset_precision": prec, "onset_recall": rec, "onset_f1": f1, "onset_events": int(t.sum())}

def classification_report(true_cls, pred_cls, prev_true_cls):
    rep = {
        "accuracy": float((true_cls == pred_cls).mean()),
        "macro_f1": float(f1_score(true_cls, pred_cls, average="macro", zero_division=0)),
    }
    for c, f1c in enumerate(f1_score(true_cls, pred_cls, average=None,
                                     labels=list(range(5)), zero_division=0)):
        rep[f"f1_class_{c}"] = float(f1c)
    rep.update(onset_f1(true_cls, pred_cls, prev_true_cls))
    return rep

def regression_report(true_ci, pred_ci, prev_true_ci):
    err = pred_ci - true_ci
    rep = {"mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err ** 2).mean()))}
    rep.update(onset_f1(ci_to_cls(true_ci), ci_to_cls(np.clip(pred_ci, 0, 1)),
                        ci_to_cls(prev_true_ci)))
    return rep

def fmt_dur(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")

# ============================================================================
# MODELS
# ============================================================================
class RNNForecaster(nn.Module):
    def __init__(self, in_feats, out_dim, cell="gru", hidden=128, layers=2, dropout=0.2):
        super().__init__()
        rnn_cls = nn.GRU if cell == "gru" else nn.LSTM
        self.rnn = rnn_cls(in_feats, hidden, num_layers=layers,
                           batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 2),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden // 2, out_dim))
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1])

class _CausalBlock(nn.Module):
    def __init__(self, c_in, c_out, dilation, k=3, dropout=0.2):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv1 = nn.Conv1d(c_in, c_out, k, dilation=dilation)
        self.conv2 = nn.Conv1d(c_out, c_out, k, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(c_out)
        self.norm2 = nn.BatchNorm1d(c_out)
        self.drop = nn.Dropout(dropout)
        self.down = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
    def forward(self, x):
        res = self.down(x)
        h = F.pad(x, (self.pad, 0))
        h = self.drop(torch.relu(self.norm1(self.conv1(h))))
        h = F.pad(h, (self.pad, 0))
        h = self.drop(torch.relu(self.norm2(self.conv2(h))))
        return torch.relu(h + res)

class TCNForecaster(nn.Module):
    def __init__(self, in_feats, out_dim, channels=64, dilations=(1,2,4,8), dropout=0.2):
        super().__init__()
        blocks, c = [], in_feats
        for d in dilations:
            blocks.append(_CausalBlock(c, channels, d, dropout=dropout))
            c = channels
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Linear(channels, out_dim)
    def forward(self, x):
        h = self.tcn(x.transpose(1, 2))
        return self.head(h[:, :, -1])

class TransformerForecaster(nn.Module):
    def __init__(self, in_feats, out_dim, d_model=64, nhead=4, num_layers=3, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(in_feats, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, 128, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model // 2, out_dim))
    def forward(self, x):
        h = self.input_proj(x) + self.pos_enc[:, :x.size(1), :]
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        h = self.encoder(h, mask=mask)
        return self.head(h[:, -1])

class GraphConv(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.W = nn.Linear(in_feats, out_feats)
    def forward(self, x, A_hat):
        return torch.relu(self.W(A_hat @ x))

class STGNNForecaster(nn.Module):
    def __init__(self, in_feats, out_dim, n_nodes, hidden=64, gcn_hidden=32, dropout=0.2):
        super().__init__()
        self.n_nodes = n_nodes
        self.gcn1 = GraphConv(in_feats, gcn_hidden)
        self.gcn2 = GraphConv(gcn_hidden, gcn_hidden)
        self.temporal = nn.GRU(gcn_hidden, hidden, num_layers=2,
                               batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 2),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden // 2, out_dim))
    def forward(self, x, A_hat):
        B, T, N, Fin = x.shape
        gcn_out = []
        for t in range(T):
            h = self.gcn1(x[:, t], A_hat)
            h = self.gcn2(h, A_hat)
            gcn_out.append(h)
        gcn_out = torch.stack(gcn_out, dim=1)
        gcn_out = gcn_out.permute(0, 2, 1, 3).reshape(B * N, T, -1)
        temporal_out, _ = self.temporal(gcn_out)
        last = temporal_out[:, -1]
        out = self.head(last)
        return out.view(B, N, -1)

def build_model(name, in_feats, out_dim):
    name = name.lower()
    if name in ("gru", "lstm"):
        return RNNForecaster(in_feats, out_dim, cell=name)
    if name == "tcn":
        return TCNForecaster(in_feats, out_dim)
    if name == "transformer":
        return TransformerForecaster(in_feats, out_dim)
    raise ValueError(f"unknown model '{name}'")

# ============================================================================
# TRAINING HELPERS
# ============================================================================
def make_loader(X, t, idx, batch, shuffle):
    ds = TensorDataset(torch.from_numpy(X[idx]), torch.from_numpy(t[idx]))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle,
                      num_workers=0, pin_memory=DEVICE == "cuda")

def train_and_eval(model, W, split, task, name, out_dir,
                   epochs=30, batch=512, lr=1e-3, patience=5, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    model = model.to(DEVICE)
    if task == "cls":
        targets = W["cls"].astype(np.int64)
        counts = np.bincount(targets[split["train"]], minlength=5).astype(np.float64)
        weights = np.sqrt(counts.sum() / np.maximum(counts, 1))
        weights /= weights.mean()
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32, device=DEVICE))
    else:
        targets = W["y"].astype(np.float32)
        criterion = nn.SmoothL1Loss()

    loaders = {k: make_loader(W["X"], targets, np.where(m)[0], batch, k == "train")
               for k, m in split.items()}
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    scaler = torch.amp.GradScaler(DEVICE, enabled=DEVICE == "cuda")

    run_dir = out_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "best.pt"

    def run_epoch(loader, train_mode):
        model.train(train_mode)
        total, n = 0.0, 0
        with torch.set_grad_enabled(train_mode):
            for xb, tb in loader:
                xb = xb.to(DEVICE, non_blocking=True)
                tb = tb.to(DEVICE, non_blocking=True)
                with torch.autocast(DEVICE, enabled=DEVICE == "cuda"):
                    out = model(xb)
                    loss = criterion(out.squeeze(-1) if task == "reg" else out, tb)
                if train_mode:
                    opt.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                total += loss.item() * len(xb); n += len(xb)
        return total / n

    t0 = time.time()
    best_val, best_epoch = float("inf"), -1
    print(f"\n  Training: {name} | task={task} | params={sum(p.numel() for p in model.parameters()):,}")
    for epoch in range(1, epochs + 1):
        tr = run_epoch(loaders["train"], True)
        va = run_epoch(loaders["val"], False)
        sched.step(va)
        if va < best_val:
            best_val, best_epoch = va, epoch
            torch.save({"model_state": model.state_dict(),
                        "config": {"model": name.split("_")[0], "task": task}}, ckpt_path)
        if epoch % 5 == 0 or epoch == 1:
            print(f"    epoch {epoch:3d}/{epochs} | train {tr:.4f} | val {va:.4f}"
                  + ("  *best" if epoch == best_epoch else ""))
        if epoch - best_epoch >= patience:
            print(f"    early stop at epoch {epoch} (best={best_epoch})")
            break

    model.load_state_dict(torch.load(ckpt_path, weights_only=True)["model_state"])
    model.eval()
    ti = np.where(split["test"])[0]
    preds = []
    with torch.no_grad():
        for xb, _ in loaders["test"]:
            with torch.autocast(DEVICE, enabled=DEVICE == "cuda"):
                preds.append(model(xb.to(DEVICE)).float().cpu().numpy())
    preds = np.concatenate(preds)
    if task == "cls":
        report = classification_report(W["cls"][ti].astype(np.int64),
                                       preds.argmax(1), W["prev_cls"][ti])
    else:
        report = regression_report(W["y"][ti], preds.squeeze(-1), W["prev_y"][ti])
    report["runtime_seconds"] = round(time.time() - t0, 1)
    (run_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    key = "macro_f1" if task == "cls" else "mae"
    print(f"    TEST: {key}={report[key]:.4f} onset_f1={report['onset_f1']:.4f} ({fmt_dur(time.time()-t0)})")
    return report

def transfer_kday(data_path, pretrained_path, target_prefix, k_days,
                  model_name="gru", horizon=1, seed=42, epochs=15, lr=3e-4):
    torch.manual_seed(seed); np.random.seed(seed)
    df = load_long_parquet(data_path)
    ck = torch.load(pretrained_path, weights_only=True, map_location=DEVICE)
    task = ck["config"]["task"]
    model_name = ck["config"]["model"]
    history = ck["config"]["history"]
    horizon_ck = ck["config"]["horizon"]
    out_dim = 5 if task == "cls" else 1
    if task == "cls" and (df["cls"] < 0).any():
        df["cls"] = ci_to_cls(df["y"].to_numpy())
    if target_prefix:
        tgt_df = df[df.series_id.str.startswith(target_prefix)]
    else:
        tgt_df = df
    W = build_windows(tgt_df, history=history, horizon=horizon_ck)
    split = time_split(W["target_ts"])
    ti = np.where(split["test"])[0]
    targets = W["cls"].astype(np.int64) if task == "cls" else W["y"].astype(np.float32)
    train_end = W["target_ts"][split["train"]].max()
    base = build_model(model_name, N_FEATURES, out_dim).to(DEVICE)
    base.load_state_dict(ck["model_state"])
    def do_eval(model):
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(ti), 2048):
                xb = torch.from_numpy(W["X"][ti[i:i+2048]]).to(DEVICE)
                preds.append(model(xb).float().cpu().numpy())
        preds = np.concatenate(preds)
        if task == "cls":
            return classification_report(W["cls"][ti].astype(np.int64),
                                         preds.argmax(1), W["prev_cls"][ti])
        return regression_report(W["y"][ti], preds.squeeze(-1), W["prev_y"][ti])
    def do_train(model, idx):
        if len(idx) == 0: return model
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss() if task == "cls" else nn.SmoothL1Loss()
        Xt = torch.from_numpy(W["X"][idx]).to(DEVICE)
        tt = torch.from_numpy(targets[idx]).to(DEVICE)
        model.train()
        for _ in range(epochs):
            perm = torch.randperm(len(idx), device=DEVICE)
            for i in range(0, len(idx), 512):
                sel = perm[i:i+512]
                out = model(Xt[sel])
                loss = crit(out.squeeze(-1) if task == "reg" else out, tt[sel])
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        return model
    key = "macro_f1" if task == "cls" else "mae"
    rows = [{"k_days": 0, "mode": "zero_shot", **do_eval(base)}]
    print(f"    zero-shot {key}={rows[0][key]:.4f}")
    for k in k_days:
        cutoff = train_end - np.timedelta64(k, "D")
        idx = np.where(split["train"] & (W["target_ts"] > cutoff))[0]
        for mode in ("finetune", "scratch"):
            m = copy.deepcopy(base) if mode == "finetune" else \
                build_model(model_name, N_FEATURES, out_dim).to(DEVICE)
            m = do_train(m, idx)
            rep = do_eval(m)
            rows.append({"k_days": k, "mode": mode, **rep})
            print(f"    k={k:2d} {mode:9s} {key}={rep[key]:.4f} onset_f1={rep['onset_f1']:.4f}")
    return pd.DataFrame(rows)

# ============================================================================
# ST-GNN HELPERS
# ============================================================================
def build_manila_adjacency():
    meta = pd.read_csv(SERIES_META)
    series_ids = sorted(meta["series_id"].tolist())
    n = len(series_ids)
    id_to_idx = {s: i for i, s in enumerate(series_ids)}
    corridors = defaultdict(list)
    for sid in series_ids:
        parts = sid.split("|")
        corridor_key = f"{parts[0]}|{parts[1]}"
        seg_num = int(parts[2][1:])
        corridors[corridor_key].append((seg_num, sid))
    A = np.zeros((n, n), dtype=np.float32)
    for corridor, segments in corridors.items():
        segments.sort(key=lambda x: x[0])
        for i in range(len(segments) - 1):
            idx_a = id_to_idx[segments[i][1]]
            idx_b = id_to_idx[segments[i + 1][1]]
            A[idx_a, idx_b] = 1.0
            A[idx_b, idx_a] = 1.0
    A_hat = A + np.eye(n, dtype=np.float32)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(A_hat.sum(axis=1), 1e-8)))
    A_hat = D_inv_sqrt @ A_hat @ D_inv_sqrt
    print(f"    Adjacency: {n} nodes, {int(A.sum())//2} edges, {len(corridors)} corridors")
    return A_hat, series_ids

def build_stgnn_windows(df, series_ids, history=24, horizon=1):
    pivot = df.pivot_table(index="ts", columns="series_id", values="y", aggfunc="first")
    pivot = pivot.reindex(columns=series_ids)
    pivot = pivot.interpolate(limit=2).dropna(thresh=int(len(series_ids) * 0.8))
    pivot = pivot.fillna(0.0)
    ts_index = pivot.index
    values = pivot.values.astype(np.float32)
    N = len(series_ids)
    tod = (ts_index.hour * 3600 + ts_index.minute * 60 + ts_index.second) / 86400.0
    dow = ts_index.dayofweek / 7.0
    time_feats = np.stack([
        np.sin(2 * np.pi * tod), np.cos(2 * np.pi * tod),
        np.sin(2 * np.pi * dow), np.cos(2 * np.pi * dow),
    ], axis=1).astype(np.float32)
    span = history + horizon
    T_total = len(values)
    diffs = np.diff(ts_index.astype(np.int64) // 10**9)
    step = np.median(diffs)
    bad = np.concatenate([[0], (diffs > 1.9 * step).astype(np.int64)])
    bad_cum = np.cumsum(bad)
    starts = np.arange(0, T_total - span + 1)
    tgt = starts + span - 1
    valid = (bad_cum[tgt] - bad_cum[starts]) == 0
    starts, tgt = starts[valid], tgt[valid]
    print(f"    STGNN windows: {len(starts)} valid from {T_total} timestamps")
    X_list, y_list, cls_list, prev_cls_list, prev_y_list, ts_list = [], [], [], [], [], []
    for s, t_idx in zip(starts, tgt):
        ci_window = values[s:s + history]
        tf_window = time_feats[s:s + history]
        feat = np.concatenate([
            ci_window[:, :, None],
            np.tile(tf_window[:, None, :], (1, N, 1))
        ], axis=2)
        X_list.append(feat)
        target_ci = values[t_idx]
        y_list.append(target_ci)
        cls_list.append(ci_to_cls(target_ci))
        prev_y_list.append(values[t_idx - 1])
        prev_cls_list.append(ci_to_cls(values[t_idx - 1]))
        ts_list.append(ts_index[t_idx])
    return {
        "X": np.array(X_list, dtype=np.float32),
        "y": np.array(y_list, dtype=np.float32),
        "cls": np.array(cls_list, dtype=np.int8),
        "prev_cls": np.array(prev_cls_list, dtype=np.int8),
        "prev_y": np.array(prev_y_list, dtype=np.float32),
        "target_ts": np.array(ts_list, dtype="datetime64[ns]"),
        "series_ids": series_ids,
    }

def train_stgnn(horizon=1, epochs=30, patience=5, batch=64, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    ssuf = "" if seed == 42 else f"_s{seed}"
    name = f"stgnn_manila_cls_h{horizon}{ssuf}"
    print(f"\n  === ST-GNN: {name} ===")
    df = load_long_parquet(MANILA_PARQUET)
    A_hat_np, series_ids = build_manila_adjacency()
    A_hat = torch.tensor(A_hat_np, dtype=torch.float32, device=DEVICE)
    N = len(series_ids)
    W = build_stgnn_windows(df, series_ids, history=24, horizon=horizon)
    split = time_split(W["target_ts"])
    targets_all = W["cls"].astype(np.int64)
    model = STGNNForecaster(in_feats=N_FEATURES, out_dim=5, n_nodes=N,
                            hidden=64, gcn_hidden=32).to(DEVICE)
    print(f"    Model: {sum(p.numel() for p in model.parameters()):,} params")
    counts = np.bincount(targets_all[split["train"]].ravel(), minlength=5).astype(np.float64)
    weights = np.sqrt(counts.sum() / np.maximum(counts, 1))
    weights /= weights.mean()
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=DEVICE))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    run_dir = OUT_ROOT / name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "best.pt"
    train_idx = np.where(split["train"])[0]
    val_idx = np.where(split["val"])[0]
    test_idx = np.where(split["test"])[0]

    def run_epoch(indices, train_mode):
        model.train(train_mode)
        total, ns = 0.0, 0
        perm = np.random.permutation(indices) if train_mode else indices
        with torch.set_grad_enabled(train_mode):
            for i in range(0, len(perm), batch):
                sel = perm[i:i + batch]
                xb = torch.from_numpy(W["X"][sel]).to(DEVICE)
                tb = torch.from_numpy(targets_all[sel]).to(DEVICE)
                with torch.autocast(DEVICE, enabled=DEVICE == "cuda"):
                    out = model(xb, A_hat)
                    loss = criterion(out.reshape(-1, 5), tb.reshape(-1))
                if train_mode:
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                total += loss.item() * len(sel); ns += len(sel)
        return total / max(ns, 1)

    t0 = time.time()
    best_val, best_epoch = float("inf"), -1
    for epoch in range(1, epochs + 1):
        tr = run_epoch(train_idx, True); va = run_epoch(val_idx, False)
        sched.step(va)
        if va < best_val:
            best_val, best_epoch = va, epoch
            torch.save({"model_state": model.state_dict()}, ckpt_path)
        if epoch % 5 == 0 or epoch == 1:
            print(f"    epoch {epoch:3d}/{epochs} | train {tr:.4f} | val {va:.4f}"
                  + ("  *best" if epoch == best_epoch else ""))
        if epoch - best_epoch >= patience: break

    model.load_state_dict(torch.load(ckpt_path, weights_only=True)["model_state"])
    model.eval()
    all_preds, all_true, all_prev = [], [], []
    with torch.no_grad():
        for i in range(0, len(test_idx), batch):
            sel = test_idx[i:i + batch]
            xb = torch.from_numpy(W["X"][sel]).to(DEVICE)
            with torch.autocast(DEVICE, enabled=DEVICE == "cuda"):
                out = model(xb, A_hat)
            all_preds.append(out.float().cpu().numpy())
            all_true.append(W["cls"][sel]); all_prev.append(W["prev_cls"][sel])
    preds = np.concatenate(all_preds).reshape(-1, 5)
    true_c = np.concatenate(all_true).ravel().astype(np.int64)
    prev_c = np.concatenate(all_prev).ravel().astype(np.int8)
    report = classification_report(true_c, preds.argmax(1), prev_c)
    report["runtime_seconds"] = round(time.time() - t0, 1)
    (run_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    print(f"    TEST: macro_f1={report['macro_f1']:.4f} onset_f1={report['onset_f1']:.4f} ({fmt_dur(time.time()-t0)})")
    return report

# ============================================================================
# SKIP HELPER — don't re-run experiments that already have results
# ============================================================================
def already_done(name):
    """Check if a run directory already has metrics.json or a CSV exists."""
    metrics = OUT_ROOT / name / "metrics.json"
    csv = OUT_ROOT / f"{name}.csv"
    if metrics.exists():
        print(f"  SKIP {name} (already done)")
        return True
    if csv.exists():
        print(f"  SKIP {name} (already done)")
        return True
    return False

# ============================================================================
# MAIN — RUN EVERYTHING REMAINING
# ============================================================================
GRAND_START = time.time()

# ------------------------------------------------------------------
# PHASE 1: ST-GNN on Manila (seed 42 + seeds 1,2 for error bars)
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("  PHASE 1: ST-GNN (all seeds × all horizons)")
print("=" * 70)
for h in [1, 2, 4]:
    for seed in [42, 1, 2]:
        ssuf = "" if seed == 42 else f"_s{seed}"
        name = f"stgnn_manila_cls_h{h}{ssuf}"
        if not already_done(name):
            try:
                train_stgnn(horizon=h, seed=seed)
            except Exception as e:
                print(f"  FAILED {name}: {e}")
                import traceback; traceback.print_exc()

# ------------------------------------------------------------------
# PHASE 2: Transformer multi-seed (seed 42 already done, add 1 & 2)
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("  PHASE 2: Transformer seed replicates")
print("=" * 70)
for data_name, data_path in [("manila_segments", MANILA_PARQUET),
                              ("sathorn", SATHORN_PARQUET),
                              ("sathorn_cctv", CCTV_PARQUET)]:
    df = load_long_parquet(data_path)
    task = "cls" if (df["cls"] >= 0).all() else "reg"
    out_dim = 5 if task == "cls" else 1
    for h in [1, 2, 4]:
        for seed in [42, 1, 2]:
            ssuf = "" if seed == 42 else f"_s{seed}"
            name = f"transformer_{data_name}_{task}_h{h}{ssuf}"
            if not already_done(name):
                try:
                    W = build_windows(df, history=24, horizon=h)
                    split = time_split(W["target_ts"])
                    model = TransformerForecaster(N_FEATURES, out_dim)
                    train_and_eval(model, W, split, task, name, OUT_ROOT, seed=seed)
                except Exception as e:
                    print(f"  FAILED {name}: {e}")
                    import traceback; traceback.print_exc()

# ------------------------------------------------------------------
# PHASE 3: GRU/LSTM/TCN multi-seed at h2/h4 on Manila
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("  PHASE 3: GRU/LSTM/TCN seed replicates (Manila h2, h4)")
print("=" * 70)
df_manila = load_long_parquet(MANILA_PARQUET)
for model_name in ["gru", "lstm", "tcn"]:
    for h in [2, 4]:
        for seed in [1, 2]:
            name = f"{model_name}_manila_segments_cls_h{h}_s{seed}"
            if not already_done(name):
                try:
                    W = build_windows(df_manila, history=24, horizon=h)
                    split = time_split(W["target_ts"])
                    model = build_model(model_name, N_FEATURES, 5)
                    train_and_eval(model, W, split, "cls", name, OUT_ROOT, seed=seed)
                except Exception as e:
                    print(f"  FAILED {name}: {e}")
                    import traceback; traceback.print_exc()

# ------------------------------------------------------------------
# PHASE 4: Transfer seed replicates (skip already-done CSVs)
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("  PHASE 4: Transfer seed replicates at h2/h4")
print("=" * 70)
transfer_jobs = []
for h, ckpt in [(2, CKPT_H2), (4, CKPT_H4)]:
    for seed in [42, 1, 2]:
        ssuf = "" if seed == 42 else f"_s{seed}"
        hsuf = f"_h{h}"
        for desc, data, prefix, out_name in [
            (f"Sathorn loop h{h} s{seed}", SATHORN_PARQUET, None,
             f"transfer_sathorn_all{hsuf}{ssuf}.csv"),
            (f"EDSA h{h} s{seed}", MANILA_PARQUET, "EDSA|",
             f"transfer_manila_segments_EDSA_{hsuf}{ssuf}.csv"),
            (f"CCTV h{h} s{seed}", CCTV_PARQUET, None,
             f"transfer_sathorn_cctv_all{hsuf}{ssuf}.csv"),
        ]:
            transfer_jobs.append({"desc": desc, "data": data, "ckpt": ckpt,
                                  "target_prefix": prefix, "horizon": h,
                                  "seed": seed, "out_name": out_name})

for i, job in enumerate(transfer_jobs):
    out_path = OUT_ROOT / job["out_name"]
    if out_path.exists():
        print(f"  SKIP {job['out_name']} (already done)")
        continue
    print(f"\n  [{i+1}/{len(transfer_jobs)}] {job['desc']}")
    try:
        result_df = transfer_kday(
            job["data"], job["ckpt"], job["target_prefix"],
            k_days=[1, 3, 7, 14, 28], horizon=job["horizon"], seed=job["seed"])
        result_df.to_csv(out_path, index=False)
        print(f"    Wrote: {out_path.name}")
    except Exception as e:
        print(f"    FAILED: {e}")
        import traceback; traceback.print_exc()

# ------------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print(f"  ALL DONE in {fmt_dur(time.time() - GRAND_START)}")
print("=" * 70)
print(f"\nResults in: {OUT_ROOT}")
for f in sorted(OUT_ROOT.rglob("*")):
    if f.is_file():
        print(f"  {f.relative_to(OUT_ROOT)}  ({f.stat().st_size:,} bytes)")
