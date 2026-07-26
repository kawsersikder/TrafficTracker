# Experiment Suite — Models, Training Commands & Time Estimates

> **CSE498R — Cross-City Traffic Congestion Study**
> Target machine: **Ryzen 5 2600 (6C/12T) · 24 GB RAM · RTX 2060 6 GB** · Windows 10
> All commands are PowerShell, run from the repo root `F:\CSE498R\TrafficTracker`.

---

## 1. One-time setup

```powershell
# Python deps (pandas/xgboost/sklearn already present; this adds the rest)
pip install -r experiments\requirements-ml.txt

# PyTorch with CUDA (RTX 2060 = Turing, fully supported)  — ~2.5 GB download
pip install torch --index-url https://download.pytorch.org/whl/cu126

# verify the GPU is seen
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"

# 7-Zip — needed only for the Bangkok Sathorn .rar archives
winget install 7zip.7zip
```

**Estimated setup time: 15–25 min** (mostly the torch download).

---

## 2. Data model

Every prepare script emits one **canonical long-format parquet** in `experiments/out/`:

| column | type | meaning |
|---|---|---|
| `ts` | datetime | observation time (city-local, tz-naive) |
| `series_id` | str | one series = one road segment / detector / junction |
| `y` | float32 | **Congestion Index (CI) ∈ [0,1]** — the harmonized target |
| `cls` | int8 | ordinal class 0–4 (`L ML M MH H`), or −1 → regression task |

CI harmonization (the paper's Phase-1 contribution):
Manila ordinal states → {0.1, 0.3, 0.5, 0.7, 0.9} · HERE jam factor → `jf/10` ·
occupancy → `/100` · volume/speed → robust per-series scaling (q05–q95, speed inverted).
**Congestion onset** = transition into MH-or-worse (CI ≥ 0.6) — the headline metric is onset-F1.

The trainer, baselines, and transfer scripts all consume this format, so every
dataset gets identical treatment.

---

## 3. Datasets → commands

### 3.1 Manila road-status logs (primary forecasting dataset) ✅ verified

Reconstructs **298 segment-level series** (2.18 M observations kept = 98.6 %,
Nov 2015 – Jun 2016, ~30-min cadence) from the flattened benjiao CSV using the
stable positional row order. Class shares: L 56.9 % · ML 17.2 % · M 13.9 % ·
MH 11.2 % · **H 0.84 %** (heavy imbalance → class-weighted loss + onset-F1).

```powershell
python experiments\manila\reconstruct_segments.py          # ~2 min  (verified)
```

### 3.2 fedesoriano 4-junction set (smoke test only — never cite as a city) ✅ verified

```powershell
python experiments\prototyping\prepare_fedesoriano.py      # ~10 s   (verified)
python experiments\baselines.py --data experiments\out\fedesoriano.parquet   # ~1 min (verified)
python experiments\train_forecaster.py --data experiments\out\fedesoriano.parquet --model gru --epochs 5   # ~2 min
```

Use this to prove the whole loop works before pointing it at Manila.

### 3.3 Jakarta / Bandung / Semarang (spatial characterization — RQ1)

v2 is aggregated to 8 time-of-day periods → **characterization only, no forecasting**.

```powershell
python experiments\jakarta\characterize.py                  # ~3–6 min
# outputs: experiments\out\jakarta\city_period_summary.csv + 2 paper figures
```

### 3.4 Bangkok Sathorn (long Bangkok series — transfer source until MeTS-10 lands)

```powershell
# extract archives first (one-time):
7z x "dataset\bangkok_sathorn-intersection\raw\loopcoil.rar"    "-odataset\bangkok_sathorn-intersection\extracted\loopcoil"
7z x "dataset\bangkok_sathorn-intersection\raw\cctv-camera.rar" "-odataset\bangkok_sathorn-intersection\extracted\cctv"

# see what the CSVs look like, then build with the right column names:
python experiments\sathorn\prepare_sathorn.py --inspect
python experiments\sathorn\prepare_sathorn.py `
  --csv "dataset/bangkok_sathorn-intersection/extracted/loopcoil/**/*.csv" `
  --time-col <TIME_COL> --value-col <OCCUPANCY_COL> --series-col <DETECTOR_COL> `
  --mode occupancy --out experiments\out\sathorn.parquet
```

(The archives' internal schemas aren't documented until extracted — `--inspect`
prints every CSV's columns so you can fill in the three `<..>` names.)

### 3.5 DhakaAI vehicle detection (vision side track — heterogeneity covariate)

```powershell
python experiments\dhaka_vision\voc_to_yolo.py              # ~5 min, copies ~1.4 GB
yolo detect train data=experiments/out/dhakaai_yolo/dataset.yaml model=yolov8n.pt `
    epochs=50 imgsz=640 batch=16 workers=2 device=0
# results land in runs\detect\train\ (weights, PR curves, confusion matrix)
```

---

## 4. Models

| Model | Where | Task | Architecture | Why it's in the paper |
|---|---|---|---|---|
| Persistence | `baselines.py` | both | last observed value | sanity floor; onset-F1 = 0 by construction |
| Seasonal-naïve | `baselines.py` | both | value 1 week earlier | strong calendar baseline |
| Historical average | `baselines.py` | both | mean per series×dow×slot | the "no ML needed?" test |
| ARIMA (sampled) | `baselines.py --arima N` | reg | SARIMAX(2,0,2) | classical reference |
| XGBoost | `baselines.py` | both | 400 trees on 24 lags + calendar | strong tabular baseline (CPU) |
| **LSTM** | `train_forecaster.py` | both | 2×128 + MLP head | deep sequence baseline |
| **GRU** | `train_forecaster.py` | both | 2×128 + MLP head | usually best of the RNNs |
| **TCN** | `train_forecaster.py` | both | 4 dilated causal blocks ×64ch (RF=61 steps) | convolutional alternative, fastest |
| YOLOv8n/s | ultralytics CLI | detection | — | Dhaka vehicle-mix heterogeneity |

Shared setup for the deep models: input = 24 steps (12 h at 30-min cadence) of
`[CI, sin/cos time-of-day, sin/cos day-of-week]`; horizon 1/2/4 steps
(30 min / 1 h / 2 h ahead); class-weighted cross-entropy (√-inverse-frequency,
so H isn't ignored) or SmoothL1 for regression; AdamW, AMP (fp16), batch 512,
early stopping on val loss (patience 5); chronological 70/15/15 split — never
random, to avoid temporal leakage.

---

## 5. Training commands (Manila = the paper's main table)

```powershell
# baselines — one command, writes experiments\runs\manila_segments_baselines_h1.json
python experiments\baselines.py --data experiments\out\manila_segments.parquet

# deep models, 30-min-ahead
python experiments\train_forecaster.py --data experiments\out\manila_segments.parquet --model gru
python experiments\train_forecaster.py --data experiments\out\manila_segments.parquet --model lstm
python experiments\train_forecaster.py --data experiments\out\manila_segments.parquet --model tcn

# longer horizons (1 h and 2 h ahead)
python experiments\train_forecaster.py --data experiments\out\manila_segments.parquet --model gru --horizon 2
python experiments\train_forecaster.py --data experiments\out\manila_segments.parquet --model gru --horizon 4
```

Each run saves `experiments\runs\<name>\best.pt` + `metrics.json`
(accuracy, macro-F1, per-class F1, **onset precision/recall/F1**).

### Headline experiment — k-day transfer curves

```powershell
# within-Manila: pre-train on all non-EDSA corridors, transfer to EDSA
python experiments\transfer_kday.py --data experiments\out\manila_segments.parquet `
    --target-prefix "EDSA|" --k-days 1 3 7 14 28

# cross-city: Manila-pretrained checkpoint -> Sathorn Bangkok (after 3.4)
python experiments\transfer_kday.py --data experiments\out\sathorn.parquet `
    --pretrained experiments\runs\manila_segments_gru_cls_h1\best.pt --k-days 1 3 7 14 28
```

Output CSV has one row per (k, finetune|scratch) with all metrics — plot
`onset_f1` vs `k_days` for both modes; the crossover is the paper's headline figure.

---

## 6. Estimated wall-clock times on YOUR hardware

RTX 2060 6 GB + Ryzen 5 2600, ~2.1 M Manila windows (X tensor ≈ 1.0 GB RAM — fine in 24 GB).

| Step | Estimate | Notes |
|---|---|---|
| Manila reconstruction | **~2 min** ✅ measured | pandas, single pass |
| Window building (per run) | 1–3 min | included in each training run |
| Baselines: persistence / seasonal / hist-avg | 2–4 min | CPU |
| Baselines: XGBoost (Manila, 5-class) | 8–20 min | `hist` method, all 12 threads |
| ARIMA sample (`--arima 10`) | 5–10 min | reference numbers only |
| GRU or LSTM, one horizon | **25–60 min** | ~1.5–3 min/epoch, early stop ≈ 10–20 epochs |
| TCN, one horizon | 15–40 min | convolutions parallelize better |
| Full grid: 3 models × 3 horizons | **4–8 h → run overnight** | or prune to GRU+TCN × 2 horizons ≈ 2–3 h |
| Transfer curves (5 k × 2 modes + pretrain) | 1.5–3 h | fine-tune sets are small; pretrain dominates |
| Jakarta characterization | 3–6 min | I/O-bound |
| DhakaAI VOC→YOLO conversion | ~5 min | copies ~1.4 GB once |
| YOLOv8**n**, 50 ep, 640 px, batch 16 | **~1–1.5 h** | ≈ 4.5 GB VRAM |
| YOLOv8**s**, 50 ep, 640 px, batch 8 | ~2.5–4 h | ≈ 5.5 GB VRAM — near the 6 GB limit |
| Sathorn prep + GRU | ~10–20 min total | few series after 15-min resample |

**Total compute for every paper table/figure: roughly 2 overnight runs.**
If no GPU is available the deep models fall back to CPU automatically — budget ~8–15× longer (Manila GRU ≈ 6–10 h); XGBoost and all baselines are CPU-native and unaffected.

### 6 GB VRAM survival guide

- Forecasters at batch 512 use **< 1 GB** — VRAM is never the constraint there; you can raise `--batch 2048` for a small speedup.
- YOLO is the only VRAM-hungry job: if you hit CUDA-OOM use `batch=8` (v8n) / `batch=4` (v8s) or `imgsz=512`. Keep `workers=2` — the 2600 has only 6 cores and Windows dataloader workers are expensive.
- Don't train YOLO and a forecaster simultaneously.

---

## 7. What feeds which part of the paper

| Paper section | Command(s) | Artifact |
|---|---|---|
| §Data — reconstruction & cleaning | `manila\reconstruct_segments.py` | `manila_reconstruction_report.json` (drop counts, class shares) |
| §RQ1 characterization | `jakarta\characterize.py` (+ Manila stats from the parquet) | summary CSV, heatmap, daily-profile figure |
| §RQ2 forecasting benchmark | `baselines.py` + 3× `train_forecaster.py` × horizons | main results table (macro-F1 / onset-F1 / MAE) |
| §RQ3 transfer / data scarcity | `transfer_kday.py` (within-Manila + Manila→Sathorn) | **headline crossover figure** |
| §Vision heterogeneity | `voc_to_yolo.py` + `yolo detect train` | mAP table + vehicle-mix distribution |
| §Instrument release | the TrafficTracker app itself | open-source GitHub repo |

## 8. Known constraints (state these in the paper)

- Manila segment **names** were lost in flattening; identity is positional
  (validated: counts constant per road+direction). Recovering names from the
  Wayback Machine of `mmdatraffic.interaksyon.com` is optional future work.
- Jakarta v2 is period-aggregated → characterization only. Zenodo **v1** may
  contain the raw 15-min series — check/email before the camera-ready.
- MMDA incidents (2018+) don't overlap the benjiao logs (2015–16) → used for
  event-frequency characterization only, never joined.
- Volume→CI scaling for Sathorn is a proxy; prefer loop-coil **occupancy**.
- MeTS-10 Bangkok still unfetched (HERE developer account) — if it lands,
  re-run `transfer_kday.py --pretrained <mets10 ckpt>` and the paper upgrades.
