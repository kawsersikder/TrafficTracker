# TrafficTracker

> **CSE498R — Computer Science Research Project**
> North South University · Department of Electrical and Computer Engineering

**Cross-city transfer learning for traffic congestion forecasting in South and
Southeast Asian megacities.**

Cities that most need congestion forecasting — Dhaka, Manila, Jakarta — have the
least data to train it on. Nearly all published deep-learning traffic research is
built on sensor-rich Western and Chinese benchmarks (METR-LA, PeMS) with
homogeneous, lane-disciplined traffic. This project asks whether a model trained
where data *is* available transfers to a city where it is not, and how many days
of local data a data-scarce city needs before local training wins.

Current status, results and roadmap: **[PROGRESS.md](PROGRESS.md)**

---

## Headline result

**Transfer works, and it is directional.** Across three seeds, fine-tuning a
pre-trained forecaster beats training from scratch in **69 of 70**
(target, horizon, k, seed) comparisons.

Mean ± std over 3 seeds, Manila → Bangkok loop-coil, 30-minute horizon, macro-F1:

| Local data | Fine-tuned | From scratch |
|---|---|---|
| 1 day | **0.564 ± 0.001** | 0.209 ± 0.040 |
| 7 days | **0.571 ± 0.004** | 0.532 ± 0.017 |
| 28 days | **0.578 ± 0.001** | 0.564 ± 0.002 |

Pre-training does not only raise the mean — it removes variance. Fine-tuned runs
vary by ≤0.02 while from-scratch runs reach 0.04, which matters more than the
average when a city has weeks rather than years of data.

Reversing the direction fails: Bangkok CCTV → Manila reaches only onset-F1 0.064
zero-shot. Manila (298 segments, nine arterials) teaches Bangkok (13 detectors,
one intersection), but not the other way round — so transfer appears to require a
source that is **structurally richer than the target**.

Two findings we report rather than bury: at 30-minute horizons a **persistence
baseline beats every learned model** on macro-F1 while scoring exactly zero on
congestion-onset detection — standard metrics mislead. And on the Sathorn
loop-coil data a **historical-average baseline wins at every horizon**, because
occupancy at four detectors on one intersection is strongly diurnal.

---

## What is in this repository

The project has two halves.

**`experiments/` — the research pipeline.** Data harmonisation into a Unified
Congestion Index, characterisation, forecasting benchmarks (GRU / LSTM / TCN
against persistence, seasonal-naïve, historical-average and XGBoost), k-day
transfer curves, and the Dhaka vision track.

**Everything else — the capture instrument.** A web application that screenshots
the Google Maps traffic layer, lets researchers draw road-arm geometries on it,
and scores congestion per arm with OpenCV. This is how the Dhaka dataset will be
collected, and it is released as a reusable instrument for any city without
sensor infrastructure.

```
experiments/
  common/          shared data loading, models, metrics
  manila/          MMDA feed reconstruction
  sathorn/         Bangkok loop-coil + CCTV preparation
  jakarta/         characterisation
  dhaka_vision/    DhakaAI conversion, TFP-BD vehicle-mix analysis
  analysis/        table and figure generation
  train_forecaster.py, baselines.py, transfer_kday.py
  collect/         plan for fresh 2026 collection

backend/           Express + Prisma API
browser/           Playwright capture
vision/            OpenCV congestion scoring
public/            dashboard UI
dataset/           one README per source: provenance, license, citation
```

Generated artifacts — `experiments/out/`, `experiments/runs/`, `runs/`, model
weights — are deliberately untracked. They rebuild from the orchestrator below.

---

## Data sources

| City | Source | Role |
|---|---|---|
| Manila | MMDA road-status logs, 2015–16 (rescued; the source site is gone) | Transfer source, 298 segment series |
| Bangkok | Sathorn loop-coil occupancy, 2016 | Cross-city transfer target |
| Bangkok | Sathorn CCTV lane volumes, 2016–2019 | Second transfer target, 3-year series |
| Jakarta / Bandung / Semarang | HERE jam factors, 2025–26 (Zenodo) | Cross-city characterisation |
| Dhaka | DhakaAI, TFP-BD | Vehicle heterogeneity |
| Dhaka | TrafficTracker (ours) | **Planned** — not yet collected |

Per-source provenance, licensing and citation details are in each
`dataset/*/README.md`.

Raw data is not committed. Each README explains how to obtain it.

---

## Reproducing the experiments

All experiments run through one resumable orchestrator:

```
cd /d F:\CSE498R\TrafficTracker && RUN_OVERNIGHT.bat
```

49 jobs, sequential, roughly 5 hours on an RTX 2060. Completed jobs are skipped
on re-run and vision jobs resume mid-training, so an interrupted run costs at
most the job that was in flight.

Individual pieces:

```
python experiments/train_forecaster.py --data experiments/out/manila_segments.parquet --model gru --horizon 2
python experiments/baselines.py        --data experiments/out/manila_segments.parquet --horizon 1
python experiments/transfer_kday.py    --data experiments/out/sathorn.parquet \
       --pretrained experiments/runs/manila_segments_gru_cls_h1/best.pt --k-days 1 3 7 14 28
python experiments/analysis/aggregate_results.py
```

Environment notes, including several Windows-specific traps worth knowing before
you run anything, are in [PROGRESS.md](PROGRESS.md) §5.

---

## Running the capture instrument

Requires Node ≥ 20, Python ≥ 3.10, PostgreSQL ≥ 14.

```
npm install
python3 -m venv .venv && .venv\Scripts\activate     # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
npx playwright install chromium

cp .env.example .env          # set DATABASE_URL and JWT_SECRET
npx prisma generate && npx prisma db push
npx tsx scripts/create-user.ts admin@example.com yourpassword ADMIN "Your Name"

npm run dev                   # http://localhost:3000
```

`DATABASE_URL` is a PostgreSQL connection string; `JWT_SECRET` should be a long
random value (`openssl rand -hex 32`). Never commit `.env` — it is gitignored.

The vision pipeline is invoked by the backend as a subprocess and prints JSON to
stdout with per-arm congestion scores, dominant signal colours, red-queue
fractions and spatial colour profiles:

```
.venv/bin/python vision/extract_traffic.py <intersection_id> <snapshot.png> <N>-arm <base64_config> <out.png>
```

Roles are `ADMIN` (draw arms, run analysis, edit weekly updates) and `TEACHER`
(read-only dashboard). API routes are listed in `backend/server.ts`.

---

## Status

Manila, Jakarta and both Bangkok datasets are complete. The Dhaka vision track is
complete (YOLOv8n/s/m, 0.433 → 0.613 mAP50, plus a 265,698-object vehicle-mix
characterisation showing 46% of Dhaka vehicles are rickshaws, CNGs, motorbikes or
bicycles).

**Dhaka congestion collection has not started**, and it is the critical path for
the dataset contribution. See [PROGRESS.md](PROGRESS.md) §4.

---

## License

Not yet licensed for redistribution. An open license is planned alongside the
dataset release; until then, all rights reserved © 2026.

Third-party datasets retain their original licenses — see each
`dataset/*/README.md`.
