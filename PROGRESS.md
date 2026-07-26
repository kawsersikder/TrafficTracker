# TrafficTracker — progress and plan

**Course:** CSE498R, North South University
**Updated:** 2026-07-26

Single status document for the project. `RESEARCH_PROPOSAL.md` remains the
motivation and is still worth reading, but its city plan has drifted — see
§2. `PAPER_DRAFT.md` holds the current paper text.

---

## 1. Where we are

Phases 1–4 of the proposal are complete. Phase 5 (Dhaka dataset release) has
not started, and that is the one thing on the critical path.

| Track | State |
|---|---|
| Manila (rescued MMDA logs) | Complete — characterisation, full benchmark, within-city transfer |
| Jakarta / Bandung / Semarang | Complete and permanently capped: source is aggregated to 8 periods, so forecasting is impossible by construction |
| Bangkok Sathorn loop-coil | Complete — transfer plus a native benchmark |
| Bangkok Sathorn CCTV (2016–2019) | Complete — new 3-year series, built and benchmarked |
| Bangkok MeTS-10 (city-scale) | Abandoned — only the code repo exists, no data |
| Istanbul (planned control city) | Abandoned — depended on MeTS-10 |
| Dhaka — vision | Strong — three detectors trained, plus a vehicle-mix characterisation |
| Dhaka — congestion time series | **Not started. This is the gap.** |

---

## 2. The framing issue, stated plainly

The proposal's headline was **Bangkok → Dhaka** transfer. That is not what we
have. MeTS-10 Bangkok could not be obtained, and no Dhaka congestion series has
been collected yet.

What we actually have is **Manila → Bangkok** transfer, using rescued 2015–16
Manila logs against 2016–2019 Bangkok sensor data. That is a real and
defensible result, but it is a different paper, and Dhaka is currently a vision
side-track rather than the transfer target.

Nothing in the repo should claim Dhaka transfer results. They do not exist yet.

---

## 3. Results so far

### 3.1 Transfer — the headline (RQ3)

Fine-tuning a pre-trained model beats training from scratch at **every** amount
of local data, on **every** target, at **every** horizon — 25 of 25 comparisons.

Manila → Bangkok CCTV (30-min horizon), macro-F1:

| Local data | Fine-tuned | From scratch |
|---|---|---|
| zero-shot | 0.676 | — |
| 1 day | 0.691 | 0.297 |
| 3 days | 0.699 | 0.566 |
| 7 days | 0.709 | 0.642 |
| 14 days | 0.710 | 0.685 |
| 28 days | 0.712 | 0.703 |

A model that has never seen Bangkok (0.676) beats a locally-trained model given
three days of Bangkok data. Local training needs roughly four weeks to catch up.
The same shape holds for the loop-coil target and for within-Manila transfer at
60- and 120-minute horizons.

### 3.2 Forecasting benchmark (RQ2)

Manila, 298 segment series, GRU/LSTM/TCN against persistence, seasonal-naïve,
historical-average and XGBoost at 30/60/120 minutes.

Two honest findings to keep in the paper:

- **The persistence paradox.** At 30 minutes persistence wins macro-F1 (0.848)
  yet scores onset-F1 of exactly 0 by construction. Any evaluation ignoring
  onsets would wrongly conclude no model is needed. Deep models earn their keep
  at 60+ minutes on onset detection.
- **Historical average wins on Sathorn loop-coil** at every horizon (0.675 vs
  0.637 best deep model). Occupancy at four detectors on a single intersection
  is strongly diurnal, so a time-of-day average is genuinely hard to beat. This
  is a second instance of the same lesson and should be reported, not buried.

**Seed variance:** across three seeds, macro-F1 standard deviation is 0.002–0.013,
so model rankings are real rather than noise.

### 3.3 Dhaka vehicle mix (new)

From TFP-BD: **265,698 annotated objects**, 4 Dhaka intersections, 5 time-of-day
windows.

**46.1% of vehicles are rickshaws, CNGs, motorbikes or bicycles** — modes that
effectively do not exist in METR-LA or PeMS. Rickshaws alone (26.5%) nearly match
cars (27.8%). DhakaAI independently gives 40.4% on the same vocabulary.

The two sources disagree on car share (38.4% vs 27.8%): DhakaAI is a curated
detection set while TFP-BD is continuous frames at fixed intersections, so TFP-BD
better reflects road occupancy. DhakaAI has no pedestrian class, so the fair
comparison renormalises both on vehicles only.

Time-of-day signal: motorbike share rises from 3.8% at midday to 8.9% at
20:00–21:00; buses peak at midday (25.1%); roadway pedestrians fall from 19.8%
in the morning to 6.8% at midday.

This is the quantitative form of the heterogeneity argument the paper makes.

### 3.4 Dhaka vision detectors

| Model | mAP50 | mAP50-95 |
|---|---|---|
| YOLOv8n | 0.433 | 0.290 |
| YOLOv8s | 0.541 | 0.364 |
| YOLOv8m | 0.613 | 0.418 |

A clean scaling curve. Three-wheelers/CNG reach 0.857 mAP50 — strong on exactly
the classes that carry the heterogeneity argument.

---

## 4. What comes next

### 4.1 Start Dhaka collection — the only time-sensitive item

Nothing has been captured; `dataset/dhaka_trafficktracker-own/` holds only a
README. RQ4 (the dataset paper) and any genuine Dhaka transfer result depend
entirely on this, and it needs roughly five weeks of wall-clock time. The plan
is in `experiments/collect/COLLECTION_PLAN.md` — HERE API, free key, the same
pipeline that produced the Jakarta Zenodo dataset.

Every day of delay is one less day of data. Nothing else on this list competes.

### 4.2 Finish the paper

`PAPER_DRAFT.md` is v0.1 with real numbers and roughly a dozen `[TODO]` markers:
authors, venue, related work, conclusion, references, per-class detector table.

New material to fold in: the Sathorn native benchmark, the CCTV three-year
results, transfer at longer horizons, seed-variance bars, and the Dhaka
vehicle-mix section.

Venue decision is still open — ICCIT is the usual Bangladesh target; verify this
year's deadline. SIGSPATIAL UrbanAI workshop is the backup.

### 4.3 Release engineering

Add a LICENSE (MIT), push to public GitHub, and mint a Zenodo DOI for the
reconstructed Manila dataset so the paper can cite it.

### 4.4 Optional

Poribohon-BD was never downloaded and largely overlaps DhakaAI. Kolkata was
formally dropped on 2026-07-13 and should not be revived. Figures for the Dhaka
mix (stacked bar by time-of-day, two-source comparison) would strengthen a
presentation.

---

## 5. Reproducing the experiments

Everything runs through one resumable orchestrator:

```
cd /d F:\CSE498R\TrafficTracker && RUN_OVERNIGHT.bat
```

49 jobs, sequential, roughly 5 hours on an RTX 2060. Completed jobs are skipped
on re-run and YOLO jobs resume mid-training, so an interrupted run costs at most
the job that was in flight.

Environment notes learned the hard way:

- Use `.venv\Scripts\python.exe` (torch 2.13+cu126, CUDA verified).
- **`workers=0` on every YOLO job**, or Windows raises
  `OSError [WinError 1455] paging file too small`.
- Keep `TEMP`/`TMP` on `F:\temp` — the C: drive filled to 0.08 GB in an earlier
  session and halted training.
- 6 GB of VRAM means sequential GPU jobs only; do not parallelise training.
- Write PowerShell files as pure ASCII. PowerShell 5.1 reads UTF-8 as ANSI and a
  single em-dash will break parsing across the whole file.
- Do not judge job success by exit code: PowerShell 5.1 raises
  `NativeCommandError` when a native command writes to stderr through a merged
  pipeline, and tqdm draws progress bars there. The orchestrator checks the log
  for a per-job success marker instead.
- Log silence does not mean a hang — `Tee-Object` buffers. Check whether the
  output artifact exists before concluding a job is stuck.

Generated outputs (`experiments/out/`, `experiments/runs/`, `runs/`, model
weights) are deliberately untracked; they rebuild from the orchestrator.
