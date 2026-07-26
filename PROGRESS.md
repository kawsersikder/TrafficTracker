# TrafficTracker — progress and plan

**Course:** CSE498R, North South University
**Updated:** 2026-07-26 (phase 2 complete)

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

**Transfer works, and it is asymmetric.** Across three seeds, fine-tuning a
pre-trained model beats training from scratch in **69 of 70**
(target, horizon, k, seed) comparisons.

Mean ± std over 3 seeds, 30-min horizon, macro-F1:

| Local data | Manila → Bangkok loop-coil | | Within-Manila → EDSA | |
|---|---|---|---|---|
| | fine-tuned | scratch | fine-tuned | scratch |
| 1 day | **0.564 ± 0.001** | 0.209 ± 0.040 | **0.659 ± 0.003** | 0.287 ± 0.013 |
| 3 days | **0.554 ± 0.003** | 0.305 ± 0.009 | **0.665 ± 0.014** | 0.383 ± 0.008 |
| 7 days | **0.571 ± 0.004** | 0.532 ± 0.017 | **0.689 ± 0.001** | 0.541 ± 0.011 |
| 14 days | **0.574 ± 0.001** | 0.552 ± 0.005 | **0.794 ± 0.013** | 0.714 ± 0.021 |
| 28 days | **0.578 ± 0.001** | 0.564 ± 0.002 | **0.794 ± 0.019** | 0.752 ± 0.026 |

Manila → Bangkok CCTV (single seed at time of writing) reaches zero-shot 0.676
against 0.297 for from-scratch on one day of local data.

Two things to draw out of this:

1. **Pre-training stabilises as well as improves.** Fine-tuned standard
   deviations are 0.0007–0.019; from-scratch reaches 0.040. Pre-training does
   not merely raise the mean, it removes the variance that makes small-data
   training unreliable — arguably the more useful property for a city with
   weeks rather than years of data.

2. **Transfer is directional.** The reverse experiment (Bangkok CCTV → Manila
   EDSA, `transfer_manila_segments_EDSA__from_sathorn.csv`) reaches only
   onset-F1 0.064 zero-shot and fine-tuning wins just 2 of 5 k-values. Manila
   teaches Bangkok; Bangkok does not teach Manila.

   The plausible cause is structural: Manila is 298 segment series over nine
   arterials, while Sathorn CCTV is 13 lane detectors at a single intersection.
   The richer, more diverse source generalises; the narrow one does not. This
   sharpens the contribution from "transfer works" to **"transfer works when the
   source is structurally richer than the target"**, which also answers the
   practical question of which city to pre-train on.

   **Caveat to state in the paper:** the forward direction is scored in
   macro-F1 and the reverse in onset-F1, because the CCTV checkpoint is a
   regressor rather than a classifier. The asymmetry claim is qualitative;
   do not present the two numbers as directly comparable.

Error bars regenerate via `experiments/analysis/transfer_variance.py`
(`experiments/out/paper/transfer_variance.csv` / `.md`).

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

Artifacts in `experiments/out/dhaka/`: per-class detector table
(`yolo_per_class.csv`/`.md`) and three 300-dpi figures — mix by time of day,
the two-source comparison, and per-class detection for v8s vs v8m.

### 3.4 Dhaka vision detectors

| Model | mAP50 | mAP50-95 |
|### 3.5 Dhaka cross-dataset generalisation (new)

Two detectors, both YOLOv8s, both trained on Dhaka, evaluated on each other.
mAP50 over the 5 shared vehicle classes (`person` excluded — DhakaAI has no
pedestrian class, so it scores 0 there by construction):

| trained on ↓ / tested on → | TFP-BD | DhakaAI |
|### 3.6 Dhaka congestion index — derived from imagery (new)

Dhaka has no public congestion time series; every open Dhaka dataset is imagery.
We derive one. TFP-BD's annotated frames are consecutive video frames (~600 per
window, gap-free, ~5.9 s apart), so per-frame **visual occupancy** —
(sum of vehicle bbox area) / (frame area) — is itself a time series. This is the
camera analogue of loop-detector occupancy, which is exactly how Bangkok
Sathorn's CI is defined. Perspective distortion is handled by per-series robust
scaling (q05..q95), as already done for Bangkok CCTV lane volumes.

Result: `experiments/out/dhaka_ci.parquet` — **40 series, 23,678 steps**,
4 intersections x 2 lane types x 5 time-of-day windows. Mean CI 0.408, mean 9.6
vehicles and 1.6 pedestrians per frame.

Forecasting benchmark (one step = 5.9 s ahead):

| model | MAE | onset-F1 |
|---|---|---|
| **GRU** | **0.1889** | **0.059** |
| TCN | 0.1890 | 0.033 |
| LSTM | 0.1892 | 0.042 |
| xgboost | 0.2004 | 0.052 |
| persistence | 0.2094 | 0.000 |
| historical average | 0.2258 | 0.000 |

**This is the first dataset in the project where deep models beat every
baseline.** On Manila persistence won macro-F1 at 30 min; on Sathorn loop-coil
historical-average won at every horizon. On Dhaka all three architectures beat
persistence by ~10% MAE. The plausible reason is temporal resolution: at ~6 s
steps there is fine-grained structure that a copy-forward baseline cannot use,
whereas at 15–30 min steps congestion is so persistent that copying wins.

**Three caveats that must appear in the paper:**

1. **Horizon is not comparable across cities.** One step here is ~5.9 s; one
   step in Manila/Bangkok is 15–30 min. Do not place these MAE values in the
   same table as the other cities without saying so.
2. **`seasonal_naive` degenerates to `persistence`** (identical 0.2094) because a
   one-week lookback does not exist inside one-hour windows. Report it as
   not-applicable rather than as a baseline.
3. **Onset-F1 is low for everyone** (0.03–0.06 vs 411 onset events). Short-horizon
   occupancy onsets are near-unpredictable at this resolution; this is a
   negative result and should be stated as one.

Coverage is five one-hour weekday windows, not continuous days, so no daily or
weekly seasonality can be estimated. It supports characterisation and
short-horizon forecasting, not 15-minute-ahead cross-city transfer.

---|---|---|
| **TFP-BD** | **0.772** | 0.192 |
| **DhakaAI** | 0.457 | **0.771** |

Within-domain the two are indistinguishable (0.772 vs 0.771). Cross-dataset,
both collapse — but **asymmetrically**. TFP-BD → DhakaAI loses 75% of
performance; DhakaAI → TFP-BD loses 41%. The curated, multi-location source
transfers roughly 2.4× better than the fixed-4-camera source.

**This is the same structural principle as the congestion transfer asymmetry in
§3.1.** There, Manila (298 segments, 9 arterials) transferred to Bangkok
(13 detectors, 1 intersection) but not the reverse. Here, DhakaAI (curated
across many scenes) transfers to TFP-BD (4 fixed cameras) far better than the
reverse. Two independent modalities, same finding: **transfer works when the
source is structurally more diverse than the target** — diversity of the source,
not its size, is what buys generalisation.

Practical consequence worth stating: a detector trained on one city's curated
benchmark loses three-quarters of its accuracy on continuous footage from the
same city. Reporting only in-domain mAP substantially overstates readiness.

Artifacts: `experiments/out/dhaka/cross_dataset_matrix.{csv,md}`.

---|---|---|
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

### 4.2 Error bars — DONE

Phase 2 (`RUN_PHASE2.bat`) added seeds 1 and 2 to the three main h1 curves and
ran the reverse direction. Result: 69 of 70 comparisons favour fine-tuning, and
the reverse direction fails — see §3.1. Remaining optional extension: seed
replicates at h2/h4 and on the CCTV curve.

### 4.3 Missing model: ST-GNN

Proposal Phase 3 lists spatio-temporal graph networks among the RQ2 models, and
none was built. With 298 Manila segment series there is real spatial structure
being ignored. Survivable at ICCIT/TENCON with a limitations sentence; likely a
required revision at IEEE Access or ITSC. Budget about a week if aiming high.

### 4.4 Finish the paper

`PAPER_DRAFT.md` is v0.1 with real numbers and roughly a dozen `[TODO]` markers:
authors, venue, related work, conclusion, references, per-class detector table.

New material to fold in: the Sathorn native benchmark, the CCTV three-year
results, transfer at longer horizons, seed-variance bars, and the Dhaka
vehicle-mix section.

Venue decision is still open — ICCIT is the usual Bangladesh target; verify this
year's deadline. SIGSPATIAL UrbanAI workshop is the backup.

### 4.5 Release engineering

Add a LICENSE (MIT), push to public GitHub, and mint a Zenodo DOI for the
reconstructed Manila dataset so the paper can cite it.

### 4.6 Optional

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

---

## 6. Repository map — what every script does

Read this before touching anything; several scripts have non-obvious contracts.

### Data preparation
| Script | Produces | Notes |
|---|---|---|
| `experiments/manila/reconstruct_segments.py` | `out/manila_segments.parquet` | Positional reconstruction of the flattened MMDA feed; 298 series |
| `experiments/sathorn/prepare_sathorn.py` | `out/sathorn.parquet`, `out/sathorn_cctv.parquet` | Loop-coil uses `--mode occupancy`; CCTV needs `--date-from-filename --series-from-parent --per-file-resample` and a comma-separated `--value-col` |
| `experiments/dhaka_vision/tfpbd_to_ci.py` | `out/dhaka_ci.parquet` | Visual occupancy -> CI; 40 series |
| `experiments/dhaka_vision/voc_to_yolo.py` | `out/dhakaai_yolo/` | DhakaAI, 21 native classes |
| `experiments/dhaka_vision/tfpbd_to_yolo.py` | `out/tfpbd_yolo/`, `out/dhakaai_shared_yolo/` | `--stride 4` subsamples redundant video frames; val = held-out Location 4. `--dhakaai-remap` builds the shared-vocabulary DhakaAI copy |

### Models and evaluation
| Script | Notes |
|---|---|
| `experiments/train_forecaster.py` | GRU/LSTM/TCN. Run dir auto-named `<data>_<model>_<task>_h<N>`; override with `--out` (needed for seed replicates) |
| `experiments/baselines.py` | persistence, seasonal-naive, historical-average, XGBoost |
| `experiments/transfer_kday.py` | k-day transfer. **Output filename encodes data+target+source+horizon+seed** — this was patched three times after silent overwrites; do not simplify it |
| `experiments/analysis/aggregate_results.py` | Rebuilds `out/paper/results_table.md` and all transfer figures |
| `experiments/analysis/transfer_variance.py` | Collapses transfer CSVs into mean +/- std; reports the 69-of-70 count |
| `experiments/dhaka_vision/per_class_table.py` | Per-class detector table, parsed from logs (no GPU) |
| `experiments/dhaka_vision/make_figures.py` | The three Dhaka figures |

### Orchestrators
`RUN_OVERNIGHT.bat` (49 jobs, ~5 h), `RUN_PHASE2.bat` (transfer seeds + reverse
direction, ~20 min), `RUN_PHASE3.bat` (Dhaka cross-dataset vision, ~3 h).

All three share the same design: sentinels in `.run_state*/`, per-job logs in
`logs*/`, a live ledger in `RUN_PROGRESS*.md`. Re-running skips completed jobs.

**To force a job to re-run you must delete BOTH its sentinel and its log** — the
resume logic treats a log containing the success marker as proof of completion.

## 7. Traps that have already cost time

1. **PowerShell files must be pure ASCII.** PS 5.1 reads UTF-8 as ANSI; one
   em-dash broke parsing across an entire 229-line script.
2. **Never judge job success by exit code.** PS 5.1 raises `NativeCommandError`
   whenever a native command writes to stderr through a merged pipeline, and
   tqdm draws progress bars there. The orchestrators check the log for a
   per-job success marker instead.
3. **Log silence is not a hang.** `Tee-Object` buffers. A job that looked wedged
   for 7 minutes had already written its output. Check for the artifact first.
4. **Ultralytics nests output**: weights land in
   `runs/detect/runs/detect/<name>/weights/best.pt`, not `runs/detect/<name>/`.
5. **`workers=0` on every YOLO job**, or Windows raises
   `OSError [WinError 1455] paging file too small`.
6. **Keep `TEMP`/`TMP` on `F:\temp`.** The C: drive filled to 0.08 GB once and
   halted training mid-run.
7. **6 GB VRAM means sequential GPU jobs only.** Do not parallelise training.
8. **Filename collisions are the recurring failure mode here.** Three separate
   results were silently overwritten before the naming was made fully explicit.
   Any new experiment variant must encode every varying parameter in its output
   filename.
9. **A corrupt `.git/objects/info/commit-graph`** can make `git status` fail with
   `improper chunk offset`. It is a derived cache — delete it and git rebuilds.

## 8. Known gaps

- **No ST-GNN.** The proposal lists spatio-temporal graph networks among the RQ2
  models; none was built, despite 298 Manila segments having real spatial
  structure. Survivable at ICCIT/TENCON with a limitations sentence; likely a
  required revision at IEEE Access or ITSC. ~1 week of work.
- **No fresh Dhaka collection.** `dataset/dhaka_trafficktracker-own/` holds only
  a README. `configs/intersections.json` has one intersection (Shahbag Square)
  and there is no `.env`, so the capture pipeline has never run. Deliberate
  decision: collection needs ~5 weeks and the project had 7 days.
- **Transformer baseline** mentioned in the proposal was never built.
- **Seed replicates** exist only at h1 on three curves; h2/h4 and the CCTV curve
  are single-seed.
- **MeTS-10 Bangkok and Istanbul** are unobtainable; documented as limitations.
