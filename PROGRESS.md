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
