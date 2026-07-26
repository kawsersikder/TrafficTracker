# Congestion Forecasting in Data-Scarce Southeast Asian Megacities: Rescued Archives, Harmonized Indices, and the Value of Cross-City Transfer

> DRAFT v0.1 — auto-assembled 2026-07-13 from the experiment artifacts in
> `experiments/out/paper/`. All numbers are real; prose needs human polish.
> `[TODO]` marks the spots that need a decision or pending result.

**Authors:** [TODO — team + supervisor]
**Course:** CSE498R, North South University
**Target venue:** [TODO — ICCIT / TENCON / SIGSPATIAL UrbanAI; verify deadline]

## Abstract (draft)

Cities that need congestion forecasting the most — Dhaka, Manila, Jakarta — have
the least data to train it. We study what can be done with what actually exists:
we *rescue* a defunct 2015–16 Metro Manila road-status feed (2.21 M rows,
reconstructing 298 segment-level ordinal series whose identities were lost in
flattening), *harmonize* it with a 2025–26 HERE jam-factor dataset for three
Indonesian cities and 110 days of Bangkok loop-coil occupancy into a common
Congestion Index (CI ∈ [0,1]), and *benchmark* classical and deep forecasters on
the result. Three findings: (1) at 30-minute horizons, congestion states are so
persistent (light-traffic dwell ≈ 18 h) that a persistence baseline beats every
learned model on macro-F1 — standard metrics mislead; (2) learned models earn
their keep at 60+ minutes, specifically on *congestion-onset* detection, where
all three deep architectures beat all baselines (best onset-F1 0.245 vs 0.206);
(3) transfer helps most exactly where data is scarcest: a model pre-trained on
Manila corridors reaches macro-F1 0.554 zero-shot on Bangkok loop-coil data,
versus 0.171 for a model trained from scratch on one day of local data — local
training needs roughly a week of data to catch up. We release the rescued
dataset, the harmonization pipeline, and TrafficTracker, an open-source
Google-Maps congestion capture instrument, to let any data-scarce city grow the
series we could not collect in one semester.

## 1. Introduction

- Dhaka/Manila/Jakarta motivation (adapt RESEARCH_PROPOSAL.md §1).
- The data gap and the domain gap; nearly all DL traffic benchmarks are
  sensor-rich Western/Chinese corpora (METR-LA, PeMS).
- Contributions:
  C1 data rescue + positional reconstruction of the Manila MMDA feed;
  C2 Unified Congestion Index across ordinal states / jam factors / occupancy;
  C3 forecasting benchmark with the persistence paradox + onset framing;
  C4 within- and cross-city k-day transfer curves;
  C5 open-source release (dataset + pipeline + TrafficTracker instrument).

## 2. Related work

[TODO — from proposal §3: arXiv:2011.02359 (Google Maps Dhaka), MeTS-10
(IEEE T-ITS 2023), Hadi et al. 2026 (Indonesian HERE pipeline, CEUS), TFP-BD
(Data in Brief 2025) as dataset-paper template, Mon et al. 2022 (Sathorn,
Scientific Data).]

## 3. Data

### 3.1 Rescued: Metro Manila road-status logs (2015-16)
MMDA official feed via `mmdatraffic.interaksyon.com`, logged by Oliveros before
the site went offline — the source no longer exists. 2,214,832 rows, Nov 2015 –
Jun 2016, ~30-min cadence, 9 arterials × direction, ordinal states
L/ML/M/MH/H (56.9 / 17.2 / 13.9 / 11.2 / 0.84 %). The flattened CSV dropped
segment names but preserved row order; per-snapshot row counts are constant per
road+direction (EDSA=39, C5=19, Quezon Ave=26 …), so segment identity is
recoverable positionally. Reconstruction keeps 2,183,712 rows (98.6 %) as
**298 segment-level series**; median per-series coverage 55–86 %
(`manila_road_ranking.csv`).

### 3.2 Fresh: Jakarta / Bandung / Semarang HERE jam factors (2025-26)
Hadi et al. Zenodo v2: 14,549 / 3,069 / 1,076 segments, 15-min collection
aggregated to 8 time-of-day periods, Mar 2025 – Feb 2026. Supports spatial and
period-level characterization (not sequence forecasting). CC-BY 4.0.

### 3.3 Bangkok Sathorn loop-coil occupancy (2016)
Chula-SSS release (Scientific Data 2022): 4 links, 5-second Volume/Occupancy,
resampled to 15 min → 41,540 observations, 27 May – 13 Sep 2016.
CI = occupancy/100 (mean 0.50). Used as the cross-city transfer target.
[Optional: 37-month CCTV volume series from the same release — extraction in
progress, may extend this to a long-range Bangkok series.]

### 3.4 Dhaka vehicle imagery
DhakaAI: 3,953 images, 21 native vehicle classes (VOC→YOLO conversion:
3,000 usable annotated images). Used for the heterogeneity side track (§7).

### 3.5 Unified Congestion Index
CI ∈ [0,1] per segment per bin: Manila ordinal → {0.1,0.3,0.5,0.7,0.9};
HERE jam factor → jf/10; occupancy → /100. **Congestion onset** := transition
into CI ≥ 0.6 (MH-or-worse) from below — the event practitioners care about.
Onset-F1 is computed only at timesteps whose previous ground-truth state was
below threshold.

## 4. RQ1 — How does congestion behave? (characterization)

- **Persistence:** Markov diagonal (30-min steps) 0.973/0.906/0.868/0.868/0.807
  for L/ML/M/MH/H → expected dwell 18.5 h / 5.3 h / 3.8 h / 3.8 h / 2.6 h
  (`fig_manila_markov.png`).
- **Corridor ranking:** EDSA is Manila's most congested corridor — 21.7 % of
  time in MH/H (C5 16.4 %, Commonwealth 1.7 %). Median EDSA jam episode 90 min,
  p90 = 5 h (`manila_episodes.csv`).
- **Rhythms:** weekday double peak with a dominant evening; weekend flattening
  (`fig_manila_daily_profile.png`, `fig_manila_week_heatmap.png`).
- **Cross-city, aligned on 8 periods** (`cross_city_periods.csv`): mean CI
  evening_peak — Manila 0.366, Jakarta 0.343, Bandung 0.322, Semarang 0.274:
  a clean city-size gradient. **Peak spreading:** Manila's evening congestion
  *keeps rising after the peak period* (19–22 h CI 0.379 > 16–19 h 0.366),
  while all three Indonesian cities decay after their peak — evidence of
  saturation-driven peak spreading in Manila. (Vintage caveat: Manila 2015-16
  vs Indonesia 2025-26 — we compare *shapes*, not levels; §9.)

## 5. RQ2 — Forecasting benchmark (Manila, 298 series)

Setup: 12 h history (24 steps) → predict the state 30/60/120 min ahead;
chronological 70/15/15 split; class-weighted cross-entropy; details §Appendix.
Full table: `results_table.md`; decay figure: `fig_horizon_decay.png`.

**The persistence paradox (30 min):** persistence wins macro-F1 (0.848 vs GRU
0.831, LSTM 0.830, TCN 0.821, XGBoost 0.785) yet scores onset-F1 = 0.000 by
construction. Any evaluation that ignores onsets would conclude "no model
needed."

**Learned models win onsets at 60+ min:** onset-F1 at 60 min — LSTM 0.245,
GRU 0.237, TCN 0.213 vs best baseline 0.206 (seasonal-naïve); every deep model
beats every baseline. At 120 min, GRU reaches onset-recall 0.456 (persistence
0.215) — the model sees twice as many upcoming jams, at precision 0.11 vs 0.12
[TODO: precision-recall trade-off sentence; consider decision-threshold sweep].

## 6. RQ3 — Transfer: how many days of local data do you need?

Protocol: pre-train GRU on source; fine-tune on the LAST k days of the target's
train period; evaluate on the target's fixed final-15 % test split; compare with
an identical model trained from scratch on the same k days. Figures:
`fig_transfer_manila_segments_EDSA_.png`, `fig_transfer_sathorn_all.png`.

- **Within-city (non-EDSA → EDSA):** zero-shot macro-F1 0.680; scratch needs
  ~14–28 days to approach it (0.693 at k=14); fine-tuning wins at every k
  (0.803 at k=28 vs 0.737 scratch).
- **Cross-city (Manila → Bangkok loop-coil):** zero-shot 0.554 vs scratch@1-day
  0.171; scratch catches up around k=7 (0.547 vs finetune 0.570); fine-tune
  stays ahead through k=28 (0.578). Onset-F1 shows the same shape (finetune
  0.490 at k=3 vs zero-shot 0.039).
- **Headline sentence:** *cross-city transfer is worth roughly one week of
  local sensing; within-city transfer roughly two.*
- Robustness: a second within-city target (C5) reproduces the macro-F1 gap
  (zero-shot 0.875 vs scratch ≤ 0.32 for k ≤ 14) but its test window contains
  almost no onset events — reported in the appendix, not the main text.

## 7. Vision side track — why Dhaka is different

YOLOv8n fine-tuned on DhakaAI (50 epochs, 640 px): 0.431 mAP50 overall;
per-class: three-wheelers/CNG 0.812, car 0.739, army vehicle 0.995 [TODO:
full per-class table from run dir]. YOLOv8s comparison: [TODO — training in
progress, ~3 h]. Use: vehicle-mix vector (rickshaw/CNG share) as the
heterogeneity covariate motivating why models must eventually be *trained* on
Dhaka-like traffic, not just transferred to it.

## 8. Released artifacts

1. Reconstructed Manila segment dataset (parquet + reconstruction script +
   validation report) → Zenodo DOI [TODO].
2. Harmonization + experiment pipeline (this repo, `experiments/`).
3. TrafficTracker capture instrument (Playwright + OpenCV scoring + dashboard)
   → public GitHub [TODO: strip .env, add MIT license].
4. [Optional] 2026 Dhaka+Manila HERE collection via the open-source pipeline —
   status: [TODO: started / not started].

## 9. Limitations (state plainly)

- Only ~8 % of raw Manila rows form gap-free 12 h windows; windows never bridge
  gaps; per-series coverage 55–86 % reported per road.
- Segment identity is positional; names lost by the source (Wayback recovery =
  future work).
- Manila data is 2015–16 (archival rescue, not a contemporary snapshot);
  cross-city *shape* comparisons only. Decade-apart temporal transfer against
  fresh 2026 Manila HERE data is the natural follow-up.
- H-class support is 0.84 % of rows; onset metrics rest on 503–607 test events
  per configuration.
- Sathorn CI uses occupancy (defensible); its 4 detectors cover one
  intersection, not a network.
- MeTS-10 Bangkok (floating-car, city-scale) could not be obtained in time;
  the Bangkok transfer target is sensor-based instead.

## 10. Conclusion

[TODO — one paragraph: the three findings + the release; data-scarce cities can
bootstrap forecasting from other cities' data for their first weeks, and the
open instrument closes the loop.]

## References

[TODO — BibTeX exists in each dataset README: benjiao repo; Hadi et al. 2026
(CEUS + Zenodo DOI 10.5281/zenodo.18650759); Mon et al. 2022 Sci Data;
DhakaAI Kaggle/Dataverse; TFP-BD 2025; MeTS-10 2023; arXiv:2011.02359.]

## Appendix A — full results

Paste `experiments/out/paper/results_table.md` (auto-generated) + C5 transfer
CSV. Reproduction: every table/figure regenerates from
`experiments/analysis/aggregate_results.py` and
`experiments/analysis/characterize_manila.py`.
