# Phase 2 — Finish Plan (target: ~2 weeks, by 2026-07-27)

All experiments below run inside the F: venv:
`F:\CSE498R\TrafficTracker\.venv\Scripts\python.exe` (torch 2.13 cu126, CUDA OK).

## What is DONE and where it lives

| Artifact | Path | Paper use |
|---|---|---|
| Results table (all models × horizons) | `experiments/out/paper/results_table.md` + `results_all.csv` | RQ2 main table |
| Transfer curve figure (EDSA) | `experiments/out/paper/fig_transfer_manila_segments_EDSA_.png` | **headline figure (RQ3)** |
| Horizon decay figure | `experiments/out/paper/fig_horizon_decay.png` | RQ2 |
| Markov transitions + dwell times | `manila_markov_transitions.csv`, `fig_manila_markov.png` | RQ1 |
| Daily profile + weekly heatmap | `fig_manila_daily_profile.png`, `fig_manila_week_heatmap.png` | RQ1 |
| Road ranking + coverage | `manila_road_ranking.csv` | RQ1 + data section honesty |
| Episode durations | `manila_episodes.csv` | RQ1 |
| Cross-city period table | `cross_city_periods.csv` | RQ1 cross-city |
| Jakarta figures | `experiments/out/jakarta/` | RQ1 |
| YOLOv8n weights (0.431 mAP50) | `runs/detect/experiments/runs/dhakaai_yolov8n-2` | vision/heterogeneity section |
| Reconstruction report | `manila_reconstruction_report.json` | data section |

## The paper's three key findings (already supported by numbers)

1. **Persistence dominates at 30 min** (macro-F1 0.848 vs GRU 0.831; onset-F1 0 by
   construction) → standard metrics mislead; congestion states persist (L dwell
   ≈ 18 h, H ≈ 2.6 h).
2. **Deep models earn their keep at 60+ min, specifically on onset**: GRU onset-F1
   0.237 at 60 min vs 0.206 (seasonal-naïve) / 0.198 (persistence); at 120 min GRU
   onset-recall 0.456 — more than double any baseline.
3. **Transfer works**: zero-shot from other Manila corridors (macro-F1 0.680) beats
   from-scratch until ~2 weeks of local data; fine-tuning wins at every k
   (0.803 vs 0.737 at k=28). Crossover ≈ 14 days = the headline figure.

## Remaining work

### Day 1 — unblock Sathorn (the only blocked experiment)
1. Manually run the `7z_setup.exe` already downloaded in the project root (needs a human click).
2. ```powershell
   & "C:\Program Files\7-Zip\7z.exe" x "dataset\bangkok_sathorn-intersection\raw\loopcoil.rar" "-odataset\bangkok_sathorn-intersection\extracted\loopcoil"
   .venv\Scripts\python.exe experiments\sathorn\prepare_sathorn.py --inspect
   # fill in the real column names printed by --inspect:
   .venv\Scripts\python.exe experiments\sathorn\prepare_sathorn.py --csv "dataset/bangkok_sathorn-intersection/extracted/loopcoil/**/*.csv" --time-col <T> --value-col <OCC> --series-col <ID> --mode occupancy --out experiments\out\sathorn.parquet
   .venv\Scripts\python.exe experiments\transfer_kday.py --data experiments\out\sathorn.parquet --pretrained experiments\runs\manila_segments_gru_cls_h1\best.pt --k-days 1 3 7 14 28
   .venv\Scripts\python.exe experiments\analysis\aggregate_results.py   # regenerates figures incl. the new curve
   ```
   Budget: ~1–2 h including extraction. This gives the **cross-city** (Manila→Bangkok) curve.

### Day 1 (parallel) — decisions & lead-time items
- HERE API key + start the collector ([collect/COLLECTION_PLAN.md](collect/COLLECTION_PLAN.md)) —
  even 2 weeks of 2026 Dhaka data upgrades the paper's dataset contribution; it
  keeps collecting while you write.
- Pick the venue and its deadline (ICCIT is the usual Bangladesh target — verify
  this year's date NOW; SIGSPATIAL UrbanAI workshop as backup).

### Days 2–3 — cheap experiment top-ups (each ≈ 1–2 min of GPU)
```powershell
.venv\Scripts\python.exe experiments\train_forecaster.py --data experiments\out\manila_segments.parquet --model lstm --horizon 2
.venv\Scripts\python.exe experiments\train_forecaster.py --data experiments\out\manila_segments.parquet --model tcn  --horizon 2
.venv\Scripts\python.exe experiments\train_forecaster.py --data experiments\out\manila_segments.parquet --model lstm --horizon 4
.venv\Scripts\python.exe experiments\train_forecaster.py --data experiments\out\manila_segments.parquet --model tcn  --horizon 4
.venv\Scripts\python.exe experiments\analysis\aggregate_results.py
```
Fills the h2/h4 rows for all three architectures (reviewers will ask).

### Days 2–6 — write the paper (the real work now)
Section → artifact mapping is the table above. Structure:
1. Intro + related work (proposal §1–3 already drafts this)
2. Data: rescue/reconstruction method + coverage stats + UCI harmonization
3. RQ1 characterization: Markov/dwell, daily/weekly rhythm, cross-city table
4. RQ2 forecasting: results table + horizon decay + the "persistence paradox" finding
5. RQ3 transfer: k-day curves (within-Manila + Manila→Bangkok)
6. Vision side note: DhakaAI YOLO vehicle-mix (0.431 mAP50, rickshaw/CNG classes)
7. Limitations: 8 % gap-free window yield, positional segment identity,
   2015-16 Manila vintage (framed as archival rescue), volume-proxy CI for Sathorn
8. Release: scraper + pipeline + manila_segments.parquet on GitHub/Zenodo

### Days 7–10 — release engineering + polish
- Scraper repo: strip secrets/.env, add LICENSE (MIT), push to GitHub public.
- Dataset release: `manila_segments.parquet` + reconstruction script + README → Zenodo (gets a DOI to cite in the paper).
- Faculty review pass; final figures at 300 dpi if the venue asks.

## Honesty notes for the paper (do not drop these)
- Only ~8 % of raw rows form gap-free 12 h windows (median coverage per series
  55–86 % — see `manila_road_ranking.csv`); windows never bridge gaps.
- Persistence/seasonal baselines are the bar at short horizons — report them
  prominently, it makes the 60-min onset result credible.
- H-class (severe) support is thin (0.84 % of rows); onset metrics computed on
  503–607 test events.
