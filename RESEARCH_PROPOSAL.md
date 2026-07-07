# Cross-City Deep Learning for Traffic Congestion in South & Southeast Asian Megacities

> **CSE498R — Computer Science Research Project**
> North South University · Department of Electrical and Computer Engineering
> Working title: *"Does a Model Trained on Bangkok Understand Dhaka? Cross-City Transferability of Deep Congestion Forecasting in Data-Scarce Asian Megacities"*

---

## 1. Motivation

Dhaka is consistently ranked among the most congested cities in the world, yet it has **no permanent traffic sensor infrastructure** — no loop detectors, no city-wide CCTV analytics, no open data portal. The same is true, to varying degrees, for Manila, Jakarta, and Kolkata. Meanwhile, nearly all published deep-learning traffic forecasting research is built on sensor-rich Western/Chinese benchmarks (METR-LA, PEMS-BAY, PeMS) with homogeneous, lane-disciplined traffic.

This creates a genuine research gap:

1. **Data gap** — congestion models cannot be trained where they are needed most, because those cities have no data.
2. **Domain gap** — it is unknown whether models trained on cities that *do* have data (e.g., Bangkok, Istanbul) transfer to heterogeneous, non-lane-based traffic like Dhaka's (rickshaws, CNGs, buses, and pedestrians sharing road space).

Our project attacks both gaps at once.

## 2. Research Questions

- **RQ1 (Comparison):** How do congestion dynamics (daily/weekly periodicity, peak spreading, recovery time after breakdown) differ between Dhaka and comparable South/Southeast Asian cities (Bangkok, Jakarta, Manila, Kolkata)?
- **RQ2 (Forecasting):** Which deep learning architectures (LSTM/GRU, Temporal Convolutional Networks, Spatio-Temporal Graph Neural Networks, Transformers) best forecast short-term congestion in these cities?
- **RQ3 (Transferability — the novel contribution):** Can a model pre-trained on a data-rich Asian city (Bangkok, via MeTS-10) be transferred to a data-scarce city (Dhaka, via our own collected data) with little or no fine-tuning? How many days of local data are needed before local training beats transfer?
- **RQ4 (Dataset contribution):** Can we publish our own harmonized, multi-city Google-Maps-derived congestion dataset as a citable resource?

## 3. Why This Is Publishable (Novelty Argument)

The topic "traffic prediction with ML" is common. The following combination is **not**:

| Common (rejected as-is) | Our angle (defensible novelty) |
|---|---|
| Train an LSTM on one city's data, report RMSE | **Cross-city transfer learning** between Asian megacities with different traffic regimes |
| Use METR-LA / PeMS benchmarks | Use **South/SE Asian data only**, incl. a **new Dhaka dataset we collect ourselves** |
| Single data modality | **Harmonize heterogeneous sources** (floating-car speeds, incident logs, crowdsourced map colors) into one comparable congestion index |
| No dataset released | **Release the Dhaka dataset** (our TrafficTracker pipeline is already built for this) — dataset papers are independently publishable (e.g., *Data in Brief*, like TFP-BD) |

Key precedent showing the niche is active but not saturated:
- *Modeling Traffic Congestion in Developing Countries using Google Maps Data* ([arXiv:2011.02359](https://arxiv.org/pdf/2011.02359)) — Dhaka, but single-city and pre-deep-learning-era methods.
- *Inferring Traffic Patterns of Dhaka City: A Spatio-temporal Analysis over a Year* ([ResearchGate](https://www.researchgate.net/publication/377131998)) — single-city analysis, no cross-city transfer.
- MeTS-10 paper ([arXiv:2302.08761](https://arxiv.org/abs/2302.08761), IEEE T-ITS 2023) — provides Bangkok/Istanbul data and explicitly invites downstream studies; **no published work yet transfers it to a South Asian city**.

## 4. Candidate Cities and Verified Datasets

### Tier 1 — Recommended core (verified, downloadable, DL-ready)

| Dataset | City / Country | Type | Access | Notes |
|---|---|---|---|---|
| **MeTS-10** ([GitHub](https://github.com/iarai/MeTS-10), [paper](https://arxiv.org/abs/2302.08761)) | **Bangkok**, Istanbul, Melbourne + 7 others | Segment-level speeds, 15-min resolution, 108–361 days (2019–2021), from massive floating-car data | Open download | **Best-in-class source city for transfer learning.** Bangkok = SE Asian megacity with real DL-scale data. Istanbul/Melbourne usable as control cities. |
| **TrafficTracker (ours)** | **Dhaka**, Bangladesh | Google Maps congestion colors per intersection arm, our own capture + OpenCV pipeline | We generate it | Already built in this repo. Reframe last week's "failed scraping" as the **data collection instrument** — run it on a fixed schedule (e.g., every 15 min, 20–30 intersections) for 6–8 weeks → citable dataset. |
| **Traffic Congestion Dataset: Semarang, Bandung, Jakarta 2025–2026** ([Zenodo](https://zenodo.org/records/19211072)) | **Jakarta** + 2 cities, Indonesia | Congestion time series, ~129 MB (v2), CC-BY 4.0 | Open download | Recent (Mar 2026), openly licensed. Inspect variables after download; Diponegoro University origin. |
| **MMDA Traffic Incident Data** ([Kaggle](https://www.kaggle.com/datasets/esparko/mmda-traffic-incident-data)) | **Manila**, Philippines | Tabular incident logs (from MMDA Twitter feed) | Open download | Good for incident-aware analysis; complements [benjiao/manila-traffic-data](https://github.com/benjiao/manila-traffic-data) (road status logs). |

### Tier 2 — Supporting / secondary (verified to exist; some need requests or cleanup)

| Dataset | City / Country | Type | Access | Notes |
|---|---|---|---|---|
| TFP-BD ([Data in Brief](https://pmc.ncbi.nlm.nih.gov/articles/PMC11919376/)) | Dhaka | 23,678 images, 4 locations | Open | Vision track / vehicle-mix characterization. Also a **template for our own dataset paper**. |
| DhakaAI ([Kaggle](https://www.kaggle.com/datasets/rifat963/dhakaai-dhaka-based-traffic-detection-dataset), [Roboflow](https://universe.roboflow.com/traffic-wake5/dhakaai-gx36d)) | Dhaka | 3,953 images, 21 vehicle classes | Open | Vehicle detection (heterogeneity quantification). |
| Poribohon-BD ([Mendeley](https://data.mendeley.com/datasets/pwyyg8zmk5/2)) | Bangladesh | Vehicle classification images | Open | Vision track support. |
| Sathorn intersection sensors ([Sci. Data](https://www.nature.com/articles/s41597-022-01448-6)) | Bangkok | Multimodal sensor time series | Open | Ground-truth validation for one Bangkok intersection. |
| Bangkok Open Data — traffic volume ([portal](https://data.bangkok.go.th/en/dataset/traffic_volume)) | Bangkok | Tabular volumes | Portal intermittently unreachable from abroad — retry / mirror | Secondary to MeTS-10. |
| iTIC Open Data ([iTIC Foundation](https://org.iticfoundation.org/download)) | Thailand | Probe + historical | Registration | Extra Thai probe data. |
| Traffic Congestion Management Dataset ([IEEE DataPort](https://ieee-dataport.org/documents/traffic-congestion-management-dataset)) | Kolkata, India | Tabular | IEEE DataPort (free w/ IEEE account) | Newest Kolkata-specific option (2026). Verify contents before committing to Kolkata as a compared city. |
| CHETNA-Road ([Sci. Data 2025](https://www.nature.com/articles/s41597-025-06287-9)) | 15 Indian cities incl. Kolkata | 500 m gridded traffic-derived emissions from floating-car data | Open | City-level traffic intensity proxy for Kolkata if the IEEE DataPort set is thin. |
| HERE API + weather, New Delhi ([method paper](https://arxiv.org/abs/2206.10983)) | Delhi, India | Jam factor + weather | API (freemium) | Fallback for India; API quota limits at scale. |
| Dhaka GPS taxi intensity ([arXiv:2308.08501](https://arxiv.org/pdf/2308.08501)) | Dhaka | GPS + OSM | Email authors | Worth one request email; do not block on it. |
| MMDA FOI datasets ([foi.gov.ph](https://www.foi.gov.ph/agencies/mmda/)) | Manila | Official tabular | Formal request | Submit early; treat as bonus. |
| Kaggle fedesoriano 4-junction ([Kaggle](https://www.kaggle.com/datasets/fedesoriano/traffic-prediction-dataset)) | City-agnostic | Tabular | Open | Prototyping/debugging models only — **not citable as a city comparison** (city unknown). |

### Recommended final city set

**Dhaka (ours) · Bangkok (MeTS-10) · Jakarta (Zenodo) · Manila (MMDA)** — with Istanbul (MeTS-10) as a non-SE-Asian control. Kolkata only if the IEEE DataPort dataset proves usable; otherwise drop it rather than weaken the paper with an incomparable source.

## 5. Methodology

### Phase 1 — Unified Congestion Index (harmonization)
Each source reports congestion differently (speeds, colors, incident counts). We map each onto a normalized **Congestion Index CI ∈ [0, 1]** per road segment per 15-minute bin:
- MeTS-10: `CI = 1 − speed / free_flow_speed` (free-flow = 85th percentile night speed).
- TrafficTracker (Dhaka): green/yellow/red/dark-red → ordinal CI via the existing OpenCV scoring, calibrated against short manual video counts at 2–3 intersections (uses TFP-BD-style footage for validation).
- Jakarta/Manila: mapped from the respective congestion/incident variables after inspection.

The harmonization procedure is itself a methodological contribution.

### Phase 2 — Characterization (RQ1)
Descriptive + statistical comparison across cities: peak-hour profiles, weekday/weekend and religious-calendar effects (Friday prayers in Dhaka vs. Sunday effects in Manila), congestion onset/recovery speed, spatial concentration (Gini coefficient of CI across segments). This alone yields a publishable comparative-analysis section.

### Phase 3 — Forecasting benchmarks (RQ2)
Predict CI 15/30/60 minutes ahead. Models, simplest first:
1. Baselines: historical average, ARIMA, XGBoost on lag features.
2. Deep: LSTM/GRU, Temporal Convolutional Network.
3. Spatio-temporal: ST-GNN (e.g., Graph WaveNet / DCRNN) on the road-segment graph — feasible for MeTS-10 and our intersection-arm graph.
Metrics: MAE, RMSE, and congestion-onset F1 (predicting the jump to CI > 0.7), which matters more to practitioners than average error.

### Phase 4 — Cross-city transfer (RQ3, headline result)
- Zero-shot: train on Bangkok → test on Dhaka/Jakarta/Manila.
- Fine-tuning curves: pre-train on Bangkok, fine-tune on *k* days of target-city data (k = 1, 3, 7, 14, 28); plot transfer vs. from-scratch. The crossover point ("how much local data does a data-scarce city need?") is the paper's headline figure.
- Optional stretch: simple domain-adaptation (feature alignment / DANN) if time allows.

### Phase 5 — Dataset release (RQ4)
Publish the cleaned Dhaka TrafficTracker dataset (CSV + annotated snapshots + collection code) on Zenodo/Mendeley with a *Data in Brief*-style paper as a second, lower-risk publication.

## 6. Expected Outcomes

1. A harmonized multi-city congestion dataset for 4–5 Asian megacities (public release).
2. Benchmark results for 6+ models across all cities.
3. Transferability findings — evidence for or against "borrowing" models from data-rich to data-scarce cities. **Negative results are publishable here too**: if Bangkok→Dhaka transfer fails, quantifying *why* (vehicle heterogeneity, signal noncompliance) is itself a finding.
4. One main paper + one optional dataset paper.

## 7. Publication Targets (realistic ladder)

| Tier | Venue | Fit |
|---|---|---|
| Reach | IEEE ITSC (conference), IEEE Access (journal) | Transfer-learning result with strong experiments |
| Realistic | IEEE TENCON, ICCIT (Bangladesh), ACM SIGSPATIAL workshops (UrbanAI / GeoAI) | Comparative study + benchmarks |
| Dataset | *Data in Brief* (Elsevier), *Scientific Data* (reach) | Dhaka dataset — TFP-BD proves *Data in Brief* accepts exactly this from Bangladesh |

## 8. Alignment with the 12-Week Timeline (today = Week 4)

| Weeks | Work |
|---|---|
| 4 | Faculty approval of this proposal; **start TrafficTracker scheduled Dhaka capture immediately** (every day of delay shortens our own dataset); download MeTS-10 Bangkok, Zenodo Jakarta, MMDA Manila; send the two request emails (Gobd authors, MMDA FOI). |
| 5–6 | Data cleaning + Unified Congestion Index; Phase 2 characterization figures. |
| 7–8 | Baselines + LSTM/TCN forecasting on Bangkok and Jakarta (most data available earliest). |
| 9–10 | ST-GNN; begin transfer experiments as Dhaka collection passes ~4 weeks of data. |
| 11 | Transfer/fine-tuning curves (headline experiments); ablations. |
| 12 | Paper draft + dataset packaging; final presentation. |

## 9. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Google Maps capture breaks again / rate-limits | Capture is now *scheduled snapshots* at fixed zoom on ~25 fixed intersections, not bulk scraping — far lighter footprint; keep the batch runner (`jobs/run_capture.ts`) under a conservative schedule and log failures. |
| Our Dhaka data window too short for DL | Transfer-learning framing *expects* a small target dataset — scarcity is part of the research question, not a flaw. |
| Zenodo Jakarta dataset turns out unusable | Fall back to Manila (MMDA) as the second target city; Jakarta becomes optional. |
| Kolkata data never materializes | Drop Kolkata; the study stands on Dhaka–Bangkok–Jakarta–Manila. |
| MeTS-10 is 2019–2021, ours is 2026 | We compare *dynamics and transferability*, not absolute contemporaneous levels; note as a limitation. |
| Datasets have incompatible semantics | The Unified Congestion Index (Phase 1) is designed precisely for this and is claimed as a contribution. |

## 10. Bottom Line for Faculty

We are **not** proposing "yet another traffic prediction model." We are proposing the first cross-city transferability study of deep congestion forecasting centered on a data-scarce South Asian megacity, powered by (a) the strongest open floating-car dataset in Asia (MeTS-10 Bangkok), and (b) a new Dhaka dataset collected by our own already-working pipeline — which we release publicly as a second contribution.
