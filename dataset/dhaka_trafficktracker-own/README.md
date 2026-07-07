# Dhaka — TrafficTracker (our own dataset)

## Credit
- **Source:** This repository (TrafficTracker, CSE498R — North South University). Capture pipeline: [`jobs/run_capture.ts`](../../jobs/run_capture.ts) + [`browser/capture.ts`](../../browser/capture.ts); scoring: [`vision/extract_traffic.py`](../../vision/extract_traffic.py); export: [`jobs/export_training_set.ts`](../../jobs/export_training_set.ts).
- **Methodology inspiration (cite in paper):** Md. Rahman et al., *Modeling Traffic Congestion in Developing Countries using Google Maps Data*, [arXiv:2011.02359](https://arxiv.org/pdf/2011.02359) — prior art for deriving congestion from Google Maps traffic colors in Dhaka.
- **Underlying data source:** Google Maps live traffic layer (crowdsourced). Note in the paper's limitations that this is a proprietary, indirect congestion signal.

## Type
**Tabular time series** (generated): one row per intersection arm per capture — timestamp, intersection id, arm id, congestion score, dominant color, red-queue fraction. Plus **annotated PNG snapshots** as supporting imagery. Stored in PostgreSQL (`traffic_observations` table); export to CSV via the export job.

## Collection protocol (start immediately — every day of delay shrinks the dataset)
- ~25 fixed Dhaka intersections, captured **every 15 minutes**, 24/7, for ≥ 6 weeks.
- Keep zoom level, map style, and arm geometries **frozen** for the whole collection period.
- Log capture failures; gaps are expected and must be reported honestly in the paper.

## Role in the project
**Target city** for the transfer-learning experiments (RQ3) and the dataset-paper contribution (RQ4). This is the data-scarce city that models pre-trained on Bangkok are transferred to.

## Usage plan
- **Congestion index:** ordinal mapping green→0.1, yellow→0.4, red→0.7, dark-red→0.95 (calibrate thresholds against short manual counts; TFP-BD footage locations can support validation).
- **Models:** fine-tuning target for LSTM/GRU and ST-GNN pre-trained on MeTS-10 Bangkok; from-scratch LSTM as the comparison baseline.
- **Links to:** [bangkok-istanbul_mets10](../bangkok-istanbul_mets10/) (transfer source), [dhaka_tfp-bd](../dhaka_tfp-bd/) (vehicle-mix explanation of transfer gap).
- **Publication:** release cleaned CSV + snapshots + collection code on Zenodo, with a *Data in Brief*-style dataset paper.
