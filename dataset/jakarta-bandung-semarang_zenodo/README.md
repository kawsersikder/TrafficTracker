# Jakarta, Bandung, Semarang — HERE API Congestion (Zenodo)

## Credit
- **Source repository:** Zenodo record [10.5281/zenodo.19211072](https://zenodo.org/records/19211072) (v2, 2026-03-25); concept DOI [10.5281/zenodo.18650759](https://doi.org/10.5281/zenodo.18650759). Creator: **Firman Hadi, Diponegoro University** (ORCID 0000-0003-2982-2430).
- **GitHub project (pipeline):** [firmanhadi21/traffic-analyses](https://github.com/firmanhadi21/traffic-analyses)
- **Research paper:** Hadi, F., Wahyuddin, Y., Sabri, L. M., & Indrajit, A. (2026). *An Open-Source Pipeline for Spatiotemporal Traffic Congestion Analysis: Integrating HERE API, OSMnx, and PySAL Across Indonesian Metropolitan Cities.* Computers, Environment and Urban Systems.
- **License:** CC BY 4.0.

## Type
**Geospatial tabular** (GeoPackage, EPSG:4326). HERE Traffic API **jam factors (0–10)** collected every 15 min from **March 2025 – February 2026**, aggregated to 8 time-of-day periods per road segment. Jakarta: 14,549 segments / 206 M raw observations; Bandung: 3,069; Semarang: 1,076. Plus `analysis_results/` CSVs (Moran's I, LISA, ANOVA, centrality & POI correlations). Columns per segment: `jam_factor_mean/std/min/max/count`, MULTILINESTRING geometry.

## Fetched
✅ `raw/zenodo_traffic_data_v2_20260320.zip` (123 MB, MD5 `ddb88a2c42604323462be7961c5023bb`) — extracted to `raw/zenodo_export/`. Re-fetch:
```bash
wget "https://zenodo.org/api/records/19211072/files/zenodo_traffic_data_v2_20260320.zip/content" -O jakarta.zip
```

## ⚠ Important caveat
v2 is aggregated to **8 time-of-day periods** — it is NOT a full 15-minute time series, so it supports **spatial characterization (RQ1)** but not sequence forecasting directly. The Zenodo record shows ~5.8 GB cumulative across versions — **check version 1 for the raw 15-min series**, and/or email the author (the pipeline is open source, and he collected 206 M raw observations). If raw series are unobtainable, Jakarta is used for characterization + spatial transfer only, and Manila becomes the second forecasting target.

## Role in the project
Second comparison city (Jakarta), with Bandung/Semarang as bonus mid-size Indonesian cities — useful for a "does city size matter?" ablation.

## Usage plan
- **Congestion index:** `CI = jam_factor_mean / 10` (HERE jam factor is already a normalized congestion measure — same source family as our Dhaka HERE/Google-style signal, which strengthens comparability).
- **Models:** spatial statistics reuse (Moran's I / LISA — results included); if raw series obtained: LSTM/ST-GNN forecasting + transfer target.
- **Links to:** [bangkok-istanbul_mets10](../bangkok-istanbul_mets10/) (transfer source), [dhaka_trafficktracker-own](../dhaka_trafficktracker-own/) (fellow target city). Read `.gpkg` with `geopandas.read_file()`.
