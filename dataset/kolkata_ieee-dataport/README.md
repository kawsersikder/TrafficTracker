> **DROPPED (2026-07-13).** Team decision: Kolkata is excluded from the study (per the proposal's decision rule — drop rather than weaken the paper with an unverified source). Kept for provenance only.

# Kolkata — Traffic Congestion Management Dataset (IEEE DataPort)

## Credit
- **Source repository:** IEEE DataPort — [Traffic Congestion Management Dataset](https://ieee-dataport.org/documents/traffic-congestion-management-dataset), DOI [10.21227/xc4j-p870](https://dx.doi.org/10.21227/xc4j-p870) (2026).
- **Authors:** Apurba Nandi, Shaoni Banerjee, Avik Kumar Das, Sangita Dutta.
- **Research paper:** check the DataPort record for the linked publication once accessed.

## Type
**Tabular (unverified).** Contents, size, time range, and spatial granularity are **not yet confirmed** — the record requires an IEEE account to open.

## How to fetch
1. Create a **free IEEE account** (IEEE DataPort open-access datasets are downloadable with a basic account; subscription only needed for "standard" tier datasets — check which tier this is).
2. Download from the record page; place files in `raw/` here and update this README with actual columns/coverage.

## Decision rule (from the proposal)
Kolkata is **optional**. If this dataset turns out to be thin, synthetic, or incomparable → **drop Kolkata** rather than weaken the study. Fallback India-wide traffic-intensity proxy: **CHETNA-Road** ([Scientific Data 2025](https://www.nature.com/articles/s41597-025-06287-9), open, 15 Indian cities incl. Kolkata — gridded traffic-derived emissions from floating-car data). Secondary fallback: HERE API live collection for Kolkata using the same pipeline as the Jakarta dataset ([firmanhadi21/traffic-analyses](https://github.com/firmanhadi21/traffic-analyses)) — freemium HERE key, needs weeks of lead time.

## Usage plan (if usable)
- Same as other comparison cities: map to Congestion Index → characterization (RQ1) → forecasting/transfer (RQ2/RQ3) if it contains a real time series.
