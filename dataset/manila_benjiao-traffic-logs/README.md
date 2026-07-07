# Manila — Road Status Logs (benjiao/manila-traffic-data)

## Credit
- **GitHub project:** [benjiao/manila-traffic-data](https://github.com/benjiao/manila-traffic-data) by Benjie Oliveros (benjiao). Cloned into `raw/`.
- **Original source:** MMDA official traffic status feed, `mmdatraffic.interaksyon.com` (site now defunct — the author logged it before shutdown, which makes this dataset irreplaceable).
- **Research paper:** none — cite the GitHub repository and MMDA/Interaksyon as primary sources.

## Type
**Tabular road-status time series.** 2,214,833 rows, Nov 2015 – Jun 2016. Columns: `date_extracted, date_published, direction (NB/SB), road (e.g., COMMONWEALTH, EDSA), status`. Status is ordinal: **L** (light), **ML** (moderate-light), **MH** (moderate-heavy), **H** (heavy). Major arterials of Metro Manila, roughly 15-min update cadence.

## Fetched
✅ `raw/manila-traffic.csv` (161 MB, extracted from `manila-traffic.csv.tar.gz`). Re-fetch:
```bash
git clone --depth 1 https://github.com/benjiao/manila-traffic-data
tar -xzf manila-traffic-data/manila-traffic.csv.tar.gz
```

## Role in the project
Manila's dense congestion-state series — the **third transfer-target city** for forecasting (RQ2/RQ3), and Manila's series for cross-city characterization (RQ1).

## Usage plan
- **Congestion index:** ordinal map L→0.15, ML→0.4, MH→0.65, H→0.9 (road-level, not segment-level — note the coarser spatial granularity as a limitation).
- **Models:** LSTM/GRU sequence classification per road+direction (predict next status); transfer experiments: Bangkok-pretrained encoder → fine-tune here.
- **Caveat:** 2015–16 data vs. 2019–21 (Bangkok) vs. 2025–26 (Jakarta, Dhaka) — the study compares *dynamics and transferability*, not contemporaneous levels; state this explicitly (already covered in [RESEARCH_PROPOSAL.md](../../RESEARCH_PROPOSAL.md) risks).
- **Links to:** [manila_mmda-incidents](../manila_mmda-incidents/) (same city, incident view), [bangkok-istanbul_mets10](../bangkok-istanbul_mets10/) (transfer source).
