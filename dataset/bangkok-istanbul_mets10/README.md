# Bangkok (+ Istanbul) — MeTS-10 / Traffic4cast

## Credit
- **GitHub project:** [iarai/MeTS-10](https://github.com/iarai/MeTS-10) — data pipeline + analysis code (Apache 2.0). Cloned into [`code/`](code/).
- **Research paper:** Neun, Eichenberger, Xin, Fu, Wiedemann, Martin, Tomko, Ambühl, Hermes, Kopp — *Metropolitan Segment Traffic Speeds from Massive Floating Car Data in 10 Cities*, IEEE Transactions on Intelligent Transportation Systems, 2023. DOI [10.1109/TITS.2023.3291737](https://doi.org/10.1109/TITS.2023.3291737) · [arXiv:2302.08761](https://arxiv.org/abs/2302.08761)
- **Underlying data:** HERE Technologies Traffic4cast "traffic map movies" (NeurIPS 2019–2022 competitions, IARAI), aggregated from >100 billion GPS probe points.

## Type
**Tabular spatio-temporal.** Segment-level speed classes at 15-minute resolution, 108–361 days per city (2019–2021), for 10 cities incl. **Bangkok** (SE Asian megacity — our transfer source) and **Istanbul** (control). Formats: HDF5 movie grids `(288, 495, 436, 8)` per day (volume+speed × 4 headings), Parquet segment tables, GraphML/GeoPackage OSM road graphs. Full spec: [`code/README_DATA_SPECIFICATION.md`](code/README_DATA_SPECIFICATION.md).

## How to get the data (⚠ read carefully — hosting changed)
1. **IARAI's S3 bucket is dead** (institute dissolved; verified 404 on 2026-07-07). Old `iarai-public.s3-eu-west-1.amazonaws.com` links in tutorials will not work.
2. **Primary route:** HERE sample data page — the paper states the Traffic4cast movies "can be downloaded for free from the HERE sample data" portal: <https://developer.here.com/sample-data> (now redirects to <https://www.here.com/developer>). Requires a **free HERE developer account**. License: academic and non-commercial use.
3. **Fallback:** email the corresponding author (Moritz Neun) or Christian Eichenberger via the GitHub repo — the community has been granted copies before.
4. Once movies are obtained, the cloned pipeline in `code/` (`dp01`…`dp03` scripts) converts them into segment speeds + road graph. Run on **Colab Pro** (per-city data is tens of GB; process **Bangkok only** first, Istanbul second).

## Role in the project
**Source city for pre-training** (data-rich Asian megacity). The headline experiment: pre-train here → transfer to Dhaka/Jakarta/Manila.

## Usage plan
- **Congestion index:** `CI = 1 − speed / free_flow_speed`, free-flow = 85th percentile night-time speed per segment.
- **Models:** deep learning — LSTM/GRU/TCN per segment; **Graph WaveNet / DCRNN** (PyTorch Geometric) on the OSM road graph; this is the pre-training corpus.
- **Links to:** [dhaka_trafficktracker-own](../dhaka_trafficktracker-own/) (transfer target), [bangkok_sathorn-intersection](../bangkok_sathorn-intersection/) (independent ground-truth check of the CI at one Bangkok intersection).
