# Fresh Data Collection Plan — start immediately

> Every day of delay = one less day of 2026 data in the paper.
> Target: **≥5 weeks** of 15-minute jam-factor series for **Dhaka + Manila** by late August 2026.

## Why this fixes the "10-year-old data" problem

- We collect **2026 data ourselves** via the HERE Traffic API using the *same
  open-source pipeline* ([firmanhadi21/traffic-analyses](https://github.com/firmanhadi21/traffic-analyses))
  that produced the Jakarta 2025–26 Zenodo dataset → perfect cross-city
  comparability (same source family, same collector, same schema).
- The 2015–16 Manila logs stop being a weakness and become a second research
  axis: **decade-apart temporal transfer** (does a model trained on Manila 2015
  still forecast Manila 2026?) — nobody has published that for a SE Asian city.
- This is an API collector, not a scraper — no fragility, no ToS gray zone.
  Our Google-Maps scraper is still open-sourced as prior work.

## Setup (one evening)

```powershell
pip install traffic-congestion-pipeline

# free HERE developer account -> API key: https://platform.here.com (Base Plan)
$env:HERE_API_KEY = "<your key>"

# smoke test — one collection cycle for Dhaka
traffic-pipeline collect --provider here --api-key $env:HERE_API_KEY --once
```

Configure two cities (check the pipeline's config format after install; these
are the starting bounding boxes, W,S,E,N — trim to the urban core if segment
counts are huge):

| City | Bounding box (W, S, E, N) | Note |
|---|---|---|
| Dhaka | `90.32, 23.66, 90.51, 23.90` | metro core |
| Manila (Metro) | `120.93, 14.35, 121.13, 14.78` | includes EDSA/C5 corridors — overlaps the 2015 roads |

## Cost math (HERE Base Plan)

- 1 bbox flow request ≈ 1 transaction; free tier ≈ **5,000 transactions/month**.
- 15-min cadence = 96 req/day ≈ **2,880/month per city** → one city free.
- Two cities at 15-min ≈ 5,760/month → either **20-min cadence for both**
  (4,320/month, still free) or ~US$2/month overage. Never exceeds a few dollars.

## Operational rules (write these into the paper's methodology)

1. Freeze the bbox and cadence for the whole collection window.
2. Run on a machine that stays on (the PC, or the Hostinger VPS if the PC
   sleeps at night — a $5 VPS is enough, it's one HTTP request per cycle).
3. Log every failed cycle; report gap statistics honestly.
4. Back up the raw GeoPackage snapshots weekly (the pipeline stores raw before
   aggregating — keep BOTH raw and aggregates; raw is what forecasting needs).
5. Windows scheduler example (every 20 min):
   ```powershell
   schtasks /create /tn TrafficCollect /sc minute /mo 20 `
     /tr "cmd /c cd /d F:\CSE498R\TrafficTracker && traffic-pipeline collect --provider here --api-key %HERE_API_KEY% --once"
   ```

## Timeline (today = 2026-07-13, week 5 of 12)

| Date | Milestone |
|---|---|
| Jul 13–14 | HERE key, pipeline installed, both cities collecting |
| Jul 20 | 1 week of data — first k=7 fine-tune point becomes real |
| Aug 10 | 4 weeks — main scarcity experiments on REAL 2026 target data |
| Aug 24 | 6 weeks — final training cut for the paper |
| Week 12 (Sep 2) | paper reports 5–6 weeks of Dhaka+Manila 2026 data; collection keeps running for the camera-ready |

## Also do (10 minutes each)

- **Email Firman Hadi** (Diponegoro Univ., pipeline author): ask for the raw
  15-min GeoPackage snapshots behind Zenodo v2 (record 19211072). His pipeline
  stores them; v2 only ships aggregates. If he shares, Jakarta becomes a fresh
  *forecasting* city too, with 11 months of 2025–26 data.
- **HERE portal**: while making the API key, also look for the MeTS-10 /
  Traffic4cast sample-data download (transfer source, 2019–21 — old but it's
  the established benchmark, reviewers accept it as such).

## How each dataset is framed against the age question

| Dataset | Years | Framing in the paper |
|---|---|---|
| Dhaka + Manila HERE (ours) | **2026** | primary fresh evidence, our dataset contribution |
| Jakarta/Bandung/Semarang Zenodo | **2025–26** | fresh comparison city (spatial; forecasting if raw obtained) |
| MeTS-10 Bangkok | 2019–21 | established benchmark corpus for pre-training (standard practice) |
| MMDA incidents | 2018–**2020** | incident characterization (say 2020, not 2018) |
| Manila benjiao logs | 2015–16 | **rescued historical archive** → decade-apart temporal-transfer axis, NOT a contemporary snapshot |
| Sathorn Bangkok | 2016–19 | ground-truth validation of CI methodology only |
