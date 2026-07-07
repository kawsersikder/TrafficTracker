# Bangkok — Sathorn Intersection Multimodal Sensors

## Credit
- **Source repository:** figshare, DOI [10.6084/m9.figshare.14643411.v1](https://doi.org/10.6084/m9.figshare.14643411.v1). License: **CC BY 4.0**.
- **Research paper:** Mon, E.E., Ochiai, H., Komolkiti, P. & Aswakul, C. *Real-world sensor dataset for city inbound-outbound critical intersection analysis.* **Scientific Data** 9, 357 (2022). DOI [10.1038/s41597-022-01448-6](https://doi.org/10.1038/s41597-022-01448-6)
- Collected under the Sathorn Model / Chula-SSS project (Chulalongkorn University), Sathorn–Narathiwas intersection, Bangkok.

## Type
**Multimodal sensor time series** (the only truly multimodal dataset in our collection):
| File (in `raw/`) | Modality | Coverage |
|---|---|---|
| `cctv-camera.rar` (261 MB) | CCTV-derived traffic volumes | 37 months, 2016–2019 |
| `loopcoil.rar` (87 MB) | Loop-coil occupancy + volume | 110 days, May–Sep 2016 |
| `thermal-cctv.rar` (16 MB) | Thermal-camera occupancy + volume | 26 days, May–Jun 2016 |
| `signal.rar` (1.6 MB) | Signal timing plans | 22 months, Nov 2014–Sep 2016 |
| `Taxi_ground_truth_speed_RAWDATA.csv` | GPS taxi ground-truth speeds | 6 evaluation days |
| `manual-*.rar` | Manual queue/signal/volume counts | validation samples |

## Fetched
✅ All 9 files (~365 MB) in `raw/`. `.rar` archives — extract with 7-Zip locally or `!apt-get install unrar && unrar x file.rar` on Colab. Re-fetch example:
```bash
curl -L https://ndownloader.figshare.com/files/29021766 -o cctv-camera.rar   # file ids: 28369986, 28369989, 28370343, 29021763, 29021766, 29021775, 34026611, 34026614, 34026617
```

## Role in the project
**Ground-truth validation, not a training corpus.** Physical-sensor measurements at one Bangkok intersection let us check that our floating-car-derived Congestion Index (from MeTS-10) tracks real volumes/occupancy — a validation subsection reviewers will like.

## Usage plan
- **Analysis:** correlate MeTS-10 CI for the Sathorn segments (2019 overlap window with CCTV volumes) against sensor volumes; report correlation + error bands.
- **Models:** none trained here; optionally a small single-intersection LSTM as a sanity baseline.
- **Links to:** [bangkok-istanbul_mets10](../bangkok-istanbul_mets10/) (validates its CI), and methodologically to [dhaka_trafficktracker-own](../dhaka_trafficktracker-own/) (same validate-the-proxy logic we apply to Google Maps colors).
