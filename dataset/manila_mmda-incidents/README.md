# Manila — MMDA Traffic Incident Data (Kaggle)

## Credit
- **Source:** Kaggle dataset [esparko/mmda-traffic-incident-data](https://www.kaggle.com/datasets/esparko/mmda-traffic-incident-data). Original data: **Metropolitan Manila Development Authority (MMDA)** official Twitter alerts (@MMDA), scraped and geocoded by the Kaggle uploader.
- **Research paper:** none canonical — cite the Kaggle dataset + MMDA as the primary source. (Prior academic use exists in Philippine traffic-incident studies; cite any specific paper only after reading it.)

## Type
**Tabular event log (geocoded).** One row per incident tweet: `Date, Time, City, Location, Latitude, Longitude, High_Accuracy, Direction, Type (e.g., VEHICULAR ACCIDENT), Lanes_Blocked, Involved, Tweet, Source`. Coverage starts Aug 2018.

## Fetched
✅ `raw/data_mmda_traffic_spatial.csv` (~0.9 MB). Re-fetch (no API key needed):
```python
import kagglehub
path = kagglehub.dataset_download("esparko/mmda-traffic-incident-data")
```

## Role in the project
Manila incident signal. Two uses: (1) incident-aware exogenous features for congestion models; (2) event-frequency characterization of Manila vs. other cities (RQ1).

## Usage plan
- **Not a congestion index by itself** — it's sparse event data. Combine with [manila_benjiao-traffic-logs](../manila_benjiao-traffic-logs/) (dense road-status series) to study congestion–incident coupling.
- **Models:** classical ML — XGBoost/logistic regression for "incident given congestion state" analysis; spatial hotspot analysis (Getis-Ord) matching the Jakarta dataset's included statistics.
- **Caveat for the paper:** the benjiao logs (2015–16) and this dataset (2018+) do not overlap in time — treat as complementary Manila evidence, not a joined table.
