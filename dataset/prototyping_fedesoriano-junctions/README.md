# Prototyping — Traffic Prediction Dataset (fedesoriano, Kaggle)

## Credit
- **Source:** Kaggle [fedesoriano/traffic-prediction-dataset](https://www.kaggle.com/datasets/fedesoriano/traffic-prediction-dataset).
- **Research paper:** none — city and collection method are not documented by the uploader.

## Type
**Tabular time series.** `raw/traffic.csv`: hourly vehicle counts at **4 junctions** (`DateTime, Junction, Vehicles, ID`), Nov 2015 – Jun 2017.

## Fetched
✅ `raw/traffic.csv` (~0.3 MB). Re-fetch:
```python
import kagglehub
path = kagglehub.dataset_download("fedesoriano/traffic-prediction-dataset")
```

## Role in the project — ⚠ strictly limited
**Model debugging only.** Because the city is unknown, this dataset **must not appear in the paper as a compared city**. Use it to smoke-test the LSTM/GRU/XGBoost training loops in minutes before pointing them at the real (much larger) city datasets.

## Usage plan
- Wire up the full training/eval harness (windowing, scaling, MAE/RMSE reporting) on this tiny CSV first; then swap in Bangkok/Manila/Jakarta loaders unchanged.
