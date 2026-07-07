# Dhaka — TFP-BD (Traffic Flow & Pedestrian image dataset)

## Credit
- **Source repository:** Mendeley Data, DOI [10.17632/h8bfgtdp2r.6](https://data.mendeley.com/datasets/h8bfgtdp2r/6). License: **CC BY 4.0**.
- **Research paper:** *TFP-BD: An image dataset for Traffic Flow and Pedestrian movement analysis on Bangladeshi urban roads.* **Data in Brief** 59:111398 (2025). DOI [10.1016/j.dib.2025.111398](https://doi.org/10.1016/j.dib.2025.111398) · [PMC11919376](https://pmc.ncbi.nlm.nih.gov/articles/PMC11919376/)

## Type
**Image (object detection).** 23,678 frames (640×480) extracted from video at four Dhaka locations — Shapla Chattar, Arambag, Bashabo, Abul Hotel — across five daily time periods and varying weather/lighting. Bounding-box annotations for **8 vehicle classes + pedestrians**.

## How to fetch (not stored locally — several GB of images)
Mendeley requires a browser session for bulk download (anonymous API download verified blocked):
1. Open <https://data.mendeley.com/datasets/h8bfgtdp2r/6> → **Download All** (works without an account).
2. Upload the zip to Google Drive, then in Colab: `from google.colab import drive; drive.mount('/content/drive')`.

## Role in the project
Vision side track: quantify Dhaka's **traffic heterogeneity** (vehicle-mix percentages) — the explanatory variable for *why* cross-city transfer succeeds or fails. Also a template: this is exactly the kind of *Data in Brief* paper we aim to publish for our own TrafficTracker dataset.

## Usage plan
- **Model:** deep learning — fine-tune **YOLOv8** (ultralytics) on the annotations; run on Colab/Kaggle GPU.
- **Output feature:** per-location vehicle-mix vector (% rickshaw/CNG/bus/car/pedestrian) → used as a city-level covariate in the transfer analysis.
- **Links to:** [dhaka_dhakaai](../dhaka_dhakaai/) and [dhaka_poribohon-bd](../dhaka_poribohon-bd/) (merge/augment classes for a stronger detector), [dhaka_trafficktracker-own](../dhaka_trafficktracker-own/) (its four locations can serve as CI calibration points).
