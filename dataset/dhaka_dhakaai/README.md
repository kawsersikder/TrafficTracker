# Dhaka — DhakaAI (vehicle detection)

## Credit
- **Source repositories:** Kaggle [rifat963/dhakaai-dhaka-based-traffic-detection-dataset](https://www.kaggle.com/datasets/rifat963/dhakaai-dhaka-based-traffic-detection-dataset) (fetched here); also mirrored on [Roboflow Universe](https://universe.roboflow.com/traffic-wake5/dhakaai-gx36d) and Harvard Dataverse.
- **Origin:** *Dhaka AI 2020* traffic-detection challenge (Dhaka, Bangladesh).
- **Research paper:** no single canonical paper — cite the Harvard Dataverse/Kaggle record plus the challenge; multiple YOLO-based papers used it (cite the specific one you benchmark against after reading it).

## Type
**Image (object detection).** 3,953 images, **21 vehicle classes** (bus, rickshaw, CNG/auto-rickshaw, motorbike, truck, van, etc.), YOLO-style annotations, train/test split. ~1.4 GB.

## Fetched
✅ `raw/` (1.43 GB, Kaggle version 2). Re-fetch anywhere (no API key needed):
```python
import kagglehub
path = kagglehub.dataset_download("rifat963/dhakaai-dhaka-based-traffic-detection-dataset")
```

## Role in the project
Main training corpus for the Dhaka vehicle detector — 21 fine-grained native classes make it the best heterogeneity-measurement source.

## Usage plan
- **Model:** deep learning — **YOLOv8** fine-tuned on the 21 classes (Colab Pro GPU; ~1–2 h for 50 epochs at 640px).
- **Links to:** [dhaka_tfp-bd](../dhaka_tfp-bd/) + [dhaka_poribohon-bd](../dhaka_poribohon-bd/) — harmonize class taxonomies (e.g., map all three to a shared 8-class scheme) and train one detector on the union; report per-dataset generalization.
- **Feeds:** vehicle-mix covariates for the transfer-learning analysis (see [dataset/README.md](../README.md), vision side track).
