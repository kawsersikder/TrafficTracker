# Bangladesh — Poribohon-BD (native vehicle classification)

## Credit
- **Source repository:** Mendeley Data, [pwyyg8zmk5 v2](https://data.mendeley.com/datasets/pwyyg8zmk5/2) (published 2020-10-01, United International University). License: **CC BY 4.0**.
- **Authors:** Shaira Tabassum, Md. Sabbir Ullah, Nakib Hossain Al-nur, Swakkhar Shatabda.
- **Research paper:** Tabassum et al., *Poribohon-BD: Bangladeshi native vehicle dataset...*, **Data in Brief** (2020) — verify the exact volume/DOI from the Mendeley record when citing.

## Type
**Image (detection/classification).** 9,058 JPG images, **15 native Bangladeshi vehicle classes + 1 multi-class category** (bus, rickshaw, CNG, launch, boat, horse-cart, etc.). Annotations: **Pascal-VOC-style XML** (LabelImg). Includes augmented images.

## How to fetch (not stored locally)
Browser: <https://data.mendeley.com/datasets/pwyyg8zmk5/2> → **Download All** (no account needed), then upload to Drive for Colab. Anonymous scripted download is blocked by Mendeley.

## Role in the project
Augmentation corpus for the Dhaka vehicle detector — adds class diversity (including waterway/rural vehicles we may exclude) and more examples of rickshaws/CNGs.

## Usage plan
- **Model:** merged into the YOLOv8 training set (convert VOC XML → YOLO txt with a small script; classes mapped to the shared taxonomy).
- **Links to:** [dhaka_dhakaai](../dhaka_dhakaai/) (primary detection set), [dhaka_tfp-bd](../dhaka_tfp-bd/) (evaluation footage).
- **Note:** images are web/roadside collected, not fixed-camera traffic scenes — use for class robustness, not for flow counting.
