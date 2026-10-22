# Computer-Vision-DeliveryRobot

Midterm project for **Computer Vision** (BS Computer Science 4B), Group Specialization 4: **Autonomous Delivery Robot**.

The Midterm covers the **SEE → RECOGNIZE** stage: a camera-based system that recognizes printed destination markers for a 4WD delivery robot. Robot navigation, delivery zones, and the LED/buzzer signal belong to the Final.

## Team

- Aguilar, Vince Clifford C.
- Benosa, Nhel Adam S.
- Caldo, John Paul B.
- Llesis, Roinel James A.
- Olivares, Elyssa
- Sario, Joshua Q.

## Overview

The robot must identify its assigned destination, ignore other markers, and stop at the correct delivery zone. This repository contains the recognition part: an image classifier that identifies five printed markers (A to E) from a camera image and reports a confidence score. Markers A, B and C are the required destination markers; D and E are additional markers.

## Dataset

Self-made: created and gathered by Nhel Adam Benosa using a smartphone camera. One video per marker was recorded while moving the camera closer and farther, tilting it, and rotating the paper. Frames were extracted with OpenCV.

| Class | Original frames | Train | Val | Test | Unseen (dim light) |
|---|---|---|---|---|---|
| A | 395 | 273 | 53 | 57 | 336 |
| B | 385 | 266 | 52 | 55 | 359 |
| C | 399 | 276 | 54 | 57 | 378 |
| D | 441 | 305 | 60 | 64 | 366 |
| E | 380 | 263 | 51 | 54 | 317 |
| **Total** | **2,000** | **1,383** | **270** | **287** | **1,756** |

- **Split:** by time segment within each video (first 70% train, next 15% validation, last 15% test, with a 3-frame gap at each boundary) so near-identical neighboring frames never appear in different sets.
- **Unseen set:** recorded separately in dim lighting and a different background, used only for final evaluation.
- The full dataset is not stored in this repository because of its size. A small sample is in `dataset_sample/`. Full dataset: https://drive.google.com/file/d/10oFsQGbYUzdecyuV2D0BHWxjkK85pctn/view?usp=drive_link and https://drive.google.com/file/d/1jzS7lQEj6-gixgbKkZLZPr1G-lUT9AHz/view?usp=drive_link.

## Method

- **Model:** MobileNetV2 pretrained on ImageNet (transfer learning), input 224 × 224.
- **Head:** global average pooling, dropout 0.3, dense layer with 5 outputs and softmax.
- **Augmentation (training only):** rotation, zoom, shift, brightness, contrast. No horizontal flip.
- **Training:** Stage 1 trains the new head (10 epochs, Adam 1e-3). Stage 2 fine-tunes the top 30 layers (8 epochs, Adam 1e-5). Batch size 32, sparse categorical cross-entropy.
- **Demo:** the webcam script shows the predicted label and confidence, and shows `unknown` below 70% confidence.

## Results

| Evaluation set | Frames | Accuracy |
|---|---|---|
| Test set (same conditions) | 287 | 100.00% |
| Unseen set (dim light, new background) | 1,756 | 99.89% |

Only 2 of 1,756 unseen frames were misclassified. Confusion matrices, training curves, and failure examples are in `results/`.

## Repository structure

```
Computer-Vision-DeliveryRobot/
├── README.md
├── requirements.txt
├── src/
│   └── webcam_demo.py
├── notebooks/
│   └── marker_training.ipynb
├── model/
│   ├── marker_model.keras
│   └── class_names.json
├── results/
│   ├── training_curves.png
│   ├── cm_test.png
│   ├── cm_dim.png
│   └── failures.png
├── dataset/
└── docs/
    └── Team_ButterBonnies_Midterm_Report_BSCS4B.pdf
```

## How to run

1. Install dependencies: `pip install tensorflow opencv-python`
2. Extract frames from the marker videos: `python src/extract_frames.py`
3. Split the dataset: `python src/split_dataset.py`
4. Train and evaluate in Google Colab using `notebooks/marker_training.ipynb`.
5. Run the live demo (needs `marker_model.keras` and `class_names.json` in the same folder): `python src/webcam_demo.py`. Press `q` to quit and `s` to save a screenshot.

## Limitations

- The model classifies the whole frame and does not locate the marker, so it cannot yet handle two markers in view.
- There is no dedicated "no marker" class; empty scenes are rejected only by the confidence threshold.
- Only one set of printed markers and one capture phone were used; occlusion, long distance, and motion blur were not tested.
- Not yet tested on the robot's own camera.

## Future work (Final)

Add a "no marker" class, marker localization (detection), delivery decision logic (continue or stop at the assigned destination), robot control, and the LED/buzzer delivery signal.

## References

- Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. *CVPR*. https://doi.org/10.1109/CVPR.2018.00474
- Deng, J., et al. (2009). ImageNet: A large-scale hierarchical image database. *CVPR*. https://doi.org/10.1109/CVPR.2009.5206848
