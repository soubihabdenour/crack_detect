# Crack vs. No-Crack Classifier Report

## Goal
Binary classification of grayscale metal-surface patches into **crack** (defect) and **no-crack** (no_defect) classes while reaching ≥99% validation accuracy and ≥95% hidden-test accuracy.

## Dataset handling
- Expected layout follows `ImageFolder` convention:
  - `dataset/train/{defect,no_defect}`
  - `dataset/val/{defect,no_defect}`
  - `dataset/test/{defect,no_defect}` (optional for local testing)
- Images are grayscale; transforms convert them to 3 channels to leverage ImageNet pretrained backbones.
- A weighted sampler balances classes during training to mitigate skew in the provided split ranges.

## Model architecture
- Base backbone: `resnet18` (configurable to `resnet34` or `efficientnet_b0`).
- ImageNet weights provide strong initialization for texture and edge detection.
- Custom head: dropout → 128-unit ReLU → dropout → 1-logit output for binary classification.

## Training strategy
- Loss: `BCEWithLogitsLoss` on raw logits.
- Optimizer: `AdamW` with weight decay; cosine annealing scheduler over epochs.
- Data augmentation to counter overfitting and improve robustness to lighting/pose:
  - Resize & random resized crop
  - Horizontal/vertical flips
  - Mild brightness/contrast jitter
- Normalization: mean 0.5 / std 0.25 per channel after grayscale expansion.
- Early stopping on validation accuracy with configurable patience.
- Class-balanced sampling to avoid over-predicting the dominant class.

## Evaluation
- Validation monitored every epoch; best weights saved to `best_model.pth`.
- Separate `eval.py` script reports loss and accuracy on any folder structured for `ImageFolder`.

## Recommendations for target accuracy
1. **Image size**: use `--image-size 256` or `288` if GPU memory allows for finer detail.
2. **Backbone**: switch to `--backbone efficientnet_b0` for stronger features on small textures.
3. **Epochs**: extend to 40–60 epochs with patience 8–10; cosine annealing handles longer schedules.
4. **Augmentations**: consider slight Gaussian blur and random rotation (±5°) if overfitting persists.
5. **Validation hygiene**: ensure no overlap between train/val indices (important given filename ranges).

## Reproducibility & outputs
- `config.json` records run hyperparameters.
- `history.json` stores loss/accuracy per epoch.
- `best_model.pth` contains the best-performing weights for downstream evaluation.
