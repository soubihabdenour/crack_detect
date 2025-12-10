# crack_detect

Jupyter notebook for training a binary classifier that distinguishes **defect** vs. **no_defect** grayscale images.

Expected dataset layout:

```
dataset/
  train/
    defect/
    no_defect/
  val/
    defect/
    no_defect/
  test/
    defect/
    no_defect/
```

Open `defect_classification_experiments.ipynb` in Jupyter to try multiple augmentation pipelines and optimizer hyperparameters. The notebook logs metrics per experiment and saves raw histories to `experiment_results.json`.
