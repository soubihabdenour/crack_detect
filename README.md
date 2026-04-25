# crack_detect

Lightweight PyTorch code for training, tuning, and evaluating a binary classifier that distinguishes **defect (crack)** vs. **no_defect** grayscale image patches.

## Dataset layout
Images must follow the `torchvision.datasets.ImageFolder` structure:

```
dataset/
  train/
    defect/
    no_defect/
  val/
    defect/
    no_defect/
  test/                # optional for local testing
    defect/
    no_defect/
```

If you only have a single directory with class subfolders (for example the Kaggle surface crack dataset), the tuning script will automatically create a reproducible 70/15/15 train/val/test split for you.

## Setup
Install dependencies (ensure the PyTorch/torchvision wheels match your CUDA setup):

```
pip install -r requirements.txt
```

## Training
Train a classifier with ImageNet-pretrained backbones and balanced sampling:

```
python -m src.train /dataset \
  --output-dir runs/resnet18_baseline \
  --backbone resnet18 \
  --batch-size 32 \
  --epochs 30 \
  --image-size 224
```

Key outputs written to `--output-dir`:
- `config.json`: hyperparameters used for the run
- `history.json`: per-epoch training/validation loss and accuracy
- `checkpoint.pth`: last epoch checkpoint with optimizer/scheduler state
- `best_model.pth`: best validation-accuracy weights

Useful flags:
- `--backbone {resnet18,resnet34,efficientnet_b0,simple_cnn}`: choose architecture
- `--no-pretrained`: disable ImageNet initialization
- `--no-balance`: disable weighted sampling if classes are already balanced
- `--patience`: early stopping patience on validation loss
- `--image-size`: resize/crop target (increase to 256–288 for more detail)

## Hyperparameter search on a single-folder dataset
Run Optuna sweeps that reuse a fixed random split across all trials. This is useful when starting from a dataset organized as `root/defect` and `root/no_defect` without predefined train/val folders:

```
python -m src.tune /path/to/full_dataset \
  --trials 20 \
  --epochs 15 \
  --output-dir runs/tuning_kaggle
```

Artifacts per trial are written to `runs/tuning_kaggle/trial_XXX/` with the best model weights and validation/test metrics. The top-performing checkpoint and a summary JSON are copied into `runs/tuning_kaggle/best/` for quick reuse in `src.eval`.

## Evaluation
Evaluate a trained model on any ImageFolder-formatted directory (validation or test split):

```
python -m src.eval /path/to/dataset/val \
  --weights runs/resnet18_baseline/best_model.pth \
  --backbone resnet18 \
  --image-size 224 \
  --debug-loader   # optional: prints a one-batch summary for sanity check
```

The script prints JSON with loss and accuracy.

## Data cleaning helper
Use the included cleaning utility to move likely-bad images to `trash/` for manual review and keep an audit in `logs/`:

```
python cleaning.py --folder dataset/train/defect \
  --min-size 28 \
  --blur-thresh 30 \
  --low 20 --high 220
```

## Notes
- Input grayscale images are converted to 3 channels to match pretrained models.
- Portrait and landscape inputs are padded to square before resizing so validation patches align with the training aspect ratio.
- Augmentations are applied only during training. You can choose `augment` strength inside `src/data.py` (light/strong/ultra) or adjust `image_size`.
- See `REPORT.md` for a deeper discussion of training strategy and results.
