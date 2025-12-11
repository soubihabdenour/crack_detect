# crack_detect

Python training and evaluation scripts for a binary classifier that distinguishes **defect (crack)** vs. **no_defect** grayscale image patches.

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

## Setup
Install dependencies (PyTorch and torchvision wheels should match your CUDA setup):

```
pip install -r requirements.txt
```

## Training
Train a classifier with ImageNet-pretrained backbones and balanced sampling:

```
python -m src.train /path/to/dataset \
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
- `--backbone {resnet18,resnet34,efficientnet_b0}`: choose architecture
- `--no-pretrained`: disable ImageNet initialization
- `--no-balance`: disable weighted sampling if classes are already balanced
- `--patience`: early stopping patience on validation loss
- `--image-size`: resize/crop target (increase to 256–288 for more detail)

## Evaluation
Evaluate a trained model on any ImageFolder-formatted directory (validation or test split):

```
python -m src.eval /path/to/dataset/val \
  --weights runs/resnet18_baseline/best_model.pth \
  --backbone resnet18 \
  --image-size 224
```

The script prints JSON with loss and accuracy.

## Notes
- Images are converted from grayscale to 3 channels to match pretrained models.
- Augmentations (random resized crop, flips, mild jitter) are applied only during training.
- See `REPORT.md` for the full training strategy, ideation, and tips for reaching the target accuracies.
