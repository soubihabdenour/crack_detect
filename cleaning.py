"""Utility script to clean a folder of images.

Operations:
- Remove corrupted files
- Remove very small images
- Remove duplicates (perceptual hash)
- Remove blurry images (Laplacian variance)
- Remove extreme brightness images

Usage:
  python cleaning.py --folder dataset/train/defect

Results are logged under ``logs/clean_log_*.txt`` and removed files are moved
into ``trash/`` for manual review.
"""

import os
import cv2
import shutil
import imagehash
import numpy as np
from PIL import Image
from datetime import datetime
import argparse

TRASH_DIR = "trash"
LOG_DIR = "logs"

os.makedirs(TRASH_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

log_file = os.path.join(LOG_DIR, f"clean_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")


def log(msg: str) -> None:
    print(msg)
    with open(log_file, "a") as f:
        f.write(msg + "\n")


def move_to_trash(path: str) -> None:
    fname = os.path.basename(path)
    dest = os.path.join(TRASH_DIR, fname)
    shutil.move(path, dest)
    log(f"[MOVED] {fname} → trash/")


def remove_corrupted(folder: str) -> None:
    """Move images that cannot be opened/verified."""
    log("\n=== Checking corrupted images ===")
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        try:
            img = Image.open(path)
            img.verify()
        except Exception:
            move_to_trash(path)


def remove_small(folder: str, min_size: int = 28) -> None:
    """Move images whose width or height is below ``min_size``."""
    log("\n=== Checking small images ===")
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        try:
            img = Image.open(path)
            w, h = img.size
            if w < min_size or h < min_size:
                move_to_trash(path)
        except Exception:
            move_to_trash(path)


def remove_duplicates(folder: str) -> None:
    """Move duplicate images based on average perceptual hash."""
    log("\n=== Checking duplicate images ===")
    hashes = {}
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        try:
            h = imagehash.average_hash(Image.open(path))
        except:
            move_to_trash(path)
            continue

        if h in hashes:
            log(f"Duplicate found: {fname} (same as {hashes[h]})")
            move_to_trash(path)
        else:
            hashes[h] = fname


def variance_of_laplacian(image: np.ndarray) -> float:
    return cv2.Laplacian(image, cv2.CV_64F).var()


def remove_blurry(folder: str, threshold: float = 30) -> None:
    """Move images whose Laplacian variance is below ``threshold``."""
    log("\n=== Checking blurry images ===")
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        img = cv2.imread(path)
        if img is None:
            move_to_trash(path)
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        score = variance_of_laplacian(gray)

        if score < threshold:
            log(f"Blurry: {fname} (score={score:.2f})")
            move_to_trash(path)


def remove_brightness_extreme(folder: str, low: float = 20, high: float = 220) -> None:
    """Move images with average brightness outside [low, high]."""
    log("\n=== Checking brightness extremes ===")
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            move_to_trash(path)
            continue
        mean = img.mean()

        if mean < low or mean > high:
            log(f"Bad exposure: {fname} (brightness={mean:.2f})")
            move_to_trash(path)


def run_cleaning(folder: str, min_size: int = 28, blur_thresh: float = 30, low: float = 20, high: float = 220) -> None:
    log(f"\n===============================")
    log(f"CLEANING FOLDER: {folder}")
    log(f"===============================\n")

    remove_corrupted(folder)
    remove_small(folder, min_size=min_size)
    remove_duplicates(folder)
    remove_blurry(folder, threshold=blur_thresh)
    remove_brightness_extreme(folder, low=low, high=high)

    log("\n=== Cleaning finished ===\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean a folder of images")
    parser.add_argument("--folder", required=True, help="Folder to clean (e.g., dataset/train/defect)")
    parser.add_argument("--min-size", type=int, default=28, help="Minimum width/height to keep")
    parser.add_argument("--blur-thresh", type=float, default=30, help="Laplacian variance threshold")
    parser.add_argument("--low", type=float, default=20, help="Low brightness cutoff")
    parser.add_argument("--high", type=float, default=220, help="High brightness cutoff")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_cleaning(args.folder, min_size=args.min_size, blur_thresh=args.blur_thresh, low=args.low, high=args.high)