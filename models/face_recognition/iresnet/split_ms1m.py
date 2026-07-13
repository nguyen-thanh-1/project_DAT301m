"""
Split MS1M-ArcFace Dataset for iResNet50 + ArcFace Training.

Input:  ms1m-arcface/           (85,742 identity folders, flat structure)
Output: ms1m_arcface_dataset/
        ├── train/     # 80% classes, 80% images each  → ArcFace classification
        ├── val/       # 80% classes (SAME as train), 20% images → monitor convergence
        └── test/      # 20% classes (UNSEEN), all images → verification evaluation

Key design decisions:
  - Split by CLASS first (train vs test identities are disjoint)
  - Then split by IMAGE within train classes (train vs val share classes)
  - Filter out classes with < 5 images (too few for meaningful training)
  - Uses symlinks instead of copying to save ~20GB disk space
  - Deterministic split (seed=42) for reproducibility

Usage:
    python split_ms1m.py
    python split_ms1m.py --input_dir path/to/ms1m-arcface --output_dir ms1m_arcface_dataset
    python split_ms1m.py --min_images 10 --test_ratio 0.1
"""

import argparse
import os
import random
import sys
from pathlib import Path


def split_ms1m(
    input_dir: str,
    output_dir: str,
    test_ratio: float = 0.20,
    val_ratio: float = 0.20,
    min_images: int = 5,
    seed: int = 42,
    use_symlinks: bool = True,
):
    """Split MS1M-ArcFace into train/val/test.

    Args:
        input_dir:    Path to ms1m-arcface root (contains identity folders)
        output_dir:   Output path for the split dataset
        test_ratio:   Fraction of CLASSES reserved for verification test (unseen)
        val_ratio:    Fraction of IMAGES from train classes used for val
        min_images:   Minimum images per class to include (filter noise)
        seed:         Random seed for reproducibility
        use_symlinks: Use symlinks instead of copying files (saves disk space)
    """
    random.seed(seed)
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.is_dir():
        print(f"[ERROR] Input directory not found: {input_dir}")
        sys.exit(1)

    # ── Step 1: Scan all identity folders ──────────────────────────────
    print("=" * 60)
    print("Split MS1M-ArcFace Dataset")
    print("=" * 60)
    print(f"\n[1/4] Scanning {input_dir} ...")

    class_images = {}
    total_scanned = 0
    skipped_small = 0

    for cls_dir in sorted(input_path.iterdir()):
        if not cls_dir.is_dir():
            continue
        total_scanned += 1

        imgs = sorted([
            p for p in cls_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ])

        if len(imgs) >= min_images:
            class_images[cls_dir.name] = imgs
        else:
            skipped_small += 1

        # Progress indicator
        if total_scanned % 10000 == 0:
            print(f"  ... scanned {total_scanned:,} folders ({len(class_images):,} kept)")

    total_images = sum(len(v) for v in class_images.values())
    print(f"  Scanned:  {total_scanned:,} identity folders")
    print(f"  Kept:     {len(class_images):,} classes (>= {min_images} images)")
    print(f"  Skipped:  {skipped_small:,} classes (< {min_images} images)")
    print(f"  Total:    {total_images:,} images")

    # ── Step 2: Split classes → train vs test ──────────────────────────
    print(f"\n[2/4] Splitting classes (test_ratio={test_ratio}) ...")

    all_classes = sorted(class_images.keys())
    random.shuffle(all_classes)

    n_test = max(1, int(len(all_classes) * test_ratio))
    test_classes = set(all_classes[:n_test])
    train_classes = sorted(all_classes[n_test:])

    print(f"  Train classes: {len(train_classes):,} ({(1 - test_ratio) * 100:.0f}%)")
    print(f"  Test classes:  {len(test_classes):,} ({test_ratio * 100:.0f}%)")

    # ── Step 3: Create output directories ──────────────────────────────
    print(f"\n[3/4] Creating output structure in {output_dir} ...")

    for split in ["train", "val", "test"]:
        (output_path / split).mkdir(parents=True, exist_ok=True)

    copy_fn = os.symlink if use_symlinks else _copy_file
    link_label = "symlink" if use_symlinks else "copy"

    stats = {"train": 0, "val": 0, "test": 0}
    stats_classes = {"train": 0, "val": 0, "test": 0}

    # ── Test set: all images from test classes ─────────────────────────
    for cls_name in sorted(test_classes):
        imgs = class_images[cls_name]
        cls_out = output_path / "test" / cls_name
        cls_out.mkdir(parents=True, exist_ok=True)

        for img in imgs:
            dst = cls_out / img.name
            if not dst.exists():
                copy_fn(str(img.resolve()), str(dst))
            stats["test"] += 1
        stats_classes["test"] += 1

    # ── Train/Val: split images within each train class ────────────────
    for cls_name in train_classes:
        imgs = class_images[cls_name]
        random.shuffle(imgs)

        n_val = max(1, int(len(imgs) * val_ratio))
        val_imgs = imgs[:n_val]
        train_imgs = imgs[n_val:]

        # Train
        cls_train = output_path / "train" / cls_name
        cls_train.mkdir(parents=True, exist_ok=True)
        for img in train_imgs:
            dst = cls_train / img.name
            if not dst.exists():
                copy_fn(str(img.resolve()), str(dst))
            stats["train"] += 1

        # Val (SAME classes as train, different images)
        cls_val = output_path / "val" / cls_name
        cls_val.mkdir(parents=True, exist_ok=True)
        for img in val_imgs:
            dst = cls_val / img.name
            if not dst.exists():
                copy_fn(str(img.resolve()), str(dst))
            stats["val"] += 1

        stats_classes["train"] += 1
        stats_classes["val"] += 1

        # Progress
        done = stats_classes["train"]
        if done % 5000 == 0:
            print(f"  ... processed {done:,}/{len(train_classes):,} train classes")

    # ── Step 4: Summary ────────────────────────────────────────────────
    print(f"\n[4/4] Done! ({link_label} mode)")
    print("-" * 60)
    print(f"  Output:  {output_dir}")
    print(f"  Train:   {stats['train']:>10,} images in {stats_classes['train']:>6,} classes")
    print(f"  Val:     {stats['val']:>10,} images in {stats_classes['val']:>6,} classes (same as train)")
    print(f"  Test:    {stats['test']:>10,} images in {stats_classes['test']:>6,} classes (unseen)")
    print(f"  Total:   {sum(stats.values()):>10,} images")
    print("-" * 60)

    # Verify: train and val must share classes, test must be disjoint
    train_set = set(os.listdir(output_path / "train"))
    val_set = set(os.listdir(output_path / "val"))
    test_set = set(os.listdir(output_path / "test"))

    assert train_set == val_set, "ERROR: train and val must have same classes!"
    assert len(train_set & test_set) == 0, "ERROR: test classes must be disjoint from train!"
    print("\n  [OK] Verified: train/val share classes, test is disjoint")
    print("=" * 60)

    return stats


def _copy_file(src, dst):
    """Fallback file copy for systems that don't support symlinks."""
    import shutil
    shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(
        description="Split MS1M-ArcFace dataset into train/val/test for ArcFace training"
    )
    parser.add_argument(
        "--input_dir", type=str,
        default=r"C:\Users\Admin\Desktop\Project_DAT301m\ms1m-arcface",
        help="Path to ms1m-arcface root folder"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=r"C:\Users\Admin\Desktop\Project_DAT301m\ms1m_arcface_dataset",
        help="Output path for split dataset"
    )
    parser.add_argument("--test_ratio", type=float, default=0.10,
                        help="Fraction of classes for test set (verification)")
    parser.add_argument("--val_ratio", type=float, default=0.10,
                        help="Fraction of images for val set (within train classes)")
    parser.add_argument("--min_images", type=int, default=5,
                        help="Minimum images per class to keep")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy", action="store_true", default=False,
                        help="Copy files instead of creating symlinks (uses more disk)")
    args = parser.parse_args()

    split_ms1m(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        min_images=args.min_images,
        seed=args.seed,
        use_symlinks=not args.copy,
    )


if __name__ == "__main__":
    main()
