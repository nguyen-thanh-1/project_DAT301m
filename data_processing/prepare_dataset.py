"""
Prepare Dataset — Merge & Split cho Face Recognition Pipeline.

Chiến lược:
1. Merge tất cả aligned data (train + val) vào pool chung
2. Split theo CLASS (không phải theo ảnh):
   - Train classes: 80% classes → dùng cho ArcFace classification training
   - Test classes:  20% classes → dùng cho verification evaluation (unseen identities)
3. Trong train classes, split 80/20 ẢNH → train/val (cùng classes, monitor convergence)

Output structure:
    dataset_final/
    ├── train/     # N classes, 80% images mỗi class
    ├── val/       # N classes (CÙNG train), 20% images → monitor convergence
    └── test/      # M classes (KHÁC train), tất cả images → verification eval
"""

import argparse
import os
import random
import shutil
from pathlib import Path
from collections import defaultdict


def collect_all_classes(dirs: list[str]) -> dict[str, list[Path]]:
    """Scan multiple directories, merge classes by folder name."""
    class_images = defaultdict(list)
    for d in dirs:
        root = Path(d)
        if not root.is_dir():
            print(f"  [SKIP] {d} not found")
            continue
        for cls_dir in sorted(root.iterdir()):
            if not cls_dir.is_dir():
                continue
            imgs = [
                p for p in cls_dir.iterdir()
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            ]
            if imgs:
                class_images[cls_dir.name].extend(imgs)
    return class_images


def split_dataset(
    class_images: dict[str, list[Path]],
    output_dir: str,
    test_ratio: float = 0.2,
    val_ratio: float = 0.2,
    min_images: int = 10,
    seed: int = 42,
):
    """
    Split dataset into train/val/test.

    Args:
        class_images: {class_name: [image_paths]}
        output_dir: Output directory
        test_ratio: Ratio of classes for test (verification)
        val_ratio: Ratio of images for val (from train classes)
        min_images: Minimum images per class to include
        seed: Random seed
    """
    random.seed(seed)

    # Filter classes with enough images
    valid_classes = {k: v for k, v in class_images.items() if len(v) >= min_images}
    removed = len(class_images) - len(valid_classes)
    if removed > 0:
        print(f"  [INFO] Removed {removed} classes with < {min_images} images")

    all_classes = sorted(valid_classes.keys())
    random.shuffle(all_classes)

    # Split classes → train vs test
    n_test = max(1, int(len(all_classes) * test_ratio))
    test_classes = set(all_classes[:n_test])
    train_classes = set(all_classes[n_test:])

    print(f"\n  Total valid classes: {len(all_classes)}")
    print(f"  Train classes: {len(train_classes)} ({100 - test_ratio*100:.0f}%)")
    print(f"  Test classes:  {len(test_classes)} ({test_ratio*100:.0f}%)")

    out = Path(output_dir)
    for split in ["train", "val", "test"]:
        (out / split).mkdir(parents=True, exist_ok=True)

    stats = {"train": 0, "val": 0, "test": 0}

    # --- Test set: all images from test classes ---
    for cls_name in sorted(test_classes):
        imgs = valid_classes[cls_name]
        cls_out = out / "test" / cls_name
        cls_out.mkdir(parents=True, exist_ok=True)
        for img in imgs:
            dst = cls_out / img.name
            if not dst.exists():
                shutil.copy2(str(img), str(dst))
            stats["test"] += 1

    # --- Train/Val: split images from train classes ---
    for cls_name in sorted(train_classes):
        imgs = valid_classes[cls_name]
        random.shuffle(imgs)

        n_val = max(1, int(len(imgs) * val_ratio))
        val_imgs = imgs[:n_val]
        train_imgs = imgs[n_val:]

        # Train
        cls_train = out / "train" / cls_name
        cls_train.mkdir(parents=True, exist_ok=True)
        for img in train_imgs:
            dst = cls_train / img.name
            if not dst.exists():
                shutil.copy2(str(img), str(dst))
            stats["train"] += 1

        # Val (same classes as train, different images)
        cls_val = out / "val" / cls_name
        cls_val.mkdir(parents=True, exist_ok=True)
        for img in val_imgs:
            dst = cls_val / img.name
            if not dst.exists():
                shutil.copy2(str(img), str(dst))
            stats["val"] += 1

    print(f"\n  Output: {output_dir}")
    print(f"  Train: {stats['train']} images in {len(train_classes)} classes")
    print(f"  Val:   {stats['val']} images in {len(train_classes)} classes (same as train)")
    print(f"  Test:  {stats['test']} images in {len(test_classes)} classes (unseen)")
    print(f"  Total: {sum(stats.values())} images")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for face recognition training")
    parser.add_argument(
        "--input_dirs", nargs="+",
        default=["dataset_aligned/train", "dataset_aligned/val"],
        help="Input directories containing class subfolders (will be merged)"
    )
    parser.add_argument("--output", type=str, default="dataset_final",
                        help="Output directory")
    parser.add_argument("--test_ratio", type=float, default=0.2,
                        help="Ratio of classes reserved for verification test")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="Ratio of images from train classes used for val")
    parser.add_argument("--min_images", type=int, default=10,
                        help="Minimum images per class to include")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("Prepare Dataset for Face Recognition")
    print("=" * 60)

    print(f"\n[1/2] Scanning input directories...")
    for d in args.input_dirs:
        print(f"  -> {d}")
    class_images = collect_all_classes(args.input_dirs)
    total_images = sum(len(v) for v in class_images.values())
    print(f"  Found {len(class_images)} classes, {total_images} total images")

    print(f"\n[2/2] Splitting dataset...")
    split_dataset(
        class_images,
        args.output,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        min_images=args.min_images,
        seed=args.seed,
    )

    print("\n" + "=" * 60)
    print("[OK] Dataset preparation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()


# =====================================================================
# Optimized tf.data Pipeline & Heavy Augmentation for MobileFaceNet
# =====================================================================
import tensorflow as tf

def random_erasing(image, p=0.5):
    """Random Erasing / Cutout augmentation.
    Masks a random 10% - 20% area of the image (L in [35, 50] for 112x112 image)
    with either black pixels or random noise.
    """
    if tf.random.uniform([]) > p:
        return image

    img_h, img_w, img_c = 112, 112, 3
    
    # Random side length representing approx 10%-20% of image area
    L = tf.random.uniform([], minval=35, maxval=51, dtype=tf.int32)
    
    x = tf.random.uniform([], minval=0, maxval=img_h - L, dtype=tf.int32)
    y = tf.random.uniform([], minval=0, maxval=img_w - L, dtype=tf.int32)
    
    row_indices = tf.range(img_h)[:, tf.newaxis, tf.newaxis]
    col_indices = tf.range(img_w)[tf.newaxis, :, tf.newaxis]
    
    mask_row = tf.logical_and(row_indices >= x, row_indices < x + L)
    mask_col = tf.logical_and(col_indices >= y, col_indices < y + L)
    mask = tf.logical_and(mask_row, mask_col)
    
    # 50% chance filling with black (zeros), 50% chance random noise in [-1, 1]
    fill_noise = tf.random.uniform(shape=[img_h, img_w, img_c], minval=-1.0, maxval=1.0)
    fill_value = tf.where(tf.random.uniform([]) > 0.5, tf.zeros_like(image), fill_noise)
    
    image = tf.where(mask, fill_value, image)
    return image

def random_blur(image, p=0.2):
    """Simulate blur via random downscale+upscale (out-of-focus / motion blur proxy).
    
    Downscales image by 2x-4x then upscales back to original size.
    No external dependencies required.
    """
    if tf.random.uniform([]) > p:
        return image
    
    orig_h = tf.shape(image)[0]
    orig_w = tf.shape(image)[1]
    scale = tf.random.uniform([], minval=2, maxval=4, dtype=tf.int32)
    
    small_h = tf.maximum(orig_h // scale, 8)
    small_w = tf.maximum(orig_w // scale, 8)
    
    image = tf.image.resize(tf.expand_dims(image, 0), [small_h, small_w])
    image = tf.image.resize(image, [orig_h, orig_w])
    return tf.squeeze(image, 0)

def heavy_augment(image, label, use_erasing=False, use_blur=False):
    """Applies augmentation per ArcFace paper spec (flip, brightness, contrast ONLY).
    
    Spec explicitly states:
    - Random horizontal flip (p=0.5)
    - Random brightness adjustment (delta=0.2)
    - Random contrast adjustment (lower=0.8, upper=1.2)
    
    Optional extras (off by default per paper spec):
    - Random Erasing / Cutout (p=0.3) when use_erasing=True
    - Random Blur (p=0.2, via downscale+upscale) when use_blur=True
    """
    # Random horizontal flip
    image = tf.image.random_flip_left_right(image)
    
    # Color Jitter (brightness + contrast only per spec)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    
    # Optional Random Blur (domain generalization)
    if use_blur:
        image = random_blur(image, p=0.2)
    
    # Optional Random Erasing (Cutout)
    if use_erasing:
        image = random_erasing(image, p=0.3)
    
    image = tf.clip_by_value(image, -1.0, 1.0)
    return (image, label), label

def build_training_dataset(image_paths, labels, batch_size=256, is_training=True, cache_path=None, use_erasing=False, use_blur=False):
    """Builds a high-performance, heavily-augmented tf.data.Dataset pipeline.
    
    Tuned for 16GB VRAM GPUs with prefetching, parallel loading, and caching.
    
    Args:
        use_erasing: Enable Random Erasing / Cutout augmentation (p=0.3).
        use_blur: Enable random blur augmentation (p=0.2, downscale+upscale).
    """
    # 1. Slice paths and labels
    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    
    # 2. Shuffle strings in memory (very cheap)
    if is_training:
        ds = ds.shuffle(buffer_size=len(image_paths), reshuffle_each_iteration=True)
    
    # 3. Load & preprocess images
    def load_img(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, [112, 112])
        img.set_shape([112, 112, 3])
        # Normalize to [-1.0, 1.0]
        img = (tf.cast(img, tf.float32) - 127.5) / 128.0
        return img, label

    ds = ds.map(load_img, num_parallel_calls=tf.data.AUTOTUNE)
    
    # 4. Cache — use disk to avoid blowing up RAM with 4.7M images
    if cache_path:
        ds = ds.cache(cache_path)
    # else: skip caching entirely — without disk path, cache() stores everything in RAM
    # which OOMs on MS1M (4.7M images × 112×112×3 × 4 bytes ≈ 70 GB)
    
    # 5. Apply augmentations and format inputs
    if is_training:
        ds = ds.map(lambda img, lbl: heavy_augment(img, lbl, use_erasing=use_erasing, use_blur=use_blur),
                    num_parallel_calls=tf.data.AUTOTUNE)
    else:
        # Just format for model: ((image, label), label)
        ds = ds.map(lambda img, lbl: ((img, lbl), lbl), num_parallel_calls=tf.data.AUTOTUNE)
        
    # 6. Batch and prefetch
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    return ds
