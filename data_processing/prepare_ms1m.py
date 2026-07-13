"""
prepare_ms1m.py — Split MS1M-ArcFace identities into train/val/test using NTFS junctions.
No files are copied — just directory references.
"""

import argparse
import os
import random
import subprocess
import sys
from pathlib import Path

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def get_identities(source_dir, min_images=2):
    """Get identity folder names with enough images."""
    source_dir = Path(source_dir)
    identities = []
    for d in sorted(source_dir.iterdir()):
        if not d.is_dir():
            continue
        imgs = [p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS]
        if len(imgs) >= min_images:
            identities.append(d.name)
    return identities


def create_junctions_batch(output_dir, source_dir, identity_ids, split_name):
    out_dir = Path(output_dir)
    src_dir = Path(source_dir)
    split_dir = out_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    existing = 0
    batch_lines = []
    for identity in identity_ids:
        target = split_dir / identity
        source = src_dir / identity
        if target.exists():
            existing += 1
        else:
            batch_lines.append(f'mklink /J "{target}" "{source}"')

    n_total = len(identity_ids)
    n_new = len(batch_lines)

    if n_new == 0:
        print(f"  [{split_name}] All {n_total} junctions already exist.")
        return

    print(f"  [{split_name}] {n_new} new / {n_total} total identities...")

    batch_path = out_dir / f"_create_{split_name}.bat"
    batch_path.write_text('@echo off\n' + '\n'.join(batch_lines), encoding='ascii')

    result = subprocess.run(
        ['cmd', '/c', str(batch_path)],
        shell=True, capture_output=True, text=True,
    )
    batch_path.unlink()

    if result.returncode != 0:
        print(f"  [Warning] {result.stderr[:300]}")
    print(f"  [{split_name}] Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Split MS1M-ArcFace into train/val/test using NTFS junctions"
    )
    parser.add_argument("--source",
                        default=r"C:\Users\Admin\Desktop\Project_DAT301m\ms1m-arcface")
    parser.add_argument("--output",
                        default=r"C:\Users\Admin\Desktop\Project_DAT301m\dataset_ms1m")
    parser.add_argument("--train_ratio", type=float, default=0.93)
    parser.add_argument("--val_ratio", type=float, default=0.035)
    parser.add_argument("--min_images", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source_dir = Path(args.source).resolve()
    output_dir = Path(args.output)

    if not source_dir.is_dir():
        print(f"ERROR: Source not found: {source_dir}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  Prepare MS1M-ArcFace Dataset")
    print("=" * 60)

    print(f"\n[1/4] Scanning identities in {source_dir}...")
    identities = get_identities(source_dir, args.min_images)
    print(f"  Found {len(identities)} identities with >= {args.min_images} images")

    print(f"\n[2/4] Splitting (train={args.train_ratio}, val={args.val_ratio}, seed={args.seed})...")
    rng = random.Random(args.seed)
    rng.shuffle(identities)

    n = len(identities)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)

    train_ids = identities[:n_train]
    val_ids = identities[n_train:n_train + n_val]
    test_ids = identities[n_train + n_val:]

    print(f"  Train: {len(train_ids):>6}  ({len(train_ids)/n*100:.1f}%)")
    print(f"  Val:   {len(val_ids):>6}  ({len(val_ids)/n*100:.1f}%)")
    print(f"  Test:  {len(test_ids):>6}  ({len(test_ids)/n*100:.1f}%)")

    print(f"\n[3/4] Creating junctions at {output_dir}...")
    create_junctions_batch(output_dir, source_dir, train_ids, "train")
    create_junctions_batch(output_dir, source_dir, val_ids, "val")
    create_junctions_batch(output_dir, source_dir, test_ids, "test")

    print(f"\n[4/4] Summary:")
    print(f"  Output: {output_dir}/")
    print(f"  Train:  {len(train_ids)} identities")
    print(f"  Val:    {len(val_ids)} identities")
    print(f"  Test:   {len(test_ids)} identities")
    print("  Done!")


if __name__ == "__main__":
    main()
