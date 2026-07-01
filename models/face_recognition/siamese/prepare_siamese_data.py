"""
Prepare Siamese dataset — re-split dataset_aligned by identity.

Input:
    dataset_aligned/ — 540 identity folders, 196K aligned 112×112 images

Output (via symlinks, zero disk usage):
    dataset_siamese/train/ — 480 identities (all images)
    dataset_siamese/val/   — 60 identities (all images)

Train and val are fully disjoint by identity — ideal for Siamese verification.
"""

import argparse
import os
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="dataset_aligned")
    parser.add_argument("--output", default="dataset_siamese")
    parser.add_argument("--val_count", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    random.seed(args.seed)

    identities = sorted([d.name for d in src.iterdir() if d.is_dir()])
    random.shuffle(identities)

    train_ids = identities[args.val_count:]
    val_ids = identities[:args.val_count]

    print(f"Total identities: {len(identities)}")
    print(f"  Train: {len(train_ids)} identities")
    print(f"  Val:   {len(val_ids)} identities (DISJOINT)")
    print(f"  Seed:  {args.seed}")

    for split, id_list in [("train", train_ids), ("val", val_ids)]:
        split_dir = dst / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for identity in id_list:
            link = split_dir / identity
            target = src / identity
            if not link.exists():
                try:
                    link.symlink_to(target.resolve(), target_is_directory=True)
                    count = sum(1 for _ in target.iterdir() if _.is_file())
                except OSError:
                    os.system(f'mklink /J "{link}" "{target.resolve()}" 2>nul')
                    count = sum(1 for _ in target.iterdir() if _.is_file())
        total = sum(1 for _ in split_dir.iterdir())
        print(f"  {split}/ : {total} identities created")

    print(f"\nDone. Output: {dst.resolve()}")


if __name__ == "__main__":
    main()
