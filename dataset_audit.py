import argparse
import csv
import os
from collections import Counter
from pathlib import Path

from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_images(root: Path):
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def list_class_dirs(root: Path):
    return sorted([p for p in root.iterdir() if p.is_dir()])


def count_images_per_class(root: Path):
    counts = Counter()
    for cls_dir in list_class_dirs(root):
        counts[cls_dir.name] = sum(1 for p in iter_images(cls_dir))
    return counts


def percentile(sorted_vals, p: float):
    if not sorted_vals:
        return None
    idx = int(p * (len(sorted_vals) - 1))
    return sorted_vals[idx]


def summarize_counts(name: str, counts: Counter):
    vals = sorted(counts.values())
    total = sum(vals)
    print(f"{name}: classes={len(vals)} images={total}")
    if not vals:
        return
    print(
        f"  per-class: min={vals[0]} p10={percentile(vals,0.1)} "
        f"median={percentile(vals,0.5)} p90={percentile(vals,0.9)} max={vals[-1]}"
    )


def main():
    ap = argparse.ArgumentParser(description="Quick audit for image-folder datasets.")
    ap.add_argument("--train_dir", default="dataset/train")
    ap.add_argument("--val_dir", default="dataset/val")
    ap.add_argument("--sample", type=int, default=0, help="If >0, sample N images per split for size/mode stats.")
    ap.add_argument("--min_dim", type=int, default=0, help="Report images with min(width,height) < min_dim.")
    ap.add_argument("--out_csv", default="", help="Write findings (tiny/corrupt) to CSV.")
    args = ap.parse_args()

    train_dir = Path(args.train_dir)
    val_dir = Path(args.val_dir)

    train_ids = {p.name for p in list_class_dirs(train_dir)} if train_dir.is_dir() else set()
    val_ids = {p.name for p in list_class_dirs(val_dir)} if val_dir.is_dir() else set()
    print(f"train_ids={len(train_ids)} val_ids={len(val_ids)} overlap={len(train_ids & val_ids)}")

    if train_dir.is_dir():
        summarize_counts("train", count_images_per_class(train_dir))
    if val_dir.is_dir():
        summarize_counts("val", count_images_per_class(val_dir))

    findings = []

    def scan_split(split_name: str, root: Path):
        if not root.is_dir():
            return
        files = list(iter_images(root))
        if not files:
            return

        if args.sample and args.sample < len(files):
            import random

            files = random.sample(files, args.sample)

        modes = Counter()
        sizes = Counter()
        corrupt = 0
        tiny = 0

        for p in files:
            try:
                with Image.open(p) as im:
                    im.verify()
                with Image.open(p) as im:
                    w, h = im.size
                    modes[im.mode] += 1
                    sizes[(w, h)] += 1
                    if args.min_dim and min(w, h) < args.min_dim:
                        tiny += 1
                        findings.append((split_name, "tiny", str(p), w, h, ""))
            except Exception as e:
                corrupt += 1
                findings.append((split_name, "corrupt", str(p), "", "", f"{type(e).__name__}: {e}"))

        print(f"{split_name}: scanned={len(files)} corrupt={corrupt} tiny={tiny}")
        print(f"  modes: {modes.most_common(5)}")
        print(f"  top_sizes: {sizes.most_common(8)}")

    if args.sample or args.min_dim:
        scan_split("train", train_dir)
        scan_split("val", val_dir)

    if args.out_csv and findings:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["split", "kind", "path", "width", "height", "error"])
            w.writerows(findings)
        print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

