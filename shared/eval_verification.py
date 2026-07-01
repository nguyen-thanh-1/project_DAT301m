"""
Verification Evaluation — Đánh giá chất lượng one-shot face verification.

Đánh giá backbone bằng cách:
1. Extract embeddings từ unseen identities (test set)
2. Tạo positive/negative pairs
3. Tính cosine similarity
4. Report EER, TAR@FAR, AUC
5. Plot ROC curve

Usage:
    python eval_verification.py \
        --test_dir dataset_final/test \
        --weights results/best_iresnet18_backbone.h5 \
        --variant iresnet18
"""

import os
import argparse
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from models.face_recognition.iresnet.iresnet50 import iResNet_Backbone


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_and_preprocess(image_path: Path, img_size: int):
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        w, h = im.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        im = im.crop((left, top, left + side, top + side))
        im = im.resize((img_size, img_size), resample=Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32)
    arr = (arr - 127.5) / 128.0
    return arr


def l2_normalize(x: np.ndarray, axis=-1, eps=1e-12):
    denom = np.linalg.norm(x, axis=axis, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom


def cosine_scores(a: np.ndarray, b: np.ndarray):
    a = l2_normalize(a)
    b = l2_normalize(b)
    return np.sum(a * b, axis=1)


def compute_eer(y_true: np.ndarray, scores: np.ndarray):
    order = np.argsort(scores)[::-1]
    y = y_true[order].astype(np.int32)
    n_pos = int(np.sum(y))
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None, None

    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)

    tpr = tp / max(n_pos, 1)
    fpr = fp / max(n_neg, 1)
    fnr = 1.0 - tpr

    i = int(np.argmin(np.abs(fpr - fnr)))
    eer = float((fpr[i] + fnr[i]) / 2.0)
    thr = float(scores[order][i])
    return eer, thr


def tpr_at_far(y_true: np.ndarray, scores: np.ndarray, far: float):
    order = np.argsort(scores)[::-1]
    y = y_true[order].astype(np.int32)
    n_pos = int(np.sum(y))
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None, None

    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    fpr = fp / max(n_neg, 1)
    tpr = tp / max(n_pos, 1)

    idx = np.where(fpr <= far)[0]
    if len(idx) == 0:
        return 0.0, float(scores[order][0])
    i = int(idx[-1])
    return float(tpr[i]), float(scores[order][i])


def compute_roc(y_true: np.ndarray, scores: np.ndarray, n_points=200):
    """Compute ROC curve points."""
    thresholds = np.linspace(scores.min(), scores.max(), n_points)
    fprs, tprs = [], []
    n_pos = int(np.sum(y_true))
    n_neg = int(len(y_true) - n_pos)

    for t in thresholds:
        predicted_pos = scores >= t
        tp = np.sum(predicted_pos & (y_true == 1))
        fp = np.sum(predicted_pos & (y_true == 0))
        tprs.append(tp / max(n_pos, 1))
        fprs.append(fp / max(n_neg, 1))

    return np.array(fprs), np.array(tprs), thresholds


def compute_auc(fprs, tprs):
    """Compute AUC using trapezoidal rule."""
    order = np.argsort(fprs)
    fprs = fprs[order]
    tprs = tprs[order]
    return float(np.trapz(tprs, fprs))


def plot_roc(fprs, tprs, eer, auc, save_path):
    """Plot ROC curve with EER point."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    ax.plot(fprs, tprs, 'b-', lw=2.5, label=f'ROC (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random')

    # Mark EER point
    if eer is not None:
        ax.plot(eer, 1 - eer, 'ro', markersize=10, label=f'EER = {eer:.4f}')
        ax.axhline(y=1-eer, color='r', ls='--', alpha=0.3)
        ax.axvline(x=eer, color='r', ls='--', alpha=0.3)

    ax.set_xlabel('False Positive Rate (FPR)', fontsize=13)
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=13)
    ax.set_title('ROC Curve - Face Verification', fontsize=15, fontweight='bold')
    ax.legend(loc='lower right', fontsize=12)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, ls='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Plot] ROC curve saved to: {save_path}")


def plot_score_distribution(pos_scores, neg_scores, save_path):
    """Plot genuine vs impostor score distribution."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.hist(neg_scores, bins=80, alpha=0.6, color='#e74c3c', label='Impostor (Different)', density=True)
    ax.hist(pos_scores, bins=80, alpha=0.6, color='#2ecc71', label='Genuine (Same)', density=True)

    ax.set_xlabel('Cosine Similarity', fontsize=13)
    ax.set_ylabel('Density', fontsize=13)
    ax.set_title('Score Distribution: Genuine vs Impostor', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, ls='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Plot] Score distribution saved to: {save_path}")


def iter_images(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def main():
    ap = argparse.ArgumentParser(
        description="Verification evaluation (one-shot) with cosine similarity."
    )
    ap.add_argument("--test_dir", default="dataset_final/test",
                    help="Folder: test/<id>/*.jpg")
    ap.add_argument("--weights", default="results/best_iresnet50_backbone.h5")
    ap.add_argument("--img_size", type=int, default=112)
    ap.add_argument("--embedding_dim", type=int, default=512)
    ap.add_argument("--variant", type=str, default="iresnet50",
                    choices=["iresnet50", "hybridface", "mobilefacenet"],
                    help="iResNet50/Hybrid/MobileFaceNet variant")
    ap.add_argument("--pairs", type=int, default=10000,
                    help="Number of positive and negative pairs each.")
    ap.add_argument("--max_ids", type=int, default=0,
                    help="If >0, limit number of identities used.")
    ap.add_argument("--max_images_per_id", type=int, default=30,
                    help="Limit images sampled per identity.")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", type=str, default="results",
                    help="Directory to save plots")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    test_dir = Path(args.test_dir)
    if not test_dir.is_dir():
        raise SystemExit(f"test_dir not found: {test_dir}")

    id_to_imgs = defaultdict(list)
    for img in iter_images(test_dir):
        if not img.parent.is_dir():
            continue
        identity = img.parent.name
        id_to_imgs[identity].append(img)

    identities = [k for k, v in id_to_imgs.items() if len(v) >= 2]
    identities.sort()
    if args.max_ids and args.max_ids < len(identities):
        identities = random.sample(identities, args.max_ids)

    if len(identities) < 2:
        raise SystemExit("Need at least 2 identities with >=2 images each.")

    # Sample image pool
    sampled_imgs = []
    for identity in identities:
        imgs = id_to_imgs[identity]
        if len(imgs) > args.max_images_per_id:
            imgs = random.sample(imgs, args.max_images_per_id)
        id_to_imgs[identity] = imgs
        sampled_imgs.extend(imgs)

    # Build iResNet backbone
    print(f"\n{'='*60}")
    print(f"Face Verification Evaluation")
    print(f"  Variant:    {args.variant}")
    print(f"  Weights:    {args.weights}")
    print(f"  Test Dir:   {args.test_dir}")
    print(f"  Identities: {len(identities)}")
    print(f"  Images:     {len(sampled_imgs)}")
    print(f"  Pairs:      {args.pairs} pos + {args.pairs} neg")
    print(f"{'='*60}\n")

    if args.variant == "hybridface":
        from models.face_recognition.hybridface.hybrid_face import HybridFace_Backbone
        backbone = HybridFace_Backbone(
            input_shape=(args.img_size, args.img_size, 3),
            embedding_dim=args.embedding_dim,
            normalize_embeddings=True,
        )
    elif args.variant == "mobilefacenet":
        from models.face_recognition.mobilefacenet.mobile_face_net import MobileFaceNet_Backbone
        backbone = MobileFaceNet_Backbone(
            input_shape=(args.img_size, args.img_size, 3),
            embedding_dim=args.embedding_dim,
            normalize_embeddings=True,
        )
    else:
        from models.face_recognition.iresnet.iresnet50 import iResNet_Backbone
        backbone = iResNet_Backbone(
            input_shape=(args.img_size, args.img_size, 3),
            embedding_dim=args.embedding_dim,
            dropout_rate=0.4,
            normalize_embeddings=True,
        )
    backbone.load_weights(str(Path(args.weights)))

    # Extract embeddings (batched)
    embeddings = {}
    batch = []
    batch_paths = []

    def flush():
        if not batch:
            return
        x = np.stack(batch, axis=0)
        out = backbone.predict(x, batch_size=args.batch_size, verbose=0)
        for p, e in zip(batch_paths, out):
            embeddings[p] = e.astype(np.float32, copy=False)
        batch.clear()
        batch_paths.clear()

    print("[1/3] Extracting embeddings...")
    for p in sampled_imgs:
        batch.append(load_and_preprocess(p, args.img_size))
        batch_paths.append(p)
        if len(batch) >= args.batch_size:
            flush()
    flush()

    # Build pairs
    print("[2/3] Building pairs...")
    pos_a, pos_b = [], []
    neg_a, neg_b = [], []

    for _ in range(args.pairs):
        identity = random.choice(identities)
        a, b = random.sample(id_to_imgs[identity], 2)
        pos_a.append(a)
        pos_b.append(b)

    for _ in range(args.pairs):
        id1, id2 = random.sample(identities, 2)
        a = random.choice(id_to_imgs[id1])
        b = random.choice(id_to_imgs[id2])
        neg_a.append(a)
        neg_b.append(b)

    a_emb = np.stack([embeddings[p] for p in pos_a + neg_a], axis=0)
    b_emb = np.stack([embeddings[p] for p in pos_b + neg_b], axis=0)
    scores = cosine_scores(a_emb, b_emb).astype(np.float32)
    y_true = np.array([1] * args.pairs + [0] * args.pairs, dtype=np.int32)

    pos_scores = scores[:args.pairs]
    neg_scores = scores[args.pairs:]

    # Compute metrics
    print("[3/3] Computing metrics...\n")
    eer, thr = compute_eer(y_true, scores)
    tpr_1e2, thr_1e2 = tpr_at_far(y_true, scores, far=1e-2)
    tpr_1e3, thr_1e3 = tpr_at_far(y_true, scores, far=1e-3)
    tpr_1e4, thr_1e4 = tpr_at_far(y_true, scores, far=1e-4)

    fprs, tprs, _ = compute_roc(y_true, scores)
    auc = compute_auc(fprs, tprs)

    # Print results
    print("=" * 60)
    print(f"Verification Results ({len(identities)} unseen identities)")
    print("=" * 60)
    print(f"  Identities:     {len(identities)}")
    print(f"  Images used:    {len(sampled_imgs)}")
    print(f"  Pairs:          {args.pairs} pos + {args.pairs} neg")
    print("-" * 60)
    if eer is not None:
        print(f"  EER:            {eer:.4f} ({eer*100:.2f}%)")
        print(f"  EER Threshold:  {thr:.4f}")
    print(f"  AUC:            {auc:.4f}")
    if tpr_1e2 is not None:
        print(f"  TAR @ FAR=1e-2: {tpr_1e2:.4f} ({tpr_1e2*100:.2f}%)")
    if tpr_1e3 is not None:
        print(f"  TAR @ FAR=1e-3: {tpr_1e3:.4f} ({tpr_1e3*100:.2f}%)")
    if tpr_1e4 is not None:
        print(f"  TAR @ FAR=1e-4: {tpr_1e4:.4f} ({tpr_1e4*100:.2f}%)")
    print("-" * 60)
    print(f"  Pos sim (mean): {np.mean(pos_scores):.4f} +/- {np.std(pos_scores):.4f}")
    print(f"  Neg sim (mean): {np.mean(neg_scores):.4f} +/- {np.std(neg_scores):.4f}")
    print("=" * 60)

    # Save plots
    os.makedirs(args.output_dir, exist_ok=True)
    plot_roc(fprs, tprs, eer, auc,
             os.path.join(args.output_dir, "roc_curve.png"))
    plot_score_distribution(pos_scores, neg_scores,
                            os.path.join(args.output_dir, "score_distribution.png"))


if __name__ == "__main__":
    main()
