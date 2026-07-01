"""
HybridFaceNet verification evaluation on unseen identities.

Outputs metrics for CNN-only, Transformer-only and Fused embeddings:
EER, TAR@FAR=0.001, TAR@FAR=0.01, AUC-ROC, ROC curves, score distributions,
and optional t-SNE for fused embeddings.
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from models.face_recognition.hybridfacenet.hybridfacenet import (
    build_ablation_models,
    build_training_model,
)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def scan_test_dir(test_dir):
    id_to_imgs = defaultdict(list)
    for class_dir in Path(test_dir).iterdir():
        if not class_dir.is_dir():
            continue
        imgs = [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        if len(imgs) >= 2:
            id_to_imgs[class_dir.name] = imgs
    return dict(id_to_imgs)


def make_pairs(id_to_imgs, num_pairs=50000, seed=42):
    rng = random.Random(seed)
    identities = sorted(id_to_imgs)
    pos, neg = [], []
    for _ in range(num_pairs // 2):
        identity = rng.choice(identities)
        pos.append(tuple(rng.sample(id_to_imgs[identity], 2)))
        a_id, b_id = rng.sample(identities, 2)
        neg.append((rng.choice(id_to_imgs[a_id]), rng.choice(id_to_imgs[b_id])))
    pairs = pos + neg
    labels = np.array([1] * len(pos) + [0] * len(neg), dtype=np.int32)
    return pairs, labels


def load_image(path):
    img = tf.io.read_file(str(path))
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [112, 112])
    img.set_shape([112, 112, 3])
    return tf.cast(img, tf.float32)


def embed_paths(model, paths, batch_size):
    images = np.stack([load_image(p).numpy() for p in paths])
    embeddings = model.predict(images, batch_size=batch_size, verbose=0)
    return embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)


def cosine_scores(model, pairs, batch_size):
    all_paths = sorted(set([p for pair in pairs for p in pair]), key=lambda p: str(p))
    embeddings = embed_paths(model, all_paths, batch_size)
    path_to_idx = {p: idx for idx, p in enumerate(all_paths)}
    scores = []
    for a, b in pairs:
        scores.append(float(np.sum(embeddings[path_to_idx[a]] * embeddings[path_to_idx[b]])))
    return np.array(scores, dtype=np.float32), all_paths, embeddings


def roc_metrics(labels, scores):
    order = np.argsort(scores)[::-1]
    y = labels[order]
    thresholds = scores[order]
    pos = max(int(np.sum(y == 1)), 1)
    neg = max(int(np.sum(y == 0)), 1)
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    tpr = tp / pos
    fpr = fp / neg
    fnr = 1.0 - tpr
    eer_idx = int(np.argmin(np.abs(fpr - fnr)))
    auc = float(np.trapz(tpr, fpr))

    def tar_at_far(target_far):
        valid = np.where(fpr <= target_far)[0]
        if len(valid) == 0:
            return 0.0
        return float(np.max(tpr[valid]))

    return {
        "eer": float((fpr[eer_idx] + fnr[eer_idx]) / 2.0),
        "auc": auc,
        "tar_far_0.001": tar_at_far(0.001),
        "tar_far_0.01": tar_at_far(0.01),
        "optimal_threshold": float(thresholds[eer_idx]),
        "fpr": fpr,
        "tpr": tpr,
    }


def plot_roc(all_metrics, save_path):
    plt.figure(figsize=(7, 6))
    for name, metrics in all_metrics.items():
        plt.plot(metrics["fpr"], metrics["tpr"], label=f"{name} AUC={metrics['auc']:.4f}")
    plt.xscale("log")
    plt.xlabel("FAR / FPR")
    plt.ylabel("TAR / TPR")
    plt.title("HybridFaceNet Verification ROC")
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_scores(labels, scores, save_path, title):
    plt.figure(figsize=(7, 5))
    plt.hist(scores[labels == 1], bins=80, alpha=0.6, label="positive")
    plt.hist(scores[labels == 0], bins=80, alpha=0.6, label="negative")
    plt.xlabel("Cosine similarity")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_tsne(paths, embeddings, save_path, max_samples=3000):
    try:
        from sklearn.manifold import TSNE
    except Exception as exc:
        print(f"[t-SNE] skipped: sklearn unavailable ({exc})")
        return

    n = min(len(paths), max_samples)
    idx = np.random.default_rng(42).choice(len(paths), size=n, replace=False)
    sample_embeddings = embeddings[idx]
    labels = [Path(paths[i]).parent.name for i in idx]
    unique = {label: j for j, label in enumerate(sorted(set(labels)))}
    colors = [unique[label] for label in labels]

    coords = TSNE(n_components=2, perplexity=30, n_iter=1000, init="pca", learning_rate="auto").fit_transform(
        sample_embeddings
    )
    plt.figure(figsize=(8, 7))
    plt.scatter(coords[:, 0], coords[:, 1], c=colors, s=5, cmap="tab20", alpha=0.8)
    plt.title("HybridFaceNet fused embedding t-SNE")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate HybridFaceNet verification metrics")
    parser.add_argument("--test_dir", type=str, default="dataset_final/test")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--weights_type", choices=["backbone", "training"], default="backbone")
    parser.add_argument("--num_classes", type=int, default=432)
    parser.add_argument("--num_pairs", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--save_dir", type=str, default=str(Path(__file__).resolve().parent / "results" / "eval"))
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    training_model, backbone, _ = build_training_model(num_classes=args.num_classes)
    if args.weights_type == "training":
        training_model.load_weights(args.weights)
    else:
        backbone.load_weights(args.weights)

    models = build_ablation_models(backbone)
    id_to_imgs = scan_test_dir(args.test_dir)
    if len(id_to_imgs) < 2:
        raise SystemExit("test_dir must contain at least two identities with >=2 images each")
    pairs, labels = make_pairs(id_to_imgs, args.num_pairs)

    metrics_out = {}
    all_metrics = {}
    fused_paths, fused_embeddings = None, None
    for name, model in models.items():
        print(f"[Eval] {name}")
        scores, paths, embeddings = cosine_scores(model, pairs, args.batch_size)
        metrics = roc_metrics(labels, scores)
        all_metrics[name] = metrics
        metrics_out[name] = {k: v for k, v in metrics.items() if k not in {"fpr", "tpr"}}
        plot_scores(labels, scores, save_dir / f"score_distribution_{name}.png", f"{name} score distribution")
        if name == "fused":
            fused_paths, fused_embeddings = paths, embeddings

    plot_roc(all_metrics, save_dir / "roc_curve_ablation.png")
    if fused_paths is not None:
        plot_tsne(fused_paths, fused_embeddings, save_dir / "tsne_fused_embeddings.png")

    with open(save_dir / "verification_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)
    print(json.dumps(metrics_out, indent=2))


if __name__ == "__main__":
    main()

