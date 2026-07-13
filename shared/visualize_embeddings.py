"""
Visualize Embeddings — t-SNE/PCA visualization cho paper figures.

Usage:
    python visualize_embeddings.py \
        --test_dir dataset_final/test \
        --weights results/best_iresnet18_backbone.h5 \
        --variant iresnet18 \
        --max_ids 20 \
        --max_images_per_id 15
"""

import argparse
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


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


def main():
    ap = argparse.ArgumentParser(description="t-SNE / PCA Embedding Visualization")
    ap.add_argument("--test_dir", default="dataset_final/test")
    ap.add_argument("--weights", default="results/best_iresnet18_backbone.h5")
    ap.add_argument("--variant", type=str, default="iresnet18",
                    choices=["iresnet18", "iresnet34", "iresnet50", "hybridface", "mobilefacenet"])
    ap.add_argument("--img_size", type=int, default=112)
    ap.add_argument("--embedding_dim", type=int, default=512)
    ap.add_argument("--max_ids", type=int, default=20,
                    help="Number of identities to visualize")
    ap.add_argument("--max_images_per_id", type=int, default=15)
    ap.add_argument("--method", type=str, default="tsne",
                    choices=["tsne", "pca", "both"])
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", type=str, default="results")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Import heavy dependencies
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    from models.face_recognition.iresnet.iresnet50 import iResNet_Backbone

    test_dir = Path(args.test_dir)
    if not test_dir.is_dir():
        raise SystemExit(f"test_dir not found: {test_dir}")

    # Collect images
    id_to_imgs = defaultdict(list)
    for cls_dir in sorted(test_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        imgs = [p for p in cls_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        if len(imgs) >= 2:
            id_to_imgs[cls_dir.name] = imgs

    identities = sorted(id_to_imgs.keys())
    if args.max_ids and args.max_ids < len(identities):
        identities = random.sample(identities, args.max_ids)
        identities.sort()

    print(f"Visualizing {len(identities)} identities")

    # Sample images
    all_images = []
    all_labels = []
    label_names = []

    for idx, identity in enumerate(identities):
        imgs = id_to_imgs[identity]
        if len(imgs) > args.max_images_per_id:
            imgs = random.sample(imgs, args.max_images_per_id)
        for img_path in imgs:
            all_images.append(load_and_preprocess(img_path, args.img_size))
            all_labels.append(idx)
        label_names.append(identity)

    all_images = np.stack(all_images, axis=0)
    all_labels = np.array(all_labels)

    print(f"Total images: {len(all_images)}")

    # Build backbone and extract embeddings
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
            normalize_embeddings=True,
            variant=args.variant,
        )
    backbone.load_weights(str(Path(args.weights)))

    print("Extracting embeddings...")
    embeddings = backbone.predict(all_images, batch_size=args.batch_size, verbose=1)

    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-12)

    # Generate color palette
    n_classes = len(identities)
    cmap = plt.cm.get_cmap('tab20' if n_classes <= 20 else 'hsv', n_classes)
    colors = [cmap(i) for i in range(n_classes)]

    def plot_embedding(coords_2d, title, save_path):
        fig, ax = plt.subplots(figsize=(12, 10))

        for idx, identity in enumerate(identities):
            mask = all_labels == idx
            ax.scatter(
                coords_2d[mask, 0], coords_2d[mask, 1],
                c=[colors[idx]], s=30, alpha=0.7,
                label=identity, edgecolors='white', linewidths=0.3
            )

        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Dimension 1', fontsize=12)
        ax.set_ylabel('Dimension 2', fontsize=12)

        # Legend outside plot
        if n_classes <= 30:
            ax.legend(
                bbox_to_anchor=(1.05, 1), loc='upper left',
                fontsize=8, markerscale=2, ncol=1 if n_classes <= 20 else 2,
                frameon=True, fancybox=True, shadow=True,
            )

        ax.grid(True, ls='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[Plot] Saved: {save_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.method in ("tsne", "both"):
        print("\nRunning t-SNE...")
        tsne = TSNE(n_components=2, perplexity=min(30, len(embeddings) - 1),
                     random_state=args.seed, max_iter=1000, learning_rate='auto',
                     init='pca')
        coords = tsne.fit_transform(embeddings)
        plot_embedding(
            coords,
            f"t-SNE Embedding Space ({args.variant.upper()} + ArcFace)",
            str(output_dir / "tsne_embeddings.png")
        )

    if args.method in ("pca", "both"):
        print("\nRunning PCA...")
        pca = PCA(n_components=2, random_state=args.seed)
        coords = pca.fit_transform(embeddings)
        explained = pca.explained_variance_ratio_
        plot_embedding(
            coords,
            f"PCA Embedding Space ({args.variant.upper()} + ArcFace)\n"
            f"Explained variance: {explained[0]:.1%} + {explained[1]:.1%}",
            str(output_dir / "pca_embeddings.png")
        )

    # Also generate cosine similarity heatmap between class centroids
    print("\nComputing class centroids...")
    centroids = []
    for idx in range(n_classes):
        mask = all_labels == idx
        centroid = np.mean(embeddings[mask], axis=0)
        centroid = centroid / np.maximum(np.linalg.norm(centroid), 1e-12)
        centroids.append(centroid)
    centroids = np.stack(centroids)

    sim_matrix = centroids @ centroids.T

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(sim_matrix, cmap='RdYlBu_r', vmin=-0.3, vmax=1.0)
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(identities, rotation=90, fontsize=6)
    ax.set_yticklabels(identities, fontsize=6)
    ax.set_title(f'Inter-Class Cosine Similarity ({args.variant.upper()})',
                 fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Cosine Similarity')
    plt.tight_layout()
    plt.savefig(str(output_dir / "similarity_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved: {output_dir / 'similarity_heatmap.png'}")

    # Print intra-class vs inter-class similarity
    intra_sims = []
    inter_sims = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i == j:
                intra_sims.append(sim_matrix[i, j])
            else:
                inter_sims.append(sim_matrix[i, j])

    print(f"\nClass Centroid Statistics:")
    print(f"  Intra-class sim: {np.mean(intra_sims):.4f} (should be ~1.0)")
    print(f"  Inter-class sim: {np.mean(inter_sims):.4f} +/- {np.std(inter_sims):.4f} (should be low)")
    print(f"  Gap:             {np.mean(intra_sims) - np.mean(inter_sims):.4f}")


if __name__ == "__main__":
    main()
