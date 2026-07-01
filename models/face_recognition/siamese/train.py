"""
Siamese / Triplet iResNet50 — One-Shot Face Verification Training.

Features:
  • [Siamese] Balanced pos/neg pair generation with progressive hard mining
  • [Triplet]  PK-sampling (P identities × K images) with batch-hard triplet loss (FaceNet-style)
  • Mixed precision (float16) for 16GB VRAM
  • Dual loss: BCE (comparison head) + Contrastive (embeddings, cosine-based) [siamese only]
  • Progressive contrastive margin (0.4 → 0.0 cosine schedule) [siamese only]
  • Progressive hard negative mining ratio (10% → 50%) [siamese only]
  • Full-state checkpoint for resume (model + optimizer + epoch)
  • Per-epoch training history chart (overwrite)
  • Verification evaluation (EER, ROC, score dist, t-SNE) each N epochs
  • All outputs saved to results/

Usage:
    # Siamese (BCE + contrastive, default)
    python models/face_recognition/siamese/train.py ^
        --dataset_dir dataset_siamese/train ^
        --val_dir dataset_siamese/val ^
        --test_dir dataset_siamese/val ^
        --batch_size 100 ^
        --epochs 80

    # Triplet (batch-hard triplet loss, FaceNet-style)
    python models/face_recognition/siamese/train.py ^
        --dataset_dir dataset_siamese/train ^
        --val_dir dataset_siamese/val ^
        --test_dir dataset_siamese/val ^
        --loss_type triplet ^
        --p_per_batch 40 ^
        --k_per_id 5 ^
        --triplet_margin 0.3 ^
        --epochs 80
"""

import argparse, json, math, os, random, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.append(str(Path(__file__).resolve().parents[3]))

from models.face_recognition.siamese.model import (
    build_siamese_model, build_backbone, make_contrastive_loss,
    get_margin_variable, ProgressiveMarginCallback,
    build_triplet_model, batch_hard_triplet_loss, triplet_active_fraction,
)

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ──────────────────────────────────────────────────────────────
# Mixed Precision (half VRAM, 1.5x faster on Ampere+ GPUs)
# ──────────────────────────────────────────────────────────────
tf.keras.mixed_precision.set_global_policy("mixed_float16")

# ──────────────────────────────────────────────────────────────
# GPU Memory Growth
# ──────────────────────────────────────────────────────────────
def configure_gpu():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        # Clear any stale session state (Jupyter re-runs)
        tf.keras.backend.clear_session()
        print(f"[GPU] {len(gpus)} GPU(s) detected. Memory growth ON. Session cleared.")
    else:
        print("[GPU] No GPU found. Running on CPU.")


def _verify_ckpt_integrity(checkpoint_path):
    """Verify checkpoint index and data files exist before restore."""
    index_file = checkpoint_path + ".index"
    data_file = checkpoint_path + ".data-00000-of-00001"
    missing = []
    if not os.path.exists(index_file):
        missing.append(index_file)
    if not os.path.exists(data_file):
        missing.append(data_file)
    if missing:
        raise FileNotFoundError(
            f"Checkpoint file(s) missing: {missing}\n"
            f"Cannot resume from {checkpoint_path} — retrain from scratch."
        )
    print(f"[Checkpoint] Integrity verified: {checkpoint_path}")


# ──────────────────────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────────────────────
def get_identity_images(dataset_dir):
    """Scan {identity_name}/image.jpg structure. Require >= 2 images per identity."""
    dataset_dir = Path(dataset_dir)
    id_imgs = defaultdict(list)
    for cls_dir in sorted(dataset_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        imgs = [p for p in cls_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        if len(imgs) >= 2:
            id_imgs[cls_dir.name] = imgs
    return dict(id_imgs)


# ──────────────────────────────────────────────────────────────
# Pair Generation (balanced pos/neg, optional hard-neg bias)
# ──────────────────────────────────────────────────────────────
def generate_pairs(identity_images, num_pairs, rng, hard_pairs=None, hard_ratio=0.5):
    """Generate balanced positive/negative pairs.

    If hard_pairs provided, hard_ratio fraction of negatives drawn from hard pairs.
    """
    identities = sorted(identity_images.keys())
    valid_ids = [i for i in identities if len(identity_images[i]) >= 2]
    if len(valid_ids) < 2:
        raise ValueError("Need >= 2 identities with >= 2 images each")

    num_pos = num_pairs // 2
    num_neg = num_pairs - num_pos

    pos_pairs = []
    for _ in range(num_pos):
        identity = rng.choice(valid_ids)
        a, b = rng.sample(list(identity_images[identity]), 2)
        pos_pairs.append((str(a), str(b), 1))

    neg_pairs = []
    for _ in range(num_neg):
        if hard_pairs and rng.random() < hard_ratio:
            id1, id2 = rng.choice(hard_pairs)
            if id1 not in identity_images or id2 not in identity_images:
                id1, id2 = rng.sample(identities, 2)
        else:
            id1, id2 = rng.sample(identities, 2)
        a = rng.choice(identity_images[id1])
        b = rng.choice(identity_images[id2])
        neg_pairs.append((str(a), str(b), 0))

    combined = pos_pairs + neg_pairs
    rng.shuffle(combined)
    paths_a, paths_b, labels = zip(*combined)
    return list(paths_a), list(paths_b), list(labels)


# ──────────────────────────────────────────────────────────────
# PK-Sampling for Triplet Loss (FaceNet-style)
# ──────────────────────────────────────────────────────────────
def generate_pk_batch(identity_images, p_per_batch, k_per_id, rng):
    """Generate one PK batch: P identities × K images each.

    Returns (image_paths, labels) where labels are 0..P-1 identity indices.
    Each identity must have >= k_per_id images, or it is skipped.
    """
    identities = sorted(identity_images.keys())
    valid_ids = [i for i in identities if len(identity_images[i]) >= k_per_id]
    if len(valid_ids) < p_per_batch:
        raise ValueError(
            f"Need >= {p_per_batch} identities with >= {k_per_id} images each; "
            f"only {len(valid_ids)} available."
        )

    selected_ids = rng.sample(valid_ids, p_per_batch)
    image_paths, labels = [], []
    for idx, identity in enumerate(selected_ids):
        imgs = rng.sample(list(identity_images[identity]), k_per_id)
        image_paths.extend(str(p) for p in imgs)
        labels.extend([idx] * k_per_id)

    # Shuffle within the batch so the model can't rely on positional order
    combined = list(zip(image_paths, labels))
    rng.shuffle(combined)
    image_paths, labels = zip(*combined)
    return list(image_paths), list(labels)


# ──────────────────────────────────────────────────────────────
# Image Loading & Augmentation
# ──────────────────────────────────────────────────────────────
def load_image_tf(path, img_size=112):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [img_size, img_size])
    img.set_shape([img_size, img_size, 3])
    img = (tf.cast(img, tf.float32) - 127.5) / 128.0
    return img


# Alias — train.py; the dataset is pre-aligned, so no MTCNN needed at runtime.
# All dataset_siamese images should be pre-aligned via data_processing/preprocess.py


def augment_image(image):
    """Per-image augmentation (applied independently to each image in a batch).
    
    Simulates real-world variation: pose, lighting, blur, occlusion.
    """
    # Horizontal flip (50%)
    image = tf.image.random_flip_left_right(image)
    # Brightness jitter (±15%)
    image = tf.image.random_brightness(image, max_delta=0.15)
    # Contrast jitter (±15%)
    image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
    # Saturation jitter (±20%) — simulates phone camera color shifts
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    # Hue jitter (±3%) — subtle lighting temperature shifts
    image = tf.image.random_hue(image, max_delta=0.03)
    # Random blur (15% chance, downscale→upscale) — motion/defocus
    if tf.random.uniform([]) < 0.15:
        orig_shape = tf.shape(image)
        scale = tf.random.uniform([], minval=2, maxval=5, dtype=tf.int32)
        small_h = tf.maximum(orig_shape[0] // scale, 16)
        small_w = tf.maximum(orig_shape[1] // scale, 16)
        image = tf.image.resize(tf.expand_dims(image, 0), [small_h, small_w])
        image = tf.image.resize(image, [orig_shape[0], orig_shape[1]])
        image = tf.squeeze(image, 0)
    # Random erasing / cutout (25% chance) — occlusion robustness
    if tf.random.uniform([]) < 0.25:
        img_h = tf.shape(image)[0]
        img_w = tf.shape(image)[1]
        L = tf.random.uniform([], minval=20, maxval=45, dtype=tf.int32)
        x = tf.random.uniform([], minval=0, maxval=img_h - L, dtype=tf.int32)
        y = tf.random.uniform([], minval=0, maxval=img_w - L, dtype=tf.int32)
        row_mask = tf.logical_and(
            tf.range(img_h)[:, tf.newaxis] >= x,
            tf.range(img_h)[:, tf.newaxis] < x + L)
        col_mask = tf.logical_and(
            tf.range(img_w)[tf.newaxis, :] >= y,
            tf.range(img_w)[tf.newaxis, :] < y + L)
        mask = tf.cast(tf.logical_and(row_mask, col_mask), tf.float32)[:, :, tf.newaxis]
        # Fill erased region with random noise (more realistic than black)
        noise = tf.random.uniform(tf.shape(image), minval=-1.0, maxval=1.0)
        image = image * (1.0 - mask) + noise * mask
    image = tf.clip_by_value(image, -1.0, 1.0)
    return image


# ──────────────────────────────────────────────────────────────
# Progressive Hard Mining Ratio (global, updated by callback)
# ──────────────────────────────────────────────────────────────
hard_pairs_store = []
_hard_ratio = 0.1


def get_hard_ratio():
    return _hard_ratio


def set_hard_ratio(val):
    global _hard_ratio
    _hard_ratio = val


# ──────────────────────────────────────────────────────────────
# tf.data Pipeline
# ──────────────────────────────────────────────────────────────
def build_pair_dataset(identity_images, num_pairs, batch_size,
                       img_size=112, is_training=True, seed=42,
                       use_hard_negatives=False):
    """Build tf.data pipeline: regenerates pairs from scratch every epoch."""

    def pair_generator():
        rng = random.Random(seed)
        while True:
            hp = hard_pairs_store if use_hard_negatives else None
            hr = get_hard_ratio() if use_hard_negatives else 0.5
            paths_a, paths_b, labs = generate_pairs(identity_images, num_pairs, rng, hp, hr)
            for pa, pb, lbl in zip(paths_a, paths_b, labs):
                yield pa, pb, lbl

    def process_pair(path_a, path_b, label):
        img_a = load_image_tf(path_a, img_size)
        img_b = load_image_tf(path_b, img_size)
        if is_training:
            img_a = augment_image(img_a)
            img_b = augment_image(img_b)
        return (img_a, img_b), tf.cast(label, tf.float32)

    ds = tf.data.Dataset.from_generator(
        pair_generator,
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )
    ds = ds.map(process_pair, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ──────────────────────────────────────────────────────────────
# Triplet tf.data Pipeline (PK-sampling)
# ──────────────────────────────────────────────────────────────
def build_triplet_dataset(identity_images, p_per_batch, k_per_id,
                          img_size=112, is_training=True, seed=42):
    """Build tf.data pipeline for triplet PK-sampling.

    Each "batch" is really a set of P*K images with identity labels;
    the loss function does online triplet mining within the batch.
    """
    total_per_batch = p_per_batch * k_per_id

    def triplet_generator():
        rng = random.Random(seed)
        while True:
            paths, labels = generate_pk_batch(identity_images, p_per_batch, k_per_id, rng)
            for p, l in zip(paths, labels):
                yield p, l

    def process_item(path, label):
        img = load_image_tf(path, img_size)
        if is_training:
            img = augment_image(img)
        return img, label

    ds = tf.data.Dataset.from_generator(
        triplet_generator,
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )
    ds = ds.map(process_item, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(total_per_batch, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ──────────────────────────────────────────────────────────────
# Warmup + Cosine Decay LR Schedule
# ──────────────────────────────────────────────────────────────
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, peak_lr, warmup_steps, total_steps, min_lr=1e-6, start_step=0):
        super().__init__()
        self.peak_lr = float(peak_lr)
        self.warmup_steps = float(warmup_steps)
        self.total_steps = float(total_steps)
        self.min_lr = float(min_lr)
        self.start_step = float(start_step)

    def __call__(self, step):
        # NOTE: Do NOT add self.start_step here.
        # When resuming from a full checkpoint, optimizer.iterations is restored
        # to the correct step count automatically by tf.train.Checkpoint.restore().
        # Adding start_step would double-count the step, causing LR phase mismatch.
        step = tf.cast(step, tf.float32)
        warmup = self.peak_lr * (step / tf.maximum(self.warmup_steps, 1.0))
        progress = (step - self.warmup_steps) / tf.maximum(self.total_steps - self.warmup_steps, 1.0)
        cosine = self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (1.0 + tf.cos(3.14159265 * progress))
        return tf.where(step < self.warmup_steps, warmup, cosine)

    def get_config(self):
        return {"peak_lr": self.peak_lr, "warmup_steps": self.warmup_steps,
                "total_steps": self.total_steps, "min_lr": self.min_lr,
                "start_step": self.start_step}


# ──────────────────────────────────────────────────────────────
# Hard Negative Mining Callback
# ──────────────────────────────────────────────────────────────
class HardNegativeMiningCallback(tf.keras.callbacks.Callback):
    """Periodically mine hardest identity pairs + progressively increase ratio.

    Every mine_every epochs after start_epoch:
      1. Discover hard identity pairs via centroid similarity
      2. Update global hard_pairs_store
      3. Progressively increase hard_ratio from ratio_start → ratio_end
    """

    def __init__(self, backbone, identity_images, mine_every=5, start_epoch=30,
                 num_images=1500, top_k=100, img_size=112, batch_size=64, seed=42,
                 ratio_start=0.1, ratio_end=0.5):
        super().__init__()
        self.backbone = backbone
        self.identity_images = identity_images
        self.mine_every = mine_every
        self.start_epoch = start_epoch
        self.num_images = num_images
        self.top_k = top_k
        self.img_size = img_size
        self.batch_size = batch_size
        self.rng = random.Random(seed)
        self.ratio_start = ratio_start
        self.ratio_end = ratio_end
        self._ratio_updated = False

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) < self.start_epoch:
            return

        # Progressive hard ratio increase
        if not self._ratio_updated or (epoch + 1) % self.mine_every == 0:
            progress = min((epoch + 1 - self.start_epoch) / max(80 - self.start_epoch, 1), 1.0)
            new_ratio = self.ratio_start + progress * (self.ratio_end - self.ratio_start)
            set_hard_ratio(new_ratio)
            self._ratio_updated = True

        if (epoch + 1) % self.mine_every != 0:
            return

        print(f"\n{'─'*50}")
        print(f"[Hard Mining] Epoch {epoch+1}: discovering hard pairs (ratio={get_hard_ratio():.2f})...")
        print(f"{'─'*50}")

        identities = sorted(self.identity_images.keys())
        all_paths, all_labels = [], []
        for identity in identities:
            imgs = self.identity_images[identity]
            sample = self.rng.sample(imgs, min(len(imgs), 5))
            for p in sample:
                all_paths.append(str(p))
                all_labels.append(identity)

        if len(all_paths) > self.num_images:
            indices = self.rng.sample(range(len(all_paths)), self.num_images)
            all_paths = [all_paths[i] for i in indices]
            all_labels = [all_labels[i] for i in indices]

        all_images = []
        for p in all_paths:
            img = load_image_tf(p, self.img_size)
            all_images.append(tf.cast(img, tf.float16).numpy())
        all_images = np.stack(all_images, axis=0)

        embeddings = []
        for i in range(0, len(all_images), self.batch_size):
            batch = all_images[i:i + self.batch_size]
            emb = self.backbone.predict(batch, batch_size=self.batch_size, verbose=0)
            embeddings.append(emb)
        embeddings = np.concatenate(embeddings, axis=0).astype(np.float32)

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-12)

        label_to_embs = defaultdict(list)
        for emb, lbl in zip(embeddings, all_labels):
            label_to_embs[lbl].append(emb)

        centroids = {}
        for lbl, embs in label_to_embs.items():
            centroids[lbl] = np.mean(embs, axis=0)
        centroid_ids = sorted(centroids.keys())

        if len(centroid_ids) < 2:
            print("[Hard Mining] Not enough identities. Skipping.")
            return

        cent_mat = np.stack([centroids[c] for c in centroid_ids], axis=0)
        cent_mat = cent_mat / np.maximum(np.linalg.norm(cent_mat, axis=1, keepdims=True), 1e-12)
        sim_mat = np.dot(cent_mat, cent_mat.T)

        n = len(centroid_ids)
        pairs_with_sim = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs_with_sim.append((centroid_ids[i], centroid_ids[j], float(sim_mat[i, j])))
        pairs_with_sim.sort(key=lambda x: x[2], reverse=True)
        top_pairs = pairs_with_sim[:self.top_k]

        global hard_pairs_store
        hard_pairs_store = [(id1, id2) for id1, id2, _ in top_pairs]

        sims = [s for _, _, s in top_pairs[:8]]
        avg_sim = np.mean([s for _, _, s in top_pairs])
        print(f"[Hard Mining] Top-8 sims: {[f'{s:.3f}' for s in sims]}")
        print(f"[Hard Mining] Average sim of top-{self.top_k}: {avg_sim:.4f}\n")


# ──────────────────────────────────────────────────────────────
# History Plotting Callback (overwrites per epoch, resume-safe)
# ──────────────────────────────────────────────────────────────
class HistoryPlotterCallback(tf.keras.callbacks.Callback):
    """Auto-detects metrics — works with both siamese (accuracy) and triplet (active_fraction)."""
    def __init__(self, save_dir):
        super().__init__()
        self.save_dir = save_dir
        self.json_path = os.path.join(save_dir, "training_history.json")
        self.png_path = os.path.join(save_dir, "training_history.png")
        self.metrics = None  # built lazily from first epoch

        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, "r") as f:
                    self.metrics = json.load(f)
            except Exception:
                self.metrics = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # Auto-build metrics dict from first epoch
        if self.metrics is None:
            self.metrics = {"loss": [], "val_loss": [], "lr": []}
            # Grab any accuracy-like metric
            for k in sorted(logs.keys()):
                if k not in ("loss", "val_loss", "lr") and isinstance(logs[k], (int, float)):
                    self.metrics[k] = []

        # Truncate to current epoch
        for k in self.metrics:
            if len(self.metrics[k]) > epoch:
                self.metrics[k] = self.metrics[k][:epoch]

        # Append values
        for k in self.metrics:
            if k == "lr":
                continue
            if k in logs:
                self.metrics[k].append(float(logs[k]))

        # LR — call the schedule with current optimizer step (fix: was using get_value which returns 0)
        try:
            opt = self.model.optimizer
            if hasattr(opt, "inner_optimizer"):
                opt = opt.inner_optimizer
            lr_schedule = opt.learning_rate
            if callable(lr_schedule):
                lr = float(lr_schedule(opt.iterations).numpy())
            else:
                lr = float(tf.keras.backend.get_value(lr_schedule))
        except Exception:
            lr = 0.0
        self.metrics["lr"].append(lr)

        with open(self.json_path, "w") as f:
            json.dump(self.metrics, f, indent=2)

        self._plot_chart()

    def _plot_chart(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        loss = self.metrics.get("loss", [])
        if not loss:
            return
        epochs = range(1, len(loss) + 1)

        val_loss = self.metrics.get("val_loss", [])
        lr_vals = self.metrics.get("lr", [])

        # Find an accuracy-like metric
        acc_key = None
        val_acc_key = None
        for k in self.metrics:
            if k in ("loss", "val_loss", "lr"):
                continue
            if k.startswith("val_"):
                val_acc_key = k
            elif acc_key is None:
                acc_key = k

        n_plots = 2 + (1 if acc_key and self.metrics.get(acc_key) else 0)
        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
        if n_plots == 1:
            axes = [axes]

        ax = axes[0]
        ax.plot(epochs, loss, "o-", lw=2, color="#e74c3c", label="Train Loss", markersize=3)
        if val_loss:
            ax.plot(epochs, val_loss, "s-", lw=2, color="#3498db", label="Val Loss", markersize=3)
        ax.set_title("Loss", fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.legend(); ax.grid(True, ls="--", alpha=0.4)

        if acc_key and self.metrics.get(val_acc_key):
            ax = axes[1]
            train = self.metrics.get(acc_key, [])
            val = self.metrics.get(val_acc_key, [])
            label = acc_key.replace("_", " ").title()
            ax.plot(epochs, train, "o-", lw=2, color="#2ecc71", label=f"Train {label}", markersize=3)
            if val:
                ax.plot(epochs, val, "s-", lw=2, color="#9b59b6", label=f"Val {label}", markersize=3)
            ax.set_title(label, fontsize=13, fontweight="bold")
            ax.set_xlabel("Epoch"); ax.set_ylabel(label)
            ax.legend(); ax.grid(True, ls="--", alpha=0.4)

        ax = axes[-1] if n_plots > 2 else (axes[1] if axes[1] else plt.subplot())
        if lr_vals:
            ax.semilogy(epochs, lr_vals, "o-", lw=2, color="#e67e22", markersize=3)
        ax.set_title("Learning Rate", fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("LR (log scale)")
        ax.grid(True, ls="--", alpha=0.4)

        plt.tight_layout()
        plt.savefig(self.png_path, dpi=150, bbox_inches="tight")
        plt.close()


# ──────────────────────────────────────────────────────────────
# Verification Callback — EER / ROC / TAR@FAR on UNSEEN identities
# ──────────────────────────────────────────────────────────────
class VerificationCallback(tf.keras.callbacks.Callback):
    def __init__(self, backbone, test_dir, img_size=112, num_pairs=10000,
                 eval_every=5, batch_size=128, seed=42, save_dir="results"):
        super().__init__()
        self.backbone = backbone
        self.test_dir = Path(test_dir)
        self.img_size = img_size
        self.num_pairs = num_pairs
        self.eval_every = eval_every
        self.batch_size = batch_size
        self.seed = seed
        self.save_dir = save_dir
        self._prepared = False
        self._id_to_imgs = None
        self._identities = None
        self.verification_log = []
        self.verification_log_path = os.path.join(self.save_dir, "verification_log.json")
        if os.path.exists(self.verification_log_path):
            try:
                with open(self.verification_log_path, "r") as f:
                    loaded = json.load(f)
                    self.verification_log = [entry for entry in loaded
                                              if entry.get("epoch", 0) <= 0]
            except Exception:
                pass

    def _prepare(self):
        if self._prepared:
            return
        self._id_to_imgs = defaultdict(list)
        for cls_dir in self.test_dir.iterdir():
            if not cls_dir.is_dir():
                continue
            imgs = [p for p in cls_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
            if len(imgs) >= 2:
                self._id_to_imgs[cls_dir.name] = imgs
        self._identities = sorted(self._id_to_imgs.keys())
        self._prepared = True
        print(f"[Verification] Loaded {len(self._identities)} test identities from {self.test_dir}")

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) < 2:
            return
        if (epoch + 1) % self.eval_every != 0:
            return
        self._prepare()
        if len(self._identities) < 2:
            return

        rng = random.Random(self.seed + epoch)

        pos_a, pos_b = [], []
        neg_a, neg_b = [], []
        for _ in range(self.num_pairs):
            identity = rng.choice(self._identities)
            a, b = rng.sample(self._id_to_imgs[identity], 2)
            pos_a.append(a); pos_b.append(b)
        for _ in range(self.num_pairs):
            id1, id2 = rng.sample(self._identities, 2)
            a = rng.choice(self._id_to_imgs[id1])
            b = rng.choice(self._id_to_imgs[id2])
            neg_a.append(a); neg_b.append(b)

        all_paths = list(set(pos_a + pos_b + neg_a + neg_b))
        path_identity = {}
        for p in all_paths:
            parent = os.path.basename(os.path.dirname(str(p)))
            path_identity[p] = parent
        all_images = []
        for p in all_paths:
            img = load_image_tf(str(p), self.img_size)
            all_images.append(tf.cast(img, tf.float16).numpy())
        all_images = np.stack(all_images, axis=0)

        embs_list = []
        for i in range(0, len(all_images), self.batch_size):
            batch = all_images[i:i + self.batch_size]
            embs_list.append(self.backbone.predict(batch, batch_size=self.batch_size, verbose=0))
        all_embs = np.concatenate(embs_list, axis=0).astype(np.float32)
        norms = np.linalg.norm(all_embs, axis=1, keepdims=True)
        all_embs = all_embs / np.maximum(norms, 1e-12)

        path_to_idx = {p: i for i, p in enumerate(all_paths)}

        def cosine(paths_a, paths_b):
            a_mat = np.stack([all_embs[path_to_idx[p]] for p in paths_a])
            b_mat = np.stack([all_embs[path_to_idx[p]] for p in paths_b])
            return np.sum(a_mat * b_mat, axis=1)

        pos_scores = cosine(pos_a, pos_b)
        neg_scores = cosine(neg_a, neg_b)
        scores = np.concatenate([pos_scores, neg_scores])
        y_true = np.array([1] * len(pos_scores) + [0] * len(neg_scores))

        eer, eer_thresh = self._compute_eer(y_true, scores)
        best_acc, best_thresh, prec, rec = self._best_threshold_metrics(y_true, scores)

        pos_mean, neg_mean = float(np.mean(pos_scores)), float(np.mean(neg_scores))
        pos_std, neg_std = float(np.std(pos_scores)), float(np.std(neg_scores))

        print(f"\n{'='*60}")
        print(f"[Verification] Epoch {epoch+1} — {len(self._identities)} unseen test identities")
        print(f"{'='*60}")
        print(f"  EER:              {eer:.4f} ({eer*100:.2f}%)  @thresh={eer_thresh:.4f}")
        print(f"  Best Accuracy:    {best_acc:.4f}  @thresh={best_thresh:.4f}")
        print(f"  Precision/Recall: {prec:.4f} / {rec:.4f}")
        print(f"  Pos sim:  mean={pos_mean:.4f} ± {pos_std:.4f}")
        print(f"  Neg sim:  mean={neg_mean:.4f} ± {neg_std:.4f}")
        print(f"{'='*60}\n")

        if logs is not None:
            logs["val_eer"] = eer
            logs["val_best_acc"] = best_acc
            logs["val_pos_sim"] = pos_mean
            logs["val_neg_sim"] = neg_mean

        self.verification_log.append({
            "epoch": epoch + 1, "eer": eer, "best_acc": best_acc,
            "best_thresh": best_thresh, "precision": prec, "recall": rec,
            "pos_mean": pos_mean, "pos_std": pos_std,
            "neg_mean": neg_mean, "neg_std": neg_std,
        })
        with open(self.verification_log_path, "w") as f:
            json.dump(self.verification_log, f, indent=2)

        self._plot_verification(pos_scores, neg_scores, y_true, scores, eer,
                                all_embs, all_paths, path_identity)

    def _compute_eer(self, y_true, scores):
        order = np.argsort(scores)[::-1]
        y = y_true[order].astype(np.int32)
        n_pos, n_neg = int(np.sum(y)), int(len(y) - np.sum(y))
        if n_pos == 0 or n_neg == 0:
            return 1.0, 0.0
        tp, fp = np.cumsum(y), np.cumsum(1 - y)
        tpr, fpr = tp / n_pos, fp / n_neg
        fnr = 1.0 - tpr
        i = np.argmin(np.abs(fpr - fnr))
        return float((fpr[i] + fnr[i]) / 2.0), float(scores[order][i])

    def _best_threshold_metrics(self, y_true, scores):
        order = np.argsort(scores)[::-1]
        y = y_true[order].astype(np.int32)
        n_pos = int(np.sum(y)); n_neg = len(y) - n_pos
        tp, fp = np.cumsum(y), np.cumsum(1 - y)
        tn, fn = n_neg - fp, n_pos - tp
        acc = (tp + tn) / len(y)
        best_i = np.argmax(acc)
        return (float(acc[best_i]), float(scores[order][best_i]),
                float(tp[best_i] / max(tp[best_i] + fp[best_i], 1)),
                float(tp[best_i] / max(n_pos, 1)))

    def _plot_verification(self, pos_scores, neg_scores, y_true, scores, eer,
                            embeddings, paths, path_identity):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(self.save_dir, exist_ok=True)

        # Score distribution
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.hist(neg_scores, bins=60, alpha=0.5, color="#e74c3c", label=f"Different (n={len(neg_scores)})", density=True)
        ax.hist(pos_scores, bins=60, alpha=0.5, color="#2ecc71", label=f"Same (n={len(pos_scores)})", density=True)
        ax.set_xlabel("Cosine Similarity"); ax.set_ylabel("Density")
        ax.set_title(f"Score Distribution (EER={eer:.4f})", fontweight="bold")
        ax.legend(); ax.grid(True, ls="--", alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(self.save_dir, "verification_distribution.png"), dpi=150); plt.close()

        # ROC curve
        order = np.argsort(scores)[::-1]
        y = y_true[order].astype(np.int32)
        n_pos, n_neg = int(np.sum(y)), int(len(y) - np.sum(y))
        if n_pos > 0 and n_neg > 0:
            tp, fp = np.cumsum(y), np.cumsum(1 - y)
            tpr_arr, fpr_arr = tp / n_pos, fp / n_neg
            auc = float(np.trapz(tpr_arr, fpr_arr))

            fig, ax = plt.subplots(1, 1, figsize=(7, 7))
            ax.plot(fpr_arr, tpr_arr, "b-", lw=2.5, label=f"ROC (AUC={auc:.4f})")
            ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
            ax.plot(eer, 1 - eer, "ro", ms=10, label=f"EER={eer:.4f}")
            ax.axhline(y=1-eer, color="r", ls="--", alpha=0.3)
            ax.axvline(x=eer, color="r", ls="--", alpha=0.3)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
            ax.set_title("ROC Curve", fontweight="bold")
            ax.legend(loc="lower right"); ax.grid(True, ls="--", alpha=0.3)
            plt.tight_layout(); plt.savefig(os.path.join(self.save_dir, "verification_roc.png"), dpi=150); plt.close()
        print(f"[Plot] Verification charts saved to {self.save_dir}/")

        self._plot_embedding_tsne(embeddings, paths, path_identity)

    def _plot_embedding_tsne(self, embeddings, paths, path_identity):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        try:
            from sklearn.manifold import TSNE
        except ImportError:
            print("[Plot] sklearn not installed. Skipping t-SNE.")
            return

        rng = random.Random(self.seed)
        identities = sorted(set(path_identity.values()))
        sampled_ids = rng.sample(identities, min(10, len(identities)))

        sampled_embs, sampled_labels = [], []
        for pid in sampled_ids:
            id_paths = [p for p in paths if path_identity[p] == pid]
            id_paths = rng.sample(id_paths, min(10, len(id_paths)))
            for p in id_paths:
                idx = paths.index(p)
                sampled_embs.append(embeddings[idx])
                sampled_labels.append(pid)

        if len(sampled_labels) < 4:
            return

        sampled_embs = np.array(sampled_embs)
        tsne = TSNE(n_components=2, perplexity=min(30, len(sampled_embs) - 1),
                    random_state=self.seed, max_iter=500)
        emb_2d = tsne.fit_transform(sampled_embs)

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        unique_labels = sorted(set(sampled_labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        for i, lbl in enumerate(unique_labels):
            mask = np.array([l == lbl for l in sampled_labels])
            ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1], c=[colors[i]],
                       label=lbl[:15], s=30, alpha=0.8, edgecolors="k", linewidth=0.3)

        ax.set_title(f"t-SNE — Test Identity Embeddings ({len(sampled_ids)} ids)", fontweight="bold", fontsize=13)
        ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=6, ncol=2)
        ax.grid(True, ls="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "verification_tsne.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Plot] t-SNE embedding visualization saved.")


# ──────────────────────────────────────────────────────────────
# Full-State Checkpoint Callback (for resume)
# ──────────────────────────────────────────────────────────────
class TrainingCheckpointCallback(tf.keras.callbacks.Callback):
    def __init__(self, full_ckpt, latest_mgr, best_mgr, epoch_var, meta_path,
                 optimizer, monitor="val_loss"):
        super().__init__()
        self.full_ckpt = full_ckpt
        self.latest_mgr = latest_mgr
        self.best_mgr = best_mgr
        self.epoch_var = epoch_var
        self.meta_path = meta_path
        self.optimizer = optimizer
        self.monitor = monitor
        self.best_val = float("inf")
        self._load_history()

    def _load_history(self):
        if not os.path.exists(self.meta_path):
            return
        try:
            with open(self.meta_path, "r") as f:
                meta = json.load(f)
            self.best_val = float(meta.get("best_val", float("inf")))
        except Exception:
            pass

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch_var.assign(epoch + 1)

        latest_path = self._save_with_retry(self.latest_mgr, epoch + 1)

        val = next((float(logs[k]) for k in logs if self.monitor in k or k == self.monitor), 0.0)
        if val < self.best_val:
            self.best_val = val
            best_path = self._save_with_retry(self.best_mgr, epoch + 1)
            print(f"[Best] New best {self.monitor}={val:.4f} @ epoch {epoch+1} -> {best_path}")

        meta = {
            "last_epoch": epoch, "next_epoch": epoch + 1,
            "latest_checkpoint": latest_path,
            "best_val": self.best_val,
        }
        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[Checkpoint] Epoch {epoch+1} saved.")

    def _save_with_retry(self, manager, ckpt_number, max_retries=3, delay=2):
        """Save checkpoint with retry on Windows file-locking errors."""
        import time
        last_error = None
        for attempt in range(max_retries):
            try:
                return manager.save(checkpoint_number=ckpt_number)
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    print(f"[Checkpoint] Save failed (attempt {attempt+1}/{max_retries}), "
                          f"retrying in {delay}s... [{type(e).__name__}]")
                    time.sleep(delay)
        raise last_error


# ──────────────────────────────────────────────────────────────
# Save + Metric callbacks
# ──────────────────────────────────────────────────────────────
class SaveBestBackboneCallback(tf.keras.callbacks.Callback):
    def __init__(self, backbone, filepath):
        super().__init__()
        self.backbone = backbone
        self.filepath = filepath
        self.best = float("inf")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_loss = next((float(logs[k]) for k in logs
                         if k in ("val_loss", "val_siamese_loss")), float("inf"))
        if val_loss < self.best:
            self.best = val_loss
            self.backbone.save_weights(self.filepath)
            print(f"[Backbone] Best weights saved (val_loss={val_loss:.4f})")


class MetricAdapterCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        if "prob_accuracy" in logs and "accuracy" not in logs:
            logs["accuracy"] = logs["prob_accuracy"]
        if "val_prob_accuracy" in logs and "val_accuracy" not in logs:
            logs["val_accuracy"] = logs["val_prob_accuracy"]


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Siamese iResNet50 — One-shot Face Verification")
    # Data
    parser.add_argument("--dataset_dir", default="dataset_ms1m/train")
    parser.add_argument("--val_dir",     default="dataset_ms1m/val")
    parser.add_argument("--test_dir",    default="dataset_ms1m/test")
    # Training
    parser.add_argument("--batch_size",        type=int, default=100)
    parser.add_argument("--epochs",            type=int, default=80)
    parser.add_argument("--img_size",          type=int, default=112)
    parser.add_argument("--embedding_dim",     type=int, default=512)
    parser.add_argument("--dropout",           type=float, default=0.35)
    parser.add_argument("--lr",                type=float, default=0.0007)
    parser.add_argument("--pairs_per_epoch",   type=int, default=200000)
    parser.add_argument("--val_pairs",         type=int, default=10000)
    parser.add_argument("--patience",          type=int, default=30)
    parser.add_argument("--warmup_epochs",     type=int, default=8)
    parser.add_argument("--seed",              type=int, default=42)
    # Contrastive loss
    parser.add_argument("--contrastive_weight",  type=float, default=0.2)
    parser.add_argument("--margin_start",        type=float, default=0.4)
    parser.add_argument("--margin_end",          type=float, default=0.0)
    # Hard negative mining
    parser.add_argument("--hard_mine_every",       type=int, default=5)
    parser.add_argument("--hard_mine_start",       type=int, default=10)
    parser.add_argument("--hard_mine_samples",     type=int, default=5000)
    parser.add_argument("--hard_mine_topk",        type=int, default=200)
    parser.add_argument("--hard_ratio_start",      type=float, default=0.1)
    parser.add_argument("--hard_ratio_end",        type=float, default=0.5)
    # Verification
    parser.add_argument("--verify_every",      type=int, default=2)
    parser.add_argument("--verify_pairs",      type=int, default=5000)
    # Resume
    parser.add_argument("--resume", action="store_true")
    # Triplet loss (replaces siamese when --loss_type triplet)
    parser.add_argument("--loss_type",  default="siamese", choices=["siamese", "triplet"],
                        help="siamese: BCE+contrastive; triplet: batch-hard triplet loss")
    parser.add_argument("--p_per_batch", type=int, default=40,
                        help="P identities per PK batch (triplet only; total batch = P*K)")
    parser.add_argument("--k_per_id",    type=int, default=5,
                        help="K images per identity (triplet only; total batch = P*K)")
    parser.add_argument("--triplet_margin", type=float, default=0.3,
                        help="Triplet loss margin (cosine distance)")
    parser.add_argument("--triplet_batches_per_epoch", type=int, default=1000,
                        help="Number of PK batches per epoch (triplet only)")
    args = parser.parse_args()

    configure_gpu()
    random.seed(args.seed); np.random.seed(args.seed)

    # Results directory
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    latest_ckpt_dir = os.path.join(results_dir, "checkpoints", "latest")
    best_ckpt_dir   = os.path.join(results_dir, "checkpoints", "best")
    os.makedirs(best_ckpt_dir, exist_ok=True)
    meta_path = os.path.join(results_dir, "training_meta.json")

    # ── Load data ──
    print(f"\n{'='*60}")
    print("Siamese iResNet50 — One-Shot Face Verification")
    print(f"{'='*60}\n")
    print("[1/5] Loading datasets...")
    train_ids = get_identity_images(args.dataset_dir)
    val_ids   = get_identity_images(args.val_dir)
    print(f"  Train: {len(train_ids)} identities, {sum(len(v) for v in train_ids.values())} images")
    print(f"  Val:   {len(val_ids)} identities, {sum(len(v) for v in val_ids.values())} images")

    # ── Build datasets ──
    print("\n[2/5] Building tf.data pipelines...")

    if args.loss_type == "triplet":
        total_batch = args.p_per_batch * args.k_per_id
        steps_per_epoch = args.triplet_batches_per_epoch
        train_ds = build_triplet_dataset(
            train_ids, args.p_per_batch, args.k_per_id,
            img_size=args.img_size, is_training=True, seed=args.seed,
        )
        val_ds = build_triplet_dataset(
            val_ids, max(4, args.p_per_batch // 4), args.k_per_id,
            img_size=args.img_size, is_training=False, seed=args.seed + 1,
        )
        print(f"  PK-sampling: P={args.p_per_batch}, K={args.k_per_id} → batch={total_batch}")
        print(f"  Steps/epoch: {steps_per_epoch}")

    else:
        steps_per_epoch = args.pairs_per_epoch // args.batch_size
        train_ds = build_pair_dataset(
            train_ids, args.pairs_per_epoch, args.batch_size,
            img_size=args.img_size, is_training=True, seed=args.seed,
            use_hard_negatives=True,
        )
        val_ds = build_pair_dataset(
            val_ids, args.val_pairs, args.batch_size,
            img_size=args.img_size, is_training=False, seed=args.seed + 1,
            use_hard_negatives=False,
        )
        print(f"  Steps/epoch: {steps_per_epoch} (pairs: {args.pairs_per_epoch}, batch: {args.batch_size})")

    # ── Build model ──
    print(f"\n[3/5] Building {args.loss_type.upper()} model...")

    if args.loss_type == "triplet":
        model, backbone = build_triplet_model(
            input_shape=(args.img_size, args.img_size, 3),
            embedding_dim=args.embedding_dim,
            dropout_rate=args.dropout,
        )
        backbone.summary()
        print(f"  Triplet params: {model.count_params():,}")
        print(f"  Triplet margin: {args.triplet_margin}")
        print(f"  PK-sampling:    P={args.p_per_batch}, K={args.k_per_id}")

    else:
        use_contrastive = args.contrastive_weight > 0.0
        model, backbone = build_siamese_model(
            input_shape=(args.img_size, args.img_size, 3),
            embedding_dim=args.embedding_dim,
            dropout_rate=args.dropout,
            include_contrastive=use_contrastive,
        )
        backbone.summary()
        print(f"  Siamese params: {model.count_params():,}")
        print(f"  Contrastive loss: {'ON' if use_contrastive else 'OFF'} "
              f"(w={args.contrastive_weight})")
        print(f"  Margin schedule: {args.margin_start:.2f} → {args.margin_end:.2f} (cosine decay)")
        print(f"  Hard mining:     start@{args.hard_mine_start}, "
              f"ratio {args.hard_ratio_start:.1f}→{args.hard_ratio_end:.1f}")

    # ── Optimizer + LR schedule ──
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch
    lr_schedule = WarmupCosineDecay(args.lr, warmup_steps, total_steps)

    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
        tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    )

    # ── Compile ──
    if args.loss_type == "triplet":
        model.compile(
            optimizer=optimizer,
            loss=batch_hard_triplet_loss(margin=args.triplet_margin),
            metrics=[triplet_active_fraction(margin=args.triplet_margin)],
        )
    else:
        use_contrastive = args.contrastive_weight > 0.0
        get_margin_variable(args.margin_start)
        if use_contrastive:
            model.compile(
                optimizer=optimizer,
                loss={"prob": "binary_crossentropy",
                      "embeddings": make_contrastive_loss()},
                loss_weights={"prob": 1.0, "embeddings": args.contrastive_weight},
                metrics={"prob": "accuracy"},
            )
        else:
            model.compile(
                optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"],
            )

    # ── Checkpoint infra ──
    checkpoint_epoch = tf.Variable(0, dtype=tf.int64, trainable=False, name="epoch")
    full_ckpt = tf.train.Checkpoint(model=model, backbone=backbone,
                                    optimizer=optimizer, epoch=checkpoint_epoch)
    latest_mgr = tf.train.CheckpointManager(full_ckpt, latest_ckpt_dir, max_to_keep=2)
    best_mgr   = tf.train.CheckpointManager(full_ckpt, best_ckpt_dir, max_to_keep=1)

    # ── Resume ──
    initial_epoch = 0
    restored_from_checkpoint = False
    if args.resume:
        latest_ckpt_path = tf.train.latest_checkpoint(latest_ckpt_dir)
        if latest_ckpt_path:
            # Verify checkpoint integrity before restore
            _verify_ckpt_integrity(latest_ckpt_path)
            full_ckpt.restore(latest_ckpt_path).expect_partial()
            initial_epoch = min(int(checkpoint_epoch.numpy()), args.epochs)
            restored_from_checkpoint = True
            print(f"\n[Resume] Restored from {latest_ckpt_path}")
            print(f"  Starting epoch: {initial_epoch + 1}")
            # optimizer.iterations is restored by tf.train.Checkpoint — no start_step needed
            opt = optimizer.inner_optimizer if hasattr(optimizer, "inner_optimizer") else optimizer
            cur_step = int(opt.iterations.numpy())
            cur_lr = float(opt.learning_rate(opt.iterations).numpy())
            print(f"  Optimizer step: {cur_step}")
            print(f"  Current LR:     {cur_lr:.6f}")
        if initial_epoch >= args.epochs:
            print("[Resume] Already completed all epochs.")
            return

    if not restored_from_checkpoint and initial_epoch > 0:
        # Weights-only restore (no optimizer state): manually advance optimizer step
        start_step = float(initial_epoch * steps_per_epoch)
        opt = optimizer.inner_optimizer if hasattr(optimizer, "inner_optimizer") else optimizer
        opt.iterations.assign(int(start_step))
        print(f"[Resume] Weights-only: set optimizer.iterations to {int(start_step)}")

    # ── Callbacks ──
    best_backbone_path = os.path.join(results_dir, "best_backbone.h5")

    callbacks = []
    if args.loss_type == "siamese" and args.contrastive_weight > 0.0:
        callbacks.append(MetricAdapterCallback())
        callbacks.append(ProgressiveMarginCallback(
            margin_start=args.margin_start, margin_end=args.margin_end,
            total_epochs=args.epochs,
        ))

    ckpt_name = "best_triplet.h5" if args.loss_type == "triplet" else "best_siamese.h5"
    callbacks.extend([
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(results_dir, ckpt_name),
            save_best_only=True, save_weights_only=True,
            monitor="val_loss", mode="min", verbose=1,
        ),
        SaveBestBackboneCallback(backbone, best_backbone_path),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=args.patience,
            restore_best_weights=True, verbose=1,
        ),
        HistoryPlotterCallback(results_dir),
        TrainingCheckpointCallback(full_ckpt, latest_mgr, best_mgr,
                                   checkpoint_epoch, meta_path, optimizer),
    ])

    # Hard negative mining (siamese only)
    if args.loss_type == "siamese" and args.hard_mine_start < args.epochs:
        callbacks.append(HardNegativeMiningCallback(
            backbone, train_ids, args.hard_mine_every, args.hard_mine_start,
            args.hard_mine_samples, args.hard_mine_topk, args.img_size,
            args.batch_size, args.seed,
            ratio_start=args.hard_ratio_start, ratio_end=args.hard_ratio_end,
        ))

    # Verification eval (works for both modes — uses backbone directly)
    if args.test_dir and os.path.isdir(args.test_dir):
        callbacks.append(VerificationCallback(
            backbone, args.test_dir, img_size=args.img_size,
            num_pairs=args.verify_pairs, eval_every=args.verify_every,
            batch_size=args.batch_size, seed=args.seed, save_dir=results_dir,
        ))

    # ── Train ──
    print(f"\n{'='*60}")
    print(f"Training Configuration ({args.loss_type.upper()})")
    print(f"{'='*60}")
    print(f"  Epochs:          {args.epochs}")
    total_batch = args.p_per_batch * args.k_per_id if args.loss_type == "triplet" else args.batch_size
    print(f"  Batch size:      {total_batch} {'(PK: P=' + str(args.p_per_batch) + ', K=' + str(args.k_per_id) + ')' if args.loss_type == 'triplet' else ''}")
    print(f"  Steps/epoch:     {steps_per_epoch}")
    print(f"  Image size:      {args.img_size}x{args.img_size}")
    print(f"  Embedding dim:   {args.embedding_dim}")
    print(f"  Dropout:         {args.dropout}")
    print(f"  Peak LR:         {args.lr}")
    print(f"  Warmup epochs:   {args.warmup_epochs}")
    print(f"  Patience:        {args.patience}")
    print(f"  Mixed precision: ON")
    print(f"{'='*60}\n")

    print("[4/5] Training...")
    val_steps = (args.val_pairs // total_batch) if args.loss_type == "triplet" else (args.val_pairs // args.batch_size)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
    )

    print("\n[5/5] Saving final weights...")
    final_name = "final_triplet.h5" if args.loss_type == "triplet" else "final_siamese.h5"
    model.save_weights(os.path.join(results_dir, final_name))
    backbone.save_weights(os.path.join(results_dir, "final_backbone.h5"))
    print(f"\n{'='*60}")
    print(f"[DONE] {args.loss_type.upper()} training complete.")
    print(f"  Results:        {results_dir}/")
    print(f"  Best backbone:  {best_backbone_path}")
    print(f"  Best model:     {results_dir}/{ckpt_name}")
    print(f"  Resume:         --resume to continue")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
