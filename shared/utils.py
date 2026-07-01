import os
import math
import random
import gc
from pathlib import Path
from collections import defaultdict

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =====================================================================
# 1. Hardware Optimization
# =====================================================================
def configure_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[GPU] Memory Growth enabled. {len(gpus)} GPU(s) detected.")
        except RuntimeError as e:
            print(f"[GPU] Error: {e}")
    else:
        print("[GPU] No GPU detected. Running on CPU.")

def configure_tfdata():
    opts = tf.data.Options()
    opts.experimental_deterministic = False
    return opts

# =====================================================================
# 2. Callbacks
# =====================================================================
class ProgressiveMarginCallback(tf.keras.callbacks.Callback):
    """Gradually increases ArcFace margin from start to end over epochs."""
    def __init__(self, arcface_layer, start_margin=0.10, end_margin=0.50, total_epochs=10):
        super().__init__()
        self.arcface_layer = arcface_layer
        self.start_m = start_margin
        self.end_m = end_margin
        self.total = total_epochs

    def on_epoch_begin(self, epoch, logs=None):
        progress = min(epoch / max(self.total - 1, 1), 1.0)
        new_m = self.start_m + progress * (self.end_m - self.start_m)
        self.arcface_layer.margin = new_m
        self.arcface_layer._update_margin_constants(new_m)
        print(f"\n[ArcFace] Epoch {epoch+1}: margin = {new_m:.3f} rad ({math.degrees(new_m):.1f} deg)")

class SaveBestBackbone(tf.keras.callbacks.Callback):
    """Saves backbone weights whenever val_loss improves."""
    def __init__(self, backbone, filepath):
        super().__init__()
        self.backbone = backbone
        self.filepath = filepath
        self.best_val_loss = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss', float('inf'))
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.backbone.save_weights(self.filepath)
            print(f"[Backbone] Best weights saved (val_loss={val_loss:.4f})")

class VerificationCallback(tf.keras.callbacks.Callback):
    """Evaluate face verification (EER) on unseen test identities every N epochs.

    This is the KEY metric for one-shot learning — measures whether the backbone
    produces embeddings that can distinguish unseen identities.
    """
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, backbone, test_dir, img_size=112, num_pairs=3000,
                 eval_every=3, batch_size=64, seed=42, save_dir="results", file_prefix=""):
        super().__init__()
        self.backbone = backbone
        self.test_dir = Path(test_dir)
        self.img_size = img_size
        self.num_pairs = num_pairs
        self.eval_every = eval_every
        self.batch_size = batch_size
        self.seed = seed
        self.save_dir = save_dir
        self.file_prefix = file_prefix
        self._prepared = False
        self._id_to_imgs = None
        self._identities = None

    def _prepare_data(self):
        """Scan test directory and build identity -> image mapping."""
        if self._prepared:
            return
        self._id_to_imgs = defaultdict(list)
        for cls_dir in self.test_dir.iterdir():
            if not cls_dir.is_dir():
                continue
            imgs = [p for p in cls_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in self.IMAGE_EXTS]
            if len(imgs) >= 2:
                self._id_to_imgs[cls_dir.name] = imgs
        self._identities = sorted(self._id_to_imgs.keys())
        self._prepared = True
        print(f"[Verification] Loaded {len(self._identities)} test identities")

    def _load_image(self, path):
        img = tf.io.read_file(str(path))
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, [self.img_size, self.img_size])
        img = (tf.cast(img, tf.float32) - 127.5) / 128.0
        return img

    def _compute_eer_fpr_tpr(self, y_true, scores):
        """Compute EER and return (fpr, tpr) arrays for threshold analysis."""
        order = np.argsort(scores)[::-1]
        y = y_true[order].astype(np.int32)
        n_pos = int(np.sum(y))
        n_neg = int(len(y) - n_pos)
        if n_pos == 0 or n_neg == 0:
            return None, None, None
        tp = np.cumsum(y)
        fp = np.cumsum(1 - y)
        tpr = tp / max(n_pos, 1)
        fpr = fp / max(n_neg, 1)
        fnr = 1.0 - tpr
        i = int(np.argmin(np.abs(fpr - fnr)))
        eer = float((fpr[i] + fnr[i]) / 2.0)
        return eer, fpr, tpr

    @staticmethod
    def _compute_tar_at_far(fpr, tpr, far_targets=[1e-3, 1e-4]):
        """Compute TAR (1-FRR) at specific FAR thresholds (interpolated)."""
        results = {}
        for far in far_targets:
            idx = int(np.searchsorted(fpr, far, side='right'))
            if idx >= len(tpr):
                results[far] = 0.0
            else:
                results[far] = float(tpr[min(idx, len(tpr) - 1)])
        return results

    @staticmethod
    def _compute_best_threshold_accuracy(y_true, scores):
        """Find threshold that maximizes accuracy and return (accuracy, threshold, precision, recall)."""
        sorted_idx = np.argsort(scores)[::-1]
        sorted_scores = scores[sorted_idx]
        sorted_y = y_true[sorted_idx].astype(np.int32)
        tp = np.cumsum(sorted_y)
        fp = np.cumsum(1 - sorted_y)
        n_pos = int(np.sum(sorted_y))
        n_neg = len(sorted_y) - n_pos
        tn = n_neg - fp
        fn = n_pos - tp
        acc = (tp + tn) / len(sorted_y)
        best_i = int(np.argmax(acc))
        best_threshold = float(sorted_scores[best_i]) if best_i < len(sorted_scores) else 0.0
        best_acc = float(np.max(acc))
        best_precision = float(tp[best_i] / max(tp[best_i] + fp[best_i], 1))
        best_recall = float(tp[best_i] / max(n_pos, 1))
        return best_acc, best_threshold, best_precision, best_recall

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.eval_every != 0:
            return

        self._prepare_data()
        if len(self._identities) < 2:
            return

        rng = random.Random(self.seed + epoch)

        # Generate positive and negative pairs
        pos_paths_a, pos_paths_b = [], []
        neg_paths_a, neg_paths_b = [], []

        for _ in range(self.num_pairs):
            identity = rng.choice(self._identities)
            imgs = self._id_to_imgs[identity]
            a, b = rng.sample(imgs, 2)
            pos_paths_a.append(str(a))
            pos_paths_b.append(str(b))

        for _ in range(self.num_pairs):
            id1, id2 = rng.sample(self._identities, 2)
            a = rng.choice(self._id_to_imgs[id1])
            b = rng.choice(self._id_to_imgs[id2])
            neg_paths_a.append(str(a))
            neg_paths_b.append(str(b))

        # Collect all unique paths (as strings for hashing)
        all_paths = list(set(pos_paths_a + pos_paths_b + neg_paths_a + neg_paths_b))
        path_to_idx = {p: i for i, p in enumerate(all_paths)}
        total = len(all_paths)

        # ── Streaming inference: load + predict in chunks to avoid OOM ──
        emb_dim = None
        all_embs = np.zeros((total, 512), dtype=np.float32)  # pre-allocate embeddings only

        chunk_size = min(self.batch_size * 4, 512)  # up to 512 images per chunk
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            chunk_paths = all_paths[start:end]
            # Load images for this chunk
            chunk_imgs = np.stack([self._load_image(p).numpy() for p in chunk_paths])
            # Predict
            chunk_embs = self.backbone.predict(chunk_imgs, batch_size=self.batch_size, verbose=0)
            # L2 normalize
            norms = np.linalg.norm(chunk_embs, axis=1, keepdims=True)
            chunk_embs = chunk_embs / np.maximum(norms, 1e-12)
            all_embs[start:end] = chunk_embs
            # Free intermediate memory
            del chunk_imgs, chunk_embs

        # Compute cosine similarity
        def cosine(paths_a, paths_b):
            a_idx = [path_to_idx[p] for p in paths_a]
            b_idx = [path_to_idx[p] for p in paths_b]
            return np.sum(all_embs[a_idx] * all_embs[b_idx], axis=1)

        pos_scores = cosine(pos_paths_a, pos_paths_b)
        neg_scores = cosine(neg_paths_a, neg_paths_b)

        scores = np.concatenate([pos_scores, neg_scores])
        y_true = np.array([1] * len(pos_scores) + [0] * len(neg_scores))

        # Free paths and embeddings to reclaim RAM
        del all_embs, all_paths, pos_paths_a, pos_paths_b, neg_paths_a, neg_paths_b

        eer, fpr, tpr = self._compute_eer_fpr_tpr(y_true, scores)
        if eer is not None:
            if logs is not None:
                logs['val_eer'] = eer
            pos_mean = np.mean(pos_scores)
            neg_mean = np.mean(neg_scores)
            
            # TAR@FAR
            tar_results = self._compute_tar_at_far(fpr, tpr, far_targets=[1e-3, 1e-4])
            tar_1e3 = tar_results.get(1e-3, 0.0)
            tar_1e4 = tar_results.get(1e-4, 0.0)
            if logs is not None:
                logs['val_tar_far1e3'] = tar_1e3
                logs['val_tar_far1e4'] = tar_1e4
            
            # Accuracy @ best threshold
            best_acc, best_thresh, best_prec, best_rec = self._compute_best_threshold_accuracy(y_true, scores)
            if logs is not None:
                logs['val_acc_best_thresh'] = best_acc
            
            print(f"\n[Verification] EER={eer:.4f} | "
                  f"TAR@FAR=1e-3: {tar_1e3:.4f} | TAR@FAR=1e-4: {tar_1e4:.4f}")
            print(f"[Verification] Acc@best_thresh={best_acc:.4f} (thresh={best_thresh:.4f}) | "
                  f"Precision={best_prec:.4f} | Recall={best_rec:.4f}")
            print(f"[Verification] Pos_sim={pos_mean:.4f} | Neg_sim={neg_mean:.4f}")

            # Plot and save ROC Curve & Score Distribution dynamically
            try:
                from shared.eval_verification import compute_roc, compute_auc, plot_roc, plot_score_distribution
                fprs, tprs, _ = compute_roc(y_true, scores)
                auc = compute_auc(fprs, tprs)
                
                os.makedirs(self.save_dir, exist_ok=True)
                suffix = f"_{self.file_prefix}" if self.file_prefix else ""
                roc_path = os.path.join(self.save_dir, f"verification_roc{suffix}.png")
                dist_path = os.path.join(self.save_dir, f"verification_distribution{suffix}.png")
                
                plot_roc(fprs, tprs, eer, auc, roc_path)
                plot_score_distribution(pos_scores, neg_scores, dist_path)
                print(f"[Verification Plot] Saved latest verification plots to {self.save_dir}")
            except Exception as e:
                print(f"[Verification Plot Warning] Failed to save plots: {e}")

        # Final cleanup
        del scores, y_true, pos_scores, neg_scores
        import gc; gc.collect()


class HistoryPlotterCallback(tf.keras.callbacks.Callback):
    """Saves training history plot after every epoch, persisting across resumes.
    
    Auto-detects metrics from logs — supports both classification (accuracy)
    and triplet (triplet_active_fraction) training modes.
    """
    def __init__(self, save_path):
        super().__init__()
        self.save_path = save_path
        self.json_path = os.path.splitext(save_path)[0] + ".json"
        self.metrics = None  # built lazily from first epoch logs
        
        # Load existing history if resuming
        if os.path.exists(self.json_path):
            try:
                import json
                with open(self.json_path, 'r') as f:
                    self.metrics = json.load(f)
            except Exception:
                self.metrics = None
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # Auto-build metrics dict from first epoch
        if self.metrics is None:
            self.metrics = {}
            for k in logs:
                if isinstance(logs[k], (int, float)):
                    self.metrics[k] = []
        
        # Truncate lists to the current epoch to handle rewinding/resume overlaps
        for k in list(self.metrics.keys()):
            if len(self.metrics[k]) > epoch:
                self.metrics[k] = self.metrics[k][:epoch]
                
        # Append new values
        for k in self.metrics:
            if k in logs:
                self.metrics[k].append(float(logs[k]))
            else:
                self.metrics[k].append(None)  # maintain alignment
            
        import json
        with open(self.json_path, 'w') as f:
            json.dump(self.metrics, f)
            
        # Create a dummy object to mimic tf.keras.callbacks.History
        class DummyHistory:
            pass
        dummy = DummyHistory()
        dummy.history = self.metrics
        
        plot_history(dummy, self.save_path)

# =====================================================================
# 3. Learning Rate Schedule
# =====================================================================
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, peak_lr, warmup_steps, total_steps, min_lr=1e-6, start_step=0.0):
        super().__init__()
        self.peak_lr = peak_lr
        self.warmup_steps = float(warmup_steps)
        self.total_steps = float(total_steps)
        self.min_lr = min_lr
        self.start_step = float(start_step)

    def __call__(self, step):
        step = tf.cast(step, tf.float32) + self.start_step
        warmup = self.peak_lr * (step / tf.maximum(self.warmup_steps, 1.0))
        progress = (step - self.warmup_steps) / tf.maximum(self.total_steps - self.warmup_steps, 1.0)
        cosine = self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (1.0 + tf.cos(3.14159265 * progress))
        return tf.where(step < self.warmup_steps, warmup, cosine)

    def get_config(self):
        return {"peak_lr": self.peak_lr, "warmup_steps": self.warmup_steps,
                "total_steps": self.total_steps, "min_lr": self.min_lr, "start_step": self.start_step}

# =====================================================================
# 4. Data Pipeline
# =====================================================================
def _maybe_cache(ds, cache_mode, cache_path):
    if cache_mode == 'none':
        return ds
    if cache_mode == 'memory':
        return ds.cache()
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    return ds.cache(cache_path)

def build_dataset(
    train_dir,
    val_dir=None,
    batch_size=128,
    image_size=(112, 112),
    val_split=0.2,
    seed=42,
    cache_mode='disk',
    cache_dir='results/tf_cache',
    shuffle_buffer=100,
    crop_to_aspect_ratio=True,
):
    print(f"\n--- Building Data Pipeline ---")
    print(f"Train: {train_dir}")

    cache_train_path = os.path.join(cache_dir, 'train.cache')
    cache_val_path = os.path.join(cache_dir, 'val.cache')

    use_val_dir = False
    if val_dir and os.path.isdir(val_dir):
        # Allow different classes in val_dir for one-shot evaluation dataset
        use_val_dir = True
        print(f"Val:   {val_dir} (external validation set)")

    if use_val_dir:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            seed=seed,
            image_size=image_size,
            batch_size=batch_size,
            label_mode='int',
            shuffle=True,
            crop_to_aspect_ratio=crop_to_aspect_ratio,
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            val_dir,
            image_size=image_size,
            batch_size=batch_size,
            label_mode='int',
            shuffle=False,
            crop_to_aspect_ratio=crop_to_aspect_ratio,
        )
    else:
        print(f"Val:   split from train (val_split={val_split})")
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            validation_split=val_split,
            subset="training",
            seed=seed,
            image_size=image_size,
            batch_size=batch_size,
            label_mode='int',
            shuffle=True,
            crop_to_aspect_ratio=crop_to_aspect_ratio,
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            validation_split=val_split,
            subset="validation",
            seed=seed,
            image_size=image_size,
            batch_size=batch_size,
            label_mode='int',
            shuffle=False,
            crop_to_aspect_ratio=crop_to_aspect_ratio,
        )

    num_classes = len(train_ds.class_names)
    num_train_batches = tf.data.experimental.cardinality(train_ds).numpy()
    print(f"Train Classes: {num_classes} | Train batches/epoch: {num_train_batches}")

    def preprocess_train(image, label):
        image = (tf.cast(image, tf.float32) - 127.5) / 128.0
        # Data Augmentation
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, 0.15)
        image = tf.image.random_contrast(image, 0.85, 1.15)
        image = tf.image.random_saturation(image, 0.85, 1.15)
        image = tf.clip_by_value(image, -1.0, 1.0)
        return (image, label), label

    def preprocess_val(image, label):
        image = (tf.cast(image, tf.float32) - 127.5) / 128.0
        return (image, label), label

    opts = configure_tfdata()

    train_ds = train_ds.with_options(opts)
    train_ds = _maybe_cache(train_ds, cache_mode, cache_train_path)
    train_ds = (train_ds
                .map(preprocess_train, num_parallel_calls=tf.data.AUTOTUNE)
                .shuffle(buffer_size=shuffle_buffer, seed=seed, reshuffle_each_iteration=True)
                .prefetch(tf.data.AUTOTUNE))

    val_ds = val_ds.with_options(opts)
    val_ds = _maybe_cache(val_ds, cache_mode, cache_val_path)
    val_ds = (val_ds
              .map(preprocess_val, num_parallel_calls=tf.data.AUTOTUNE)
              .prefetch(tf.data.AUTOTUNE))

    return train_ds, val_ds, num_classes, num_train_batches

# =====================================================================
# 5. Plot History
# =====================================================================
def plot_history(history, save_path):
    """Flexible plot: handles both classification (accuracy) and triplet (active_fraction) metrics."""
    h = history.history
    loss = h.get('loss', [])
    val_loss = h.get('val_loss', [])
    
    # Find an "accuracy-like" metric for the right subplot
    acc_keys = ['accuracy', 'sparse_categorical_accuracy', 'prob_accuracy',
                'triplet_active_fraction', 'val_accuracy', 'val_sparse_categorical_accuracy',
                'val_prob_accuracy', 'val_triplet_active_fraction']
    train_acc = None
    for k in acc_keys:
        if k in h and not k.startswith('val_') and h[k] and any(v is not None for v in h[k]):
            train_acc = h[k]; break
    val_acc = None
    for k in acc_keys:
        if k.startswith('val_') and k in h and h[k] and any(v is not None for v in h[k]):
            val_acc = h[k]; break
    # Fallback: try non-prefixed val_ keys after looking at prefix
    if val_acc is None:
        for k in h:
            if k.startswith('val_') and k not in ('val_loss',) and h[k] and any(v is not None for v in h[k]):
                val_acc = h[k]; break
    
    epochs_range = range(1, len(loss) + 1) if loss else range(1)

    plt.figure(figsize=(15, 6))
    
    # Left: Loss
    plt.subplot(1, 2, 1)
    if loss:
        plt.plot(epochs_range, loss, label='Train Loss', color='#e74c3c', lw=2.5)
    if val_loss:
        plt.plot(epochs_range, val_loss, label='Val Loss', color='#3498db', lw=2.5)
    plt.title('Loss', fontsize=13, fontweight='bold')
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend(); plt.grid(True, ls='--', alpha=0.6)

    # Right: Accuracy / Active Fraction
    plt.subplot(1, 2, 2)
    acc_label = (train_acc[0] if train_acc and train_acc[0] is not None else 'accuracy') if train_acc else ''
    if 'triplet_active_fraction' in str(train_acc) or 'triplet' in str(acc_label).lower():
        label = 'Active Fraction (lower=better)'
        plt.gca().invert_yaxis()  # lower is better for active fraction
    else:
        label = 'Accuracy'
    if train_acc is not None and train_acc:
        valid_train = [(i+1, v) for i, v in enumerate(train_acc) if v is not None]
        if valid_train:
            tx, ty = zip(*valid_train)
            plt.plot(tx, ty, label=f'Train {label}', color='#2ecc71', lw=2.5)
    if val_acc is not None and val_acc:
        valid_val = [(i+1, v) for i, v in enumerate(val_acc) if v is not None]
        if valid_val:
            vx, vy = zip(*valid_val)
            plt.plot(vx, vy, label=f'Val {label}', color='#9b59b6', lw=2.5)
    plt.title(label, fontsize=13, fontweight='bold')
    plt.xlabel('Epochs'); plt.ylabel(label); plt.legend(); plt.grid(True, ls='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"\n[INFO] Plot saved to: {save_path}")

# =====================================================================
# 6. Test-Time Augmentation (TTA) — Flip + Average Embedding
# =====================================================================
def compute_embedding_tta(backbone, images, batch_size=64, l2_norm=True):
    """Compute embedding with horizontal flip TTA.
    
    Averages the L2-normalized embeddings of the original and horizontally
    flipped image. Typically improves TAR by 0.3-0.5%.
    
    Args:
        backbone: Keras model that outputs embeddings
        images: numpy array (N, H, W, 3) in [-1, 1] range
        batch_size: Inference batch size
        l2_norm: If True, L2-normalize output embeddings
    
    Returns:
        numpy array (N, embedding_dim)
    """
    # Original embeddings
    emb_orig = backbone.predict(images, batch_size=batch_size, verbose=0)
    
    # Flipped embeddings
    flipped = np.flip(images, axis=2)  # flip horizontally (width axis)
    emb_flip = backbone.predict(flipped, batch_size=batch_size, verbose=0)
    
    # L2 normalize individually
    if l2_norm:
        n1 = np.linalg.norm(emb_orig, axis=1, keepdims=True)
        n2 = np.linalg.norm(emb_flip, axis=1, keepdims=True)
        emb_orig = emb_orig / np.maximum(n1, 1e-12)
        emb_flip = emb_flip / np.maximum(n2, 1e-12)
    
    # Average and re-normalize
    emb_avg = emb_orig + emb_flip
    if l2_norm:
        n_avg = np.linalg.norm(emb_avg, axis=1, keepdims=True)
        emb_avg = emb_avg / np.maximum(n_avg, 1e-12)
    
    return emb_avg
