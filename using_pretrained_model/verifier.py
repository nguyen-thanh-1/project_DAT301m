"""
Face Verifier — using pretrained iResNet50 (ArcFace) backbone.

Self-contained module for one-shot face verification:
  1. Load pretrained iResNet50 backbone weights
  2. Preprocess: MTCNN detect → eye-alignment → expand bbox → resize(112×112) → norm [-1,1]
  3. Extract 512-D L2-normalized embedding
  4. Compare two faces via cosine similarity → same/different
  5. One-shot identification: query vs gallery of enrolled identities

Model:     iResNet50 Backbone (InsightFace/ArcFace paper, Deng et al. 2019)
Input:     112x112x3, float32, range [-1, 1]
Output:    512-D L2-normalized embedding on unit hypersphere
Weights:   models/face_recognition/iresnet/results/arcface/best_iresnet50_backbone.h5

Preprocessing matches dataset_final pipeline exactly:
  MTCNN detect → eye-keypoint alignment → expand bbox → resize 112 -> norm
"""

import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model

# Optional: MTCNN + OpenCV for face alignment (matches dataset_final pipeline)
try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

try:
    from mtcnn import MTCNN as _MTCNN
    _HAS_MTCNN = True
except ImportError:
    _HAS_MTCNN = False

# ─────────────────────────────────────────────────────────────────────
# iResNet50 Backbone (self-contained)
# ─────────────────────────────────────────────────────────────────────

def _ir_block(x, filters, strides, name):
    """IBasicBlock — BN→Conv(3×3)→BN→PReLU→Conv(3×3)→BN + Shortcut→Add (no ReLU after Add)."""
    shortcut = x
    if strides != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), strides=strides, use_bias=False,
                                 kernel_initializer="glorot_normal",
                                 name=f"{name}_shortcut_conv")(shortcut)
        shortcut = layers.BatchNormalization(momentum=0.9, epsilon=2e-5,
                                            name=f"{name}_shortcut_bn")(shortcut)

    x = layers.BatchNormalization(momentum=0.9, epsilon=2e-5, name=f"{name}_bn1")(x)
    x = layers.Conv2D(filters, (3, 3), strides=strides, padding='same', use_bias=False,
                      kernel_initializer="glorot_normal", name=f"{name}_conv1")(x)
    x = layers.BatchNormalization(momentum=0.9, epsilon=2e-5, name=f"{name}_bn2")(x)
    x = layers.PReLU(shared_axes=[1, 2], name=f"{name}_prelu")(x)
    x = layers.Conv2D(filters, (3, 3), strides=1, padding='same', use_bias=False,
                      kernel_initializer="glorot_normal", name=f"{name}_conv2")(x)
    x = layers.BatchNormalization(momentum=0.9, epsilon=2e-5, name=f"{name}_bn3")(x)
    x = layers.Add(name=f"{name}_add")([shortcut, x])
    return x


IRESNET50_STAGES = [3, 4, 14, 3]   # blocks per stage
IRESNET50_FILTERS = [64, 128, 256, 512]


def _build_backbone(input_shape=(112, 112, 3), embedding_dim=512, dropout_rate=0.4):
    """Build iResNet50 backbone (matches the trained ArcFace checkpoint)."""
    img_input = layers.Input(shape=input_shape, name="image_input")

    x = layers.Conv2D(64, (3, 3), strides=1, padding='same', use_bias=False,
                      kernel_initializer="glorot_normal", name="stem_conv")(img_input)
    x = layers.BatchNormalization(momentum=0.9, epsilon=2e-5, name="stem_bn")(x)
    x = layers.PReLU(shared_axes=[1, 2], name="stem_prelu")(x)

    for stage_idx, (num_blocks, filters) in enumerate(
        zip(IRESNET50_STAGES, IRESNET50_FILTERS)):
        for block_idx in range(num_blocks):
            strides = 2 if block_idx == 0 else 1
            x = _ir_block(x, filters=filters, strides=strides,
                          name=f"stage{stage_idx+1}_block{block_idx+1}")

    x = layers.BatchNormalization(momentum=0.9, epsilon=2e-5, name="bn_output")(x)
    if dropout_rate > 0.0:
        x = layers.Dropout(dropout_rate, name="dropout")(x)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(embedding_dim, use_bias=False,
                     kernel_initializer="glorot_normal", name="fc_embedding")(x)
    embeddings = layers.BatchNormalization(momentum=0.9, epsilon=2e-5,
                                          name="bn_embedding")(x)
    embeddings = layers.Lambda(lambda t: tf.nn.l2_normalize(t, axis=1),
                               name="l2_norm")(embeddings)
    return Model(inputs=img_input, outputs=embeddings, name="iResNet50_Backbone")


# ─────────────────────────────────────────────────────────────────────
# FaceVerifier
# ─────────────────────────────────────────────────────────────────────

class FaceVerifier:
    """Face verification using pretrained iResNet50 backbone.

    Usage:
        verifier = FaceVerifier("path/to/backbone.h5", threshold=0.35)
        same = verifier.verify("img1.jpg", "img2.jpg")
    """

    def __init__(self, weights_path, threshold=0.30, img_size=112, dropout_rate=0.4):
        """
        Args:
            weights_path: Path to pretrained iResNet50 backbone .h5 file
            threshold: Cosine similarity threshold (>= threshold → same person)
            img_size: Input image size (default 112, matches ArcFace training)
            dropout_rate: Dropout rate used in the backbone (0.4 per ArcFace paper)
        """
        self.img_size = img_size
        self.threshold = threshold

        self.backbone = _build_backbone(
            input_shape=(img_size, img_size, 3),
            embedding_dim=512,
            dropout_rate=dropout_rate,
        )
        self.backbone.load_weights(weights_path)
        print(f"[FaceVerifier] Loaded weights from: {weights_path}")
        print(f"[FaceVerifier] Threshold: {threshold:.4f}")

    # ── Preprocessing (MTCNN align, matches dataset_final pipeline) ──

    def _init_mtcnn(self):
        """Lazy-init MTCNN detector (heavy, only load once)."""
        if not hasattr(self, '_detector'):
            if _HAS_MTCNN:
                self._detector = _MTCNN()
            else:
                self._detector = None

    def _align_face(self, img_bgr, left_eye, right_eye):
        """Rotate face so eyes are horizontal (same as data_processing/preprocess.py)."""
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))
        center = (float((left_eye[0] + right_eye[0]) / 2.0),
                  float((left_eye[1] + right_eye[1]) / 2.0))
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned = cv2.warpAffine(img_bgr, M, (img_bgr.shape[1], img_bgr.shape[0]),
                                 flags=cv2.INTER_CUBIC)
        return aligned

    def preprocess(self, image, use_alignment=True):
        """Preprocess an image for the backbone.

        Supports:
          - str / Path -> load from disk
          - PIL.Image  -> convert to array
          - np.ndarray -> pass through (expects HxWx3, uint8 0-255)

        When use_alignment=True and MTCNN+cv2 available:
          MTCNN detect → eye-alignment → expand bbox → resize(112x112) → norm [-1,1]
        Falls back to: center-crop → resize(112x112) → norm [-1,1]
        """
        # Load image
        if isinstance(image, (str, Path)):
            im = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            im = image
        elif isinstance(image, np.ndarray):
            im = Image.fromarray(image.astype(np.uint8))
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        rgb_arr = np.asarray(im, dtype=np.uint8)

        # --- Try MTCNN face detection + alignment ---
        if use_alignment and _HAS_MTCNN and _HAS_CV2:
            self._init_mtcnn()
            if self._detector is not None:
                try:
                    bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
                    results = self._detector.detect_faces(rgb_arr)

                    if len(results) > 0:
                        # Largest face (by bbox area)
                        face = max(results, key=lambda b: b['box'][2] * b['box'][3])
                        x, y, w, h = face['box']
                        kp = face['keypoints']

                        # Eye alignment (same as dataset_final)
                        aligned = self._align_face(bgr,
                                                   kp['left_eye'], kp['right_eye'])

                        # Expand bbox: 30% width, 50% top (hair), 20% bottom (chin)
                        mx = int(w * 0.3)
                        my_top = int(h * 0.5)
                        my_bot = int(h * 0.2)

                        new_x = max(0, x - mx // 2)
                        new_y = max(0, y - my_top)
                        new_w = min(aligned.shape[1] - new_x, w + mx)
                        new_h = min(aligned.shape[0] - new_y, h + my_top + my_bot)

                        cropped = aligned[new_y:new_y+new_h, new_x:new_x+new_w]

                        if cropped.size > 0:
                            resized = cv2.resize(cropped, (self.img_size, self.img_size),
                                                 interpolation=cv2.INTER_CUBIC)
                            arr = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
                            arr = (arr - 127.5) / 128.0
                            return arr
                except Exception:
                    pass  # MTCNN failed, fall through to center-crop

        # --- Fallback: center-crop → resize → normalize ---
        w, h = im.size
        side = min(w, h)
        left = (w - side) // 2
        top  = (h - side) // 2
        im = im.crop((left, top, left + side, top + side))
        im = im.resize((self.img_size, self.img_size), resample=Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32)
        arr = (arr - 127.5) / 128.0
        return arr

    # ── Embedding ──────────────────────────────────────────────────

    def get_embedding(self, image):
        """Extract 512-D L2-normalized embedding from an image.

        Args:
            image: image path, PIL.Image, or np.ndarray

        Returns:
            np.ndarray of shape (512,) — L2-normalized embedding
        """
        arr = self.preprocess(image)
        batch = np.expand_dims(arr, axis=0)
        emb = self.backbone.predict(batch, verbose=0)
        return emb[0].astype(np.float32)

    def get_embeddings_batch(self, images, batch_size=128):
        """Extract embeddings for a list of images efficiently.

        Args:
            images: list of image paths or ndarrays
            batch_size: batch size for inference

        Returns:
            np.ndarray of shape (N, 512)
        """
        arrs = np.stack([self.preprocess(img) for img in images], axis=0)
        embs = self.backbone.predict(arrs, batch_size=batch_size, verbose=0)
        return embs.astype(np.float32)

    # ── Verification ───────────────────────────────────────────────

    def cosine_similarity(self, emb_a, emb_b):
        """Compute cosine similarity between two embeddings (both should be L2-normalized)."""
        return float(np.dot(emb_a, emb_b))

    def verify(self, image_a, image_b):
        """Check if two images show the same person.

        Returns:
            (is_same: bool, similarity: float)
        """
        emb_a = self.get_embedding(image_a)
        emb_b = self.get_embedding(image_b)
        sim = self.cosine_similarity(emb_a, emb_b)
        return sim >= self.threshold, sim

    def verify_embeddings(self, emb_a, emb_b):
        """Check if two pre-computed embeddings are the same person."""
        sim = self.cosine_similarity(emb_a, emb_b)
        return sim >= self.threshold, sim

    # ── One-shot Identification (gallery → query) ──────────────────

    def enroll(self, image_paths, identity_names):
        """Build a gallery of enrolled identities.

        Args:
            image_paths: list of image paths (one per identity for one-shot)
            identity_names: list of identity labels

        Returns:
            dict: {identity: embedding}
        """
        gallery = {}
        for path, name in zip(image_paths, identity_names):
            gallery[name] = self.get_embedding(path)
        return gallery

    def identify(self, query_image, gallery):
        """One-shot identification: find the closest identity in the gallery.

        Args:
            query_image: image path or array
            gallery: {identity: embedding} from enroll()

        Returns:
            (best_identity, best_similarity, is_matched)
        """
        query_emb = self.get_embedding(query_image)

        best_id = None
        best_sim = -1.0
        for ident, emb in gallery.items():
            sim = self.cosine_similarity(query_emb, emb)
            if sim > best_sim:
                best_sim = sim
                best_id = ident

        is_matched = best_sim >= self.threshold
        return best_id, best_sim, is_matched

    # ── Evaluate on a dataset ──────────────────────────────────────

    def evaluate_dataset(self, dataset_dir, num_pairs=5000, seed=42):
        """Evaluate verification performance on a dataset.

        Scans dataset_dir/<identity>/*.jpg structure, generates pairs,
        and computes EER, accuracy, AUC.

        Returns:
            dict with keys: eer, threshold_eer, accuracy, auc,
                            pos_mean, pos_std, neg_mean, neg_std,
                            pos_scores, neg_scores, y_true, all_scores
        """
        dataset_dir = Path(dataset_dir)
        rng = random.Random(seed)
        np.random.seed(seed)

        # Build identity → images mapping
        id_to_imgs = {}
        for cls_dir in sorted(dataset_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            imgs = [p for p in cls_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp'}]
            if len(imgs) >= 2:
                id_to_imgs[cls_dir.name] = imgs

        identities = sorted(id_to_imgs.keys())
        if len(identities) < 2:
            return {"error": "Need at least 2 identities with >=2 images"}

        # Sample image pool (max 30 per identity)
        for ident in identities:
            imgs = id_to_imgs[ident]
            if len(imgs) > 30:
                id_to_imgs[ident] = rng.sample(imgs, 30)

        # Extract all embeddings first
        print(f"[Eval] Extracting embeddings for {len(identities)} identities...")
        all_paths = []
        for ident in identities:
            all_paths.extend([str(p) for p in id_to_imgs[ident]])

        embs = self.get_embeddings_batch(all_paths, batch_size=128)
        path_to_emb = {p: e for p, e in zip(all_paths, embs)}
        print(f"[Eval] Done. Generating {num_pairs} positive + {num_pairs} negative pairs...")

        # Generate pairs
        pos_a, pos_b = [], []
        for _ in range(num_pairs):
            ident = rng.choice(identities)
            a, b = rng.sample(id_to_imgs[ident], 2)
            pos_a.append(str(a))
            pos_b.append(str(b))

        neg_a, neg_b = [], []
        for _ in range(num_pairs):
            id1, id2 = rng.sample(identities, 2)
            a = rng.choice(id_to_imgs[id1])
            b = rng.choice(id_to_imgs[id2])
            neg_a.append(str(a))
            neg_b.append(str(b))

        # Compute cosine scores
        pos_scores = np.array([self.cosine_similarity(path_to_emb[a], path_to_emb[b])
                               for a, b in zip(pos_a, pos_b)])
        neg_scores = np.array([self.cosine_similarity(path_to_emb[a], path_to_emb[b])
                               for a, b in zip(neg_a, neg_b)])

        scores = np.concatenate([pos_scores, neg_scores])
        y_true = np.array([1] * num_pairs + [0] * num_pairs, dtype=np.int32)

        # EER
        eer, eer_thr = self._compute_eer(y_true, scores)

        # Accuracy at best threshold
        if eer_thr is not None:
            preds = (scores >= eer_thr).astype(np.int32)
            acc = np.mean(preds == y_true)
        else:
            acc = 0.0

        # AUC
        auc = self._compute_auc(y_true, scores)

        return {
            "eer": eer,
            "threshold_eer": eer_thr,
            "accuracy": float(acc),
            "auc": auc,
            "pos_mean": float(np.mean(pos_scores)),
            "pos_std": float(np.std(pos_scores)),
            "neg_mean": float(np.mean(neg_scores)),
            "neg_std": float(np.std(neg_scores)),
            "pos_scores": pos_scores.tolist(),
            "neg_scores": neg_scores.tolist(),
            "y_true": y_true.tolist(),
            "all_scores": scores.tolist(),
        }

    def _compute_eer(self, y_true, scores):
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

    def _compute_auc(self, y_true, scores):
        order = np.argsort(scores)[::-1]
        y = y_true[order].astype(np.int32)
        n_pos = int(np.sum(y))
        n_neg = int(len(y) - n_pos)
        if n_pos == 0 or n_neg == 0:
            return 0.0
        tp = np.cumsum(y)
        fp = np.cumsum(1 - y)
        tpr = np.concatenate([[0.0], tp / max(n_pos, 1), [1.0]])
        fpr = np.concatenate([[0.0], fp / max(n_neg, 1), [1.0]])
        return float(np.trapz(tpr, fpr))

    # ── Fine-tune the backbone with contrastive loss ─────────────────

    def fine_tune(self, train_dir, output_path, epochs=15, batch_size=32,
                  pairs_per_epoch=20000, lr=0.0001, contrastive_margin=0.5,
                  freeze_ratio=0.7, val_dir=None, val_pairs=5000, seed=42):
        """Fine-tune the pretrained backbone with contrastive loss on pairs.

        Freezes early backbone layers (BN stable) and trains later layers +
        the final embedding projection to produce embeddings optimized for
        one-shot verification (compact intra-class, separated inter-class).

        Args:
            train_dir: Path to training images (identity subfolders, aligned 112x112)
            output_path: Where to save the fine-tuned backbone .h5
            epochs: Number of fine-tuning epochs (default 15)
            batch_size: Training batch size (pairs per step)
            pairs_per_epoch: Number of balanced pos/neg pairs per epoch
            lr: Learning rate (low to avoid catastrophic forgetting)
            contrastive_margin: Euclidean margin on unit sphere
            freeze_ratio: Fraction of backbone layers to freeze (0.7 = freeze 70%)
            val_dir: Optional validation directory for monitoring
            val_pairs: Number of validation pairs
            seed: Random seed
        """
        from tensorflow.keras import layers as kl
        from tensorflow.keras.models import Model as KModel
        from tensorflow.keras.optimizers import Adam

        print(f"\n{'='*60}")
        print(f"Fine-tuning Backbone with Contrastive Loss")
        print(f"{'='*60}")

        # --- Build training model (shared backbone + contrastive output) ---
        input_a = kl.Input(shape=(self.img_size, self.img_size, 3), name="ft_input_a")
        input_b = kl.Input(shape=(self.img_size, self.img_size, 3), name="ft_input_b")

        emb_a = self.backbone(input_a)
        emb_b = self.backbone(input_b)
        emb_concat = kl.Concatenate(name="ft_embeddings")([emb_a, emb_b])

        training_model = KModel(
            inputs=[input_a, input_b],
            outputs=emb_concat,
            name="FineTune_Siamese",
        )

        # --- Freeze early layers, train later layers ---
        layers = self.backbone.layers
        freeze_until = int(len(layers) * freeze_ratio)
        for layer in layers[:freeze_until]:
            layer.trainable = False
        for layer in layers[freeze_until:]:
            layer.trainable = True

        n_total = len(layers)
        n_trainable = sum(1 for l in layers if l.trainable)
        print(f"  Layers: total={n_total}, frozen={n_total - n_trainable}, "
              f"trainable={n_trainable}")

        # --- Loss: contrastive only (directly shapes embedding space) ---
        def _contrastive_loss(y_true, y_pred):
            half = tf.shape(y_pred)[-1] // 2
            ea = y_pred[:, :half]
            eb = y_pred[:, half:]
            d2 = tf.reduce_sum(tf.square(ea - eb), axis=1)
            d = tf.sqrt(d2 + 1e-7)
            yt = tf.cast(tf.squeeze(y_true), tf.float32)
            loss_pos = yt * d2
            loss_neg = (1.0 - yt) * tf.square(tf.maximum(contrastive_margin - d, 0.0))
            return loss_pos + loss_neg

        training_model.compile(optimizer=Adam(lr), loss=_contrastive_loss)

        # --- Data pipelines ---
        def _gen_pairs(id_imgs, n_pairs, rng):
            ids = sorted(id_imgs.keys())
            valid = [i for i in ids if len(id_imgs[i]) >= 2]
            pa, pb, labels = [], [], []
            for _ in range(n_pairs // 2):
                i = rng.choice(valid)
                a, b = rng.sample(list(id_imgs[i]), 2)
                pa.append(str(a)); pb.append(str(b)); labels.append(1)
            for _ in range(n_pairs - n_pairs // 2):
                i1, i2 = rng.sample(ids, 2)
                a = rng.choice(id_imgs[i1])
                b = rng.choice(id_imgs[i2])
                pa.append(str(a)); pb.append(str(b)); labels.append(0)
            combined = list(zip(pa, pb, labels))
            rng.shuffle(combined)
            pa, pb, labels = zip(*combined)
            return list(pa), list(pb), list(labels)

        def _load_img(path):
            img = Image.open(path).convert("RGB")
            arr = np.asarray(img, dtype=np.float32)
            arr = (arr - 127.5) / 128.0
            return arr

        def _augment(img):
            if random.random() < 0.5:
                img = np.fliplr(img)
            img = img + np.random.uniform(-0.2, 0.2)  # brightness
            img = np.clip(img, -1.0, 1.0)
            return img

        def _build_ds(id_imgs, n_pairs, bs, train, rng_seed):
            rng = random.Random(rng_seed + seed)
            def generator():
                while True:
                    pa, pb, lbl = _gen_pairs(id_imgs, n_pairs, rng)
                    for a, b, l in zip(pa, pb, lbl):
                        yield a, b, l

            def load_pair(pa, pb, lbl):
                ia = _load_img(pa)
                ib = _load_img(pb)
                if train:
                    ia = _augment(ia)
                    ib = _augment(ib)
                return (ia, ib), tf.cast(lbl, tf.float32)

            ds = tf.data.Dataset.from_generator(
                generator,
                output_signature=(
                    tf.TensorSpec(shape=(), dtype=tf.string),
                    tf.TensorSpec(shape=(), dtype=tf.string),
                    tf.TensorSpec(shape=(), dtype=tf.int32),
                ),
            )
            ds = ds.map(load_pair, num_parallel_calls=tf.data.AUTOTUNE)
            ds = ds.batch(bs, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
            return ds

        # Scan data
        train_ids = _scan_dir(train_dir)
        print(f"  Train: {len(train_ids)} identities, "
              f"{sum(len(v) for v in train_ids.values())} images")

        train_ds = _build_ds(train_ids, pairs_per_epoch, batch_size, train=True, rng_seed=0)
        steps = pairs_per_epoch // batch_size

        val_data = None
        val_steps = None
        if val_dir and os.path.isdir(val_dir):
            val_ids = _scan_dir(val_dir)
            val_ds = _build_ds(val_ids, val_pairs, batch_size, train=False, rng_seed=1)
            val_data = val_ds
            val_steps = val_pairs // batch_size
            print(f"  Val:   {len(val_ids)} identities")

        # --- Train ---
        print(f"  Pairs/epoch: {pairs_per_epoch}, Steps: {steps}")
        print(f"  LR: {lr}, Margin: {contrastive_margin}, Freeze ratio: {freeze_ratio}")
        print(f"{'='*60}\n")

        history = training_model.fit(
            train_ds,
            validation_data=val_data,
            epochs=epochs,
            steps_per_epoch=steps,
            validation_steps=val_steps,
            verbose=1,
        )

        # --- Save and reload ---
        self.backbone.save_weights(output_path)
        print(f"\n[Fine-Tune] Saved fine-tuned backbone to: {output_path}")

        # Reload the improved weights (already in the model)
        print("[Fine-Tune] Backbone weights updated in-memory.")

        return history


def _scan_dir(dataset_dir):
    """Helper: scan directory -> {identity: [Path, ...]}."""
    from collections import defaultdict
    from pathlib import Path
    dd = defaultdict(list)
    for cls_dir in sorted(Path(dataset_dir).iterdir()):
        if not cls_dir.is_dir():
            continue
        for f in cls_dir.iterdir():
            if f.is_file() and f.suffix.lower() in {'.jpg','.jpeg','.png','.bmp'}:
                dd[cls_dir.name].append(f)
    return {k: v for k, v in dd.items() if len(v) >= 2}
