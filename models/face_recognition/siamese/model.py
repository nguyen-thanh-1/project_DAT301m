"""
Siamese iResNet50 — One-Shot Face Verification Model.

Architecture:
  Shared iResNet50 backbone (44M params)
  → L2-normalized embeddings (512D)
  → |emb_a - emb_b|
  → Comparison Head: Dense(128)→BN→PReLU→Dropout→Dense(1)→Sigmoid

Outputs:
  prob:       binary probability (same/different)
  embeddings: (emb_a, emb_b) for optional contrastive loss
"""

import math
import tensorflow as tf
from tensorflow.keras import layers, Model, Input


# =====================================================================
# IBasicBlock — Improved Residual Block (per InsightFace/ArcFace paper)
# =====================================================================
def ir_block(x, filters, strides, stage, block):
    """IBasicBlock: BN→Conv(stride)→BN→PReLU→Conv(1)→BN + Shortcut→Add."""
    name = f"stage{stage}_block{block}"

    shortcut = x
    if strides != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), strides=strides, use_bias=False,
                                 kernel_initializer="glorot_normal",
                                 name=f"{name}_sc_conv")(shortcut)
        shortcut = layers.BatchNormalization(momentum=0.9, epsilon=2e-5,
                                             name=f"{name}_sc_bn")(shortcut)

    x = layers.BatchNormalization(momentum=0.9, epsilon=2e-5, name=f"{name}_bn1")(x)
    x = layers.Conv2D(filters, (3, 3), strides=strides, padding="same", use_bias=False,
                      kernel_initializer="glorot_normal", name=f"{name}_conv1")(x)
    x = layers.BatchNormalization(momentum=0.9, epsilon=2e-5, name=f"{name}_bn2")(x)
    x = layers.PReLU(shared_axes=[1, 2], name=f"{name}_prelu")(x)
    x = layers.Conv2D(filters, (3, 3), strides=1, padding="same", use_bias=False,
                      kernel_initializer="glorot_normal", name=f"{name}_conv2")(x)
    x = layers.BatchNormalization(momentum=0.9, epsilon=2e-5, name=f"{name}_bn3")(x)
    x = layers.Add(name=f"{name}_add")([shortcut, x])
    return x


IRESNET50_STAGES = [3, 4, 14, 3]
FILTERS_PER_STAGE = [64, 128, 256, 512]


# =====================================================================
# iResNet50 Backbone (shared-weight, embedding extractor)
# =====================================================================
def build_backbone(input_shape=(112, 112, 3), embedding_dim=512,
                   dropout_rate=0.4, name="Backbone"):
    """iResNet50 producing L2-normalized 512D embeddings."""
    img_input = Input(shape=input_shape, name=f"{name}_input")

    x = layers.Conv2D(64, (3, 3), strides=1, padding="same", use_bias=False,
                      kernel_initializer="glorot_normal", name="stem_conv")(img_input)
    x = layers.BatchNormalization(momentum=0.9, epsilon=2e-5, name="stem_bn")(x)
    x = layers.PReLU(shared_axes=[1, 2], name="stem_prelu")(x)

    for stage_idx, (num_blocks, filters) in enumerate(zip(IRESNET50_STAGES, FILTERS_PER_STAGE)):
        for block_idx in range(num_blocks):
            strides = 2 if block_idx == 0 else 1
            x = ir_block(x, filters, strides, stage=stage_idx + 1, block=block_idx + 1)

    x = layers.BatchNormalization(momentum=0.9, epsilon=2e-5, name="bn_output")(x)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name="dropout")(x)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(embedding_dim, use_bias=False,
                     kernel_initializer="glorot_normal", name="fc_embedding")(x)
    x = layers.BatchNormalization(momentum=0.9, epsilon=2e-5, name="bn_embedding")(x)
    embedding = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1),
                              name="l2_norm")(x)
    return Model(inputs=img_input, outputs=embedding, name=name)


# =====================================================================
# Siamese Model — Full One-Shot Verification Pipeline
# =====================================================================
def build_siamese_model(input_shape=(112, 112, 3), embedding_dim=512,
                        dropout_rate=0.4, include_contrastive=True):
    """Build complete Siamese One-Shot Verification model."""
    backbone = build_backbone(input_shape, embedding_dim, dropout_rate, name="Siamese_Backbone")

    img_a = Input(shape=input_shape, name="img_a")
    img_b = Input(shape=input_shape, name="img_b")

    emb_a = backbone(img_a)
    emb_b = backbone(img_b)

    diff = layers.Subtract(name="diff")([emb_a, emb_b])
    diff_abs = layers.Lambda(lambda t: tf.abs(t), name="diff_abs")(diff)

    x = layers.Dense(128, kernel_initializer="glorot_normal", name="comp_dense1")(diff_abs)
    x = layers.BatchNormalization(momentum=0.9, epsilon=2e-5, name="comp_bn1")(x)
    x = layers.PReLU(shared_axes=[1], name="comp_prelu")(x)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate * 0.5, name="comp_dropout")(x)
    prob = layers.Dense(1, activation="sigmoid", name="prob")(x)

    if include_contrastive:
        embeddings_concat = layers.Concatenate(name="embeddings")([emb_a, emb_b])
        siamese_model = Model(inputs=[img_a, img_b], outputs=[prob, embeddings_concat],
                              name="Siamese_iResNet50")
    else:
        siamese_model = Model(inputs=[img_a, img_b], outputs=prob, name="Siamese_iResNet50")

    return siamese_model, backbone


# =====================================================================
# Progressive Margin — dynamic tf.Variable changed by callback
# =====================================================================
_margin_var = None


def get_margin_variable(initial=0.4):
    """Get or create the global margin tf.Variable used by contrastive_loss."""
    global _margin_var
    if _margin_var is None:
        _margin_var = tf.Variable(initial, dtype=tf.float32, trainable=False, name="contrastive_margin")
    return _margin_var


class ProgressiveMarginCallback(tf.keras.callbacks.Callback):
    """Gradually decreases contrastive margin over epochs.

    Schedule: margin_start → margin_end via cosine curve.
    Lower margin = harder push for negatives (see contrastive loss).
    """

    def __init__(self, margin_start=0.4, margin_end=0.0, total_epochs=80):
        super().__init__()
        self.margin_start = margin_start
        self.margin_end = margin_end
        self.total_epochs = total_epochs
        self.margin_var = get_margin_variable(margin_start)

    def on_epoch_begin(self, epoch, logs=None):
        progress = min(epoch / max(self.total_epochs - 1, 1), 1.0)
        new_m = self.margin_end + 0.5 * (self.margin_start - self.margin_end) * \
                (1.0 + math.cos(math.pi * progress))
        self.margin_var.assign(new_m)
        if epoch % 5 == 0:
            print(f"\n[ProgressiveMargin] Epoch {epoch+1}: margin = {new_m:.4f}")


# =====================================================================
# Contrastive Loss (cosine-based, reads margin from global Variable)
# =====================================================================
def make_contrastive_loss():
    """Cosine-based contrastive loss using dynamic margin.

    L_same  = (1 - cos_sim)^2   → push toward cosine +1
    L_diff  = max(cos_sim - margin, 0)^2 → push below margin

    Margin is read from the global margin_variable, allowing
    ProgressiveMarginCallback to adjust it during training.
    """

    @tf.function
    def contrastive_loss(y_true, y_pred):
        """y_pred = concatenated [emb_a, emb_b] (L2-normalized)."""
        y_pred = tf.cast(y_pred, tf.float32)
        half = tf.shape(y_pred)[-1] // 2
        emb_a = y_pred[:, :half]
        emb_b = y_pred[:, half:]

        cos_sim = tf.reduce_sum(emb_a * emb_b, axis=1)  # [-1, 1]
        y = tf.reshape(tf.cast(y_true, tf.float32), [-1])

        loss_same = tf.square(1.0 - cos_sim)
        loss_diff = tf.square(tf.maximum(cos_sim - get_margin_variable(), 0.0))

        return tf.reduce_mean(y * loss_same + (1.0 - y) * loss_diff) * 0.5

    return contrastive_loss


# =====================================================================
# Triplet Model — Embedding extractor for Batch-Hard Triplet Loss
# =====================================================================

def build_triplet_model(input_shape=(112, 112, 3), embedding_dim=512,
                        dropout_rate=0.4):
    """Build model that outputs L2-normalized embeddings for triplet training.

    Returns (model, backbone) where:
      - model:   Input(image) → Output(512-D L2-normalized embedding)
      - backbone: Same as model (shared weight extractor for inference)

    During training, the loss receives a flat batch of PK-sampled images
    (P identities × K images each) and their identity labels. It performs
    batch-hard online triplet mining internally (see batch_hard_triplet_loss).
    """
    backbone = build_backbone(input_shape, embedding_dim, dropout_rate,
                              name="Triplet_Backbone")
    img_input = Input(shape=input_shape, name="image_input")
    embedding = backbone(img_input)
    model = Model(inputs=img_input, outputs=embedding, name="Triplet_iResNet50")
    return model, backbone


# =====================================================================
# Batch-Hard Triplet Loss (FaceNet-style online mining)
# =====================================================================

def batch_hard_triplet_loss(margin=0.3, distance_metric="cosine"):
    """Batch-hard triplet loss with online mining.

    FaceNet-style (Schroff et al., 2015):
      For each anchor, select the HARDEST positive (farthest same-class)
      and the HARDEST negative (closest different-class).

    Input layout — expects a flat batch of PK-sampled images:
      y_pred: (batch_size, embedding_dim) — L2-normalized embeddings
      y_true: (batch_size,) — integer identity labels (0..P-1)

    Each batch must have P identities with K images each:
      indices [0..K-1]        → identity 0
      indices [K..2K-1]       → identity 1
      ...
      indices [(P-1)*K..P*K-1] → identity P-1

    The margin parameter controls how far apart positive and negative
    pairs should be in cosine distance [0, 2].  Typical values: 0.2-0.4.
    """
    @tf.function
    def loss_fn(y_true, y_pred):
        y_pred = tf.cast(y_pred, tf.float32)

        # Cosine distance matrix: D[i,j] = 1 - cos(e_i, e_j) ∈ [0, 2]
        # Since embeddings are L2-normalized, dot product = cosine similarity
        sim_matrix = tf.matmul(y_pred, y_pred, transpose_b=True)
        dist_matrix = 1.0 - sim_matrix  # cosine distance

        labels = tf.cast(tf.squeeze(y_true), tf.int32)
        labels_eq = tf.equal(labels[:, tf.newaxis], labels[tf.newaxis, :])
        labels_diff = tf.logical_not(labels_eq)

        # Mask out self-comparisons from positive mask
        batch_size = tf.cast(tf.shape(y_pred)[0], tf.int32)
        diag_mask = 1.0 - tf.eye(batch_size, dtype=tf.float32)
        pos_mask = tf.cast(labels_eq, tf.float32) * diag_mask
        neg_mask = tf.cast(labels_diff, tf.float32)

        neg_inf = tf.constant(-1e9, dtype=tf.float32)
        pos_inf = tf.constant(1e9, dtype=tf.float32)

        # Hardest positive: maximum distance among same-class samples
        # (the one the model finds hardest to recognize as same person)
        hardest_pos_dist = tf.reduce_max(
            dist_matrix * pos_mask + (1.0 - pos_mask) * neg_inf, axis=1,
        )

        # Hardest negative: minimum distance among different-class samples
        # (the one the model finds hardest to distinguish)
        hardest_neg_dist = tf.reduce_min(
            dist_matrix * neg_mask + (1.0 - neg_mask) * pos_inf, axis=1,
        )

        # Triplet loss: max(0, d_pos - d_neg + margin)
        margin_f = tf.cast(margin, tf.float32)
        loss = tf.maximum(hardest_pos_dist - hardest_neg_dist + margin_f, 0.0)
        return tf.reduce_mean(loss)

    return loss_fn


# =====================================================================
# Triplet Accuracy / Fraction of Active Triplets
# =====================================================================

def triplet_active_fraction(margin=0.3):
    """Metric: fraction of anchors with non-zero triplet loss.

    Lower = better (fewer triplets violating the margin).
    """
    @tf.function
    def metric_fn(y_true, y_pred):
        y_pred = tf.cast(y_pred, tf.float32)
        sim_matrix = tf.matmul(y_pred, y_pred, transpose_b=True)
        dist_matrix = 1.0 - sim_matrix

        labels = tf.cast(tf.squeeze(y_true), tf.int32)
        labels_eq = tf.equal(labels[:, tf.newaxis], labels[tf.newaxis, :])
        labels_diff = tf.logical_not(labels_eq)

        batch_size = tf.shape(y_pred)[0]
        diag_mask = 1.0 - tf.eye(batch_size, dtype=tf.float32)
        pos_mask = tf.cast(labels_eq, tf.float32) * diag_mask
        neg_mask = tf.cast(labels_diff, tf.float32)

        neg_inf = tf.constant(-1e9, dtype=tf.float32)
        pos_inf = tf.constant(1e9, dtype=tf.float32)

        hardest_pos = tf.reduce_max(
            dist_matrix * pos_mask + (1.0 - pos_mask) * neg_inf, axis=1,
        )
        hardest_neg = tf.reduce_min(
            dist_matrix * neg_mask + (1.0 - neg_mask) * pos_inf, axis=1,
        )

        margin_f = tf.cast(margin, tf.float32)
        violations = tf.cast(
            hardest_pos - hardest_neg + margin_f > 0.0, tf.float32,
        )
        return tf.reduce_mean(violations)

    return metric_fn
