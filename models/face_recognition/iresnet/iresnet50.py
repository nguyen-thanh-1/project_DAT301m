import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import math

# =====================================================================
# Angular Margin Loss Variants (for Paper Comparison)
# =====================================================================
# ArcFace:    cos(theta + m)     — Additive Angular Margin  (Deng et al., 2019)
# CosFace:    cos(theta) - m     — Additive Cosine Margin   (Wang et al., 2018)
# SphereFace: cos(m * theta)     — Multiplicative Angular   (Liu et al., 2017)
# Softmax:    cos(theta)         — Normalized Softmax baseline (no margin)
# =====================================================================

class ArcFace(layers.Layer):
    """ArcFace: Additive Angular Margin Loss (Deng et al., 2019)
    
    Forward pass (per spec):
      1. W_norm = l2_normalize(W, axis=0)         [W shape: (512, num_classes)]
      2. cos_theta = matmul(embedding, W_norm)
      3. cos_theta = clip(cos_theta, -1+1e-7, 1-1e-7)
      4. theta = arccos(cos_theta)
      5. one_hot mask from labels
      6. theta_target = theta + m  (margin ONLY for correct class)
      7. cos_theta_m = cos(theta_target)
      8. logits = scale * scatter(cos_theta_m into correct class position)
      9. Loss = CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)
    
    Supports ElasticFace (Boutros et al. 2021):
      When elastic=True, margin is sampled from N(margin, 0.025) each forward pass
      during training, improving intra-class compactness on small datasets.
    
    margin is stored as tf.Variable for tf.train.Checkpoint persistence.
    """
    def __init__(self, num_classes, margin=0.50, scale=64.0, elastic=False, **kwargs):
        super(ArcFace, self).__init__(**kwargs)
        self.num_classes = num_classes
        self._init_margin = margin
        self.scale = scale
        self.elastic = elastic
        # tf.Variable so tf.train.Checkpoint can persist margin across resume
        self.margin_var = tf.Variable(margin, dtype=tf.float32,
                                      trainable=False, name='margin')
        
    def build(self, input_shape):
        embedding_shape, _ = input_shape
        self.W = self.add_weight(name='weights',
                                 shape=(embedding_shape[-1], self.num_classes),
                                 initializer='glorot_normal',
                                 trainable=True)
        super(ArcFace, self).build(input_shape)

    def _update_margin_constants(self, margin):
        """Called by ProgressiveMarginCallback to update margin dynamically."""
        self.margin_var.assign(margin)

    def call(self, inputs, training=None):
        embeddings, labels = inputs
        # Step 1: Normalize W and embeddings
        norm_w = tf.nn.l2_normalize(self.W, axis=0)
        norm_emb = tf.nn.l2_normalize(embeddings, axis=1)
        
        # Step 2-3: Cosine similarity with clipping
        cos_theta = tf.matmul(norm_emb, norm_w)
        cos_theta = tf.clip_by_value(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
        
        # Step 4-5: One-hot mask
        one_hot = tf.cast(tf.one_hot(tf.cast(labels, tf.int32), depth=self.num_classes), 
                          dtype=cos_theta.dtype)
        
        # Step 6: Compute margin (ElasticFace or fixed ArcFace)
        current_margin = self.margin_var
        if self.elastic and training:
            # ElasticFace: stochastic margin ~ N(margin, 0.025) with safety clipping
            current_margin = current_margin + tf.random.normal([], mean=0.0, stddev=0.025)
            current_margin = tf.clip_by_value(current_margin, 0.05, 0.7)
        
        # Step 7: cos(θ+m) = cos(θ)cos(m) - sin(θ)sin(m)
        # InsightFace-style numerically stable formula — avoids arccos→cos round-trip
        # and naturally handles the θ+m > π edge case
        cos_m = tf.cos(current_margin)
        sin_m = tf.sin(current_margin)
        sin_theta = tf.sqrt(tf.maximum(1.0 - cos_theta ** 2, 1e-12))
        cos_theta_m = cos_theta * cos_m - sin_theta * sin_m
        
        # Step 8: Scatter — correct class uses cos_theta_m, others keep cos_theta
        output = tf.where(one_hot == 1.0, cos_theta_m, cos_theta)
        output = output * self.scale
        return output

    def get_config(self):
        config = super(ArcFace, self).get_config()
        config.update({
            'num_classes': self.num_classes,
            'margin': float(self.margin_var.numpy()),
            'scale': self.scale,
            'elastic': self.elastic,
        })
        return config


class CosFace(layers.Layer):
    """CosFace: Large Margin Cosine Loss (Wang et al., 2018)
    L = -log( exp(s * (cos(theta_yi) - m)) / [exp(...) + sum_j exp(s * cos(theta_j))] )
    
    margin is stored as tf.Variable for tf.train.Checkpoint persistence.
    """
    def __init__(self, num_classes, margin=0.35, scale=64.0, elastic=False, **kwargs):
        super(CosFace, self).__init__(**kwargs)
        self.num_classes = num_classes
        self._init_margin = margin
        self.scale = scale
        self.elastic = elastic
        self.margin_var = tf.Variable(margin, dtype=tf.float32,
                                      trainable=False, name='margin')

    def build(self, input_shape):
        embedding_shape, _ = input_shape
        self.W = self.add_weight(name='weights',
                                 shape=(embedding_shape[-1], self.num_classes),
                                 initializer='glorot_normal',
                                 trainable=True)
        super(CosFace, self).build(input_shape)

    def _update_margin_constants(self, margin):
        """Compatibility with ProgressiveMarginCallback."""
        self.margin_var.assign(margin)

    def call(self, inputs, training=None):
        embeddings, labels = inputs
        norm_w = tf.nn.l2_normalize(self.W, axis=0)
        norm_emb = tf.nn.l2_normalize(embeddings, axis=1)

        cos_theta = tf.matmul(norm_emb, norm_w)
        cos_theta = tf.clip_by_value(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)

        # CosFace: subtract margin from cosine of target class
        one_hot = tf.cast(tf.one_hot(tf.cast(labels, tf.int32), depth=self.num_classes), 
                          dtype=cos_theta.dtype)
        
        current_margin = self.margin_var
        if self.elastic and training:
            current_margin = current_margin + tf.random.normal([], mean=0.0, stddev=0.025)
            current_margin = tf.clip_by_value(current_margin, 0.05, 0.7)
        
        output = cos_theta - one_hot * current_margin
        output = output * self.scale
        return output

    def get_config(self):
        config = super(CosFace, self).get_config()
        config.update({
            'num_classes': self.num_classes,
            'margin': float(self.margin_var.numpy()),
            'scale': self.scale,
            'elastic': self.elastic,
        })
        return config


class SphereFace(layers.Layer):
    """SphereFace: A-Softmax / Multiplicative Angular Margin (Liu et al., 2017)
    L = -log( exp(s * cos(m * theta_yi)) / [exp(...) + sum_j exp(s * cos(theta_j))] )
    
    margin is stored as tf.Variable for tf.train.Checkpoint persistence.
    """
    def __init__(self, num_classes, margin=4, scale=64.0, **kwargs):
        super(SphereFace, self).__init__(**kwargs)
        self.num_classes = num_classes
        self._init_margin = margin  # integer multiplier (typically 2 or 4)
        self.scale = scale
        self.margin_var = tf.Variable(float(margin), dtype=tf.float32,
                                      trainable=False, name='margin')

    def build(self, input_shape):
        embedding_shape, _ = input_shape
        self.W = self.add_weight(name='weights',
                                 shape=(embedding_shape[-1], self.num_classes),
                                 initializer='glorot_normal',
                                 trainable=True)
        super(SphereFace, self).build(input_shape)

    def _update_margin_constants(self, margin):
        """Compatibility with ProgressiveMarginCallback."""
        self.margin_var.assign(float(margin))

    def call(self, inputs, training=None):
        embeddings, labels = inputs
        norm_w = tf.nn.l2_normalize(self.W, axis=0)
        norm_emb = tf.nn.l2_normalize(embeddings, axis=1)

        cos_theta = tf.matmul(norm_emb, norm_w)
        cos_theta = tf.clip_by_value(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)

        # cos(m * theta) via cos(m * arccos(cos_theta))
        theta = tf.math.acos(cos_theta)
        cos_m_theta = tf.math.cos(self.margin_var * theta)

        one_hot = tf.cast(tf.one_hot(tf.cast(labels, tf.int32), depth=self.num_classes), 
                          dtype=cos_theta.dtype)

        # Monotonicity fix: only use cos_m_theta when theta < pi/m
        theta_threshold = math.pi / tf.maximum(self.margin_var, 1.0)
        keep_mask = tf.cast(theta < theta_threshold, dtype=cos_theta.dtype)
        cos_m_theta_safe = keep_mask * cos_m_theta + (1.0 - keep_mask) * cos_theta

        output = tf.where(one_hot == 1.0, cos_m_theta_safe, cos_theta)
        output = output * self.scale
        return output

    def get_config(self):
        config = super(SphereFace, self).get_config()
        config.update({
            'num_classes': self.num_classes,
            'margin': float(self.margin_var.numpy()),
            'scale': self.scale,
        })
        return config


class SoftmaxFace(layers.Layer):
    """Standard Normalized Softmax baseline (no margin).
    L = -log( exp(s * cos(theta_yi)) / sum_j exp(s * cos(theta_j)) )
    
    margin_var is a dummy tf.Variable (always 0.0) for API compatibility.
    """
    def __init__(self, num_classes, scale=64.0, **kwargs):
        super(SoftmaxFace, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.scale = scale
        self.margin_var = tf.Variable(0.0, dtype=tf.float32,
                                      trainable=False, name='margin')

    def build(self, input_shape):
        embedding_shape, _ = input_shape
        self.W = self.add_weight(name='weights',
                                 shape=(embedding_shape[-1], self.num_classes),
                                 initializer='glorot_normal',
                                 trainable=True)
        super(SoftmaxFace, self).build(input_shape)

    def _update_margin_constants(self, margin):
        """No-op for compatibility with ProgressiveMarginCallback."""
        pass

    def call(self, inputs, training=None):
        embeddings, labels = inputs
        norm_w = tf.nn.l2_normalize(self.W, axis=0)
        norm_emb = tf.nn.l2_normalize(embeddings, axis=1)
        cos_theta = tf.matmul(norm_emb, norm_w)
        cos_theta = tf.clip_by_value(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
        output = cos_theta * self.scale
        return output

    def get_config(self):
        config = super(SoftmaxFace, self).get_config()
        config.update({'num_classes': self.num_classes, 'scale': self.scale})
        return config


# Loss layer registry and default margins
LOSS_LAYERS = {
    "arcface": ArcFace,
    "cosface": CosFace,
    "sphereface": SphereFace,
    "softmax": SoftmaxFace,
}

DEFAULT_MARGINS = {
    "arcface": 0.50,
    "cosface": 0.35,
    "sphereface": 4,
    "softmax": 0.0,
}

# =====================================================================
# IBasicBlock (Improved Residual Block) — per InsightFace/ArcFace paper
# =====================================================================
#
# Structure (per spec):
#   - BatchNormalization(momentum=0.9, epsilon=2e-5)    [BN TRƯỚC conv]
#   - Conv2D(planes, 3x3, stride=STRIDE, padding='same', use_bias=False)
#   - BatchNormalization(momentum=0.9, epsilon=2e-5)
#   - PReLU(shared_axes=[1,2])
#   - Conv2D(planes, 3x3, stride=1, padding='same', use_bias=False)
#   - BatchNormalization(momentum=0.9, epsilon=2e-5)
#   - Skip connection: if stride>1 or in_channels≠out_channels →
#       Sequential[Conv2D(planes, 1x1, stride, use_bias=False), BN]
#   - Add([main_path, skip])
#   [NOTE: NO activation after Add — key difference from standard ResNet]
# =====================================================================
def ir_block(x, filters, strides, name):
    """IBasicBlock — BN → Conv(stride) → BN → PReLU → Conv(1) → BN + Shortcut → Add."""
    shortcut = x
    
    # Skip connection (downsample if needed)
    if strides != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), strides=strides, use_bias=False,
                                 kernel_initializer="glorot_normal", 
                                 name=f"{name}_shortcut_conv")(shortcut)
        shortcut = layers.BatchNormalization(momentum=0.9, epsilon=2e-5,
                                            name=f"{name}_shortcut_bn")(shortcut)

    # Main path
    x = layers.BatchNormalization(momentum=0.9, epsilon=2e-5, name=f"{name}_bn1")(x)
    x = layers.Conv2D(filters, (3, 3), strides=strides, padding='same', use_bias=False,
                      kernel_initializer="glorot_normal", name=f"{name}_conv1")(x)
    x = layers.BatchNormalization(momentum=0.9, epsilon=2e-5, name=f"{name}_bn2")(x)
    x = layers.PReLU(shared_axes=[1, 2], name=f"{name}_prelu")(x)
    x = layers.Conv2D(filters, (3, 3), strides=1, padding='same', use_bias=False,
                      kernel_initializer="glorot_normal", name=f"{name}_conv2")(x)
    x = layers.BatchNormalization(momentum=0.9, epsilon=2e-5, name=f"{name}_bn3")(x)

    # Add (NO activation after Add)
    x = layers.Add(name=f"{name}_add")([shortcut, x])
    return x

# =====================================================================
# iResNet50 Backbone — [3, 4, 14, 3] blocks
# =====================================================================
IRESNET50_STAGES = [3, 4, 14, 3]   # ~44M params

def iResNet_Backbone(input_shape=(112, 112, 3), embedding_dim=512, dropout_rate=0.5,
                     normalize_embeddings=True, variant="iresnet50"):
    """iResNet50 Backbone per InsightFace/ArcFace paper (Deng et al., 2019).
    
    Architecture:
      CONV1:  Conv2D(64, 3x3, stride=1) → BN(0.9, 2e-5) → PReLU
      Stage1: 3  × IBasicBlock(64,  stride_first=2) → 56×56×64
      Stage2: 4  × IBasicBlock(128, stride_first=2) → 28×28×128
      Stage3: 14 × IBasicBlock(256, stride_first=2) → 14×14×256
      Stage4: 3  × IBasicBlock(512, stride_first=2) → 7×7×512
      HEAD:   BN → Dropout(0.4) → Flatten → Dense(512) → BN → L2Norm
    
    Args:
        dropout_rate: Dropout rate (0.4 = drop 40%, keep 60%)
        normalize_embeddings: If True, L2-normalize output to unit hypersphere
        variant: Only 'iresnet50' is supported
    """
    filters_per_stage = [64, 128, 256, 512]
    
    img_input = layers.Input(shape=input_shape, name="image_input")

    # CONV1 — ENTRY BLOCK: Conv2D(64, 3x3, stride=1) → BN → PReLU
    x = layers.Conv2D(64, (3, 3), strides=1, padding='same', use_bias=False,
                      kernel_initializer="glorot_normal", name="stem_conv")(img_input)
    x = layers.BatchNormalization(momentum=0.9, epsilon=2e-5, name="stem_bn")(x)
    x = layers.PReLU(shared_axes=[1, 2], name="stem_prelu")(x)

    # 4 STAGES: [3, 4, 14, 3] IBasicBlocks
    for stage_idx, (num_blocks, filters) in enumerate(zip(IRESNET50_STAGES, filters_per_stage)):
        for block_idx in range(num_blocks):
            strides = 2 if block_idx == 0 else 1
            x = ir_block(x, filters=filters, strides=strides,
                         name=f"stage{stage_idx+1}_block{block_idx+1}")

    # OUTPUT HEAD: BN → Dropout → Flatten → Dense(512) → BN
    x = layers.BatchNormalization(momentum=0.9, epsilon=2e-5, name="bn_output")(x)
    
    if dropout_rate > 0.0:
        x = layers.Dropout(dropout_rate, name="dropout")(x)
    
    x = layers.Flatten(name="flatten")(x)  # 7*7*512 = 25088
    x = layers.Dense(embedding_dim, use_bias=False,
                     kernel_initializer="glorot_normal", name="fc_embedding")(x)
    embeddings = layers.BatchNormalization(momentum=0.9, epsilon=2e-5,
                                          name="bn_embedding")(x)
    
    # L2 NORMALIZATION — embedding on unit hypersphere
    if normalize_embeddings:
        embeddings = layers.Lambda(lambda t: tf.nn.l2_normalize(t, axis=1), 
                                   name="l2_norm")(embeddings)
        
    return Model(inputs=img_input, outputs=embeddings, name="IRESNET50_Backbone")

# =====================================================================
# Build Training Model — Supports multiple loss types
# =====================================================================
def build_training_model(input_shape, num_classes, embedding_dim=512, dropout_rate=0.5,
                         arcface_scale=64.0, loss_type="arcface", margin=None,
                         elastic=False):
    """Builds the full training model: backbone + loss head.
    
    MODEL A (Training):  Input([image, label]) → backbone → LossHead → logits
    MODEL B (Inference):  backbone only → embeddings (shared weights with Model A)
    
    Args:
        loss_type: 'arcface' | 'cosface' | 'sphereface' | 'softmax'
        margin: Override default margin. If None, uses DEFAULT_MARGINS[loss_type].
        arcface_scale: Scale factor (s=64.0 per ArcFace paper)
        dropout_rate: Dropout rate (0.4 per spec)
    """
    backbone = iResNet_Backbone(
        input_shape=input_shape, 
        embedding_dim=embedding_dim, 
        dropout_rate=dropout_rate,
        normalize_embeddings=False,  # Loss layer normalizes internally
    )
    
    img_in = layers.Input(shape=input_shape, name="image_input")
    lbl_in = layers.Input(shape=(), name="label_input")
    emb = backbone(img_in)
    
    if margin is None:
        margin = DEFAULT_MARGINS.get(loss_type, 0.5)
    
    LossClass = LOSS_LAYERS.get(loss_type, ArcFace)
    
    if loss_type == "softmax":
        loss_layer = LossClass(num_classes=num_classes, scale=arcface_scale,
                               name=f"{loss_type}_head")
    elif loss_type == "sphereface":
        # SphereFace uses integer margin multiplier, no elastic support
        loss_layer = LossClass(num_classes=num_classes, margin=margin if margin else 0.0,
                               scale=arcface_scale, name=f"{loss_type}_head")
    else:
        # ArcFace/CosFace: Start with margin=0, will be increased by ProgressiveMarginCallback
        loss_layer = LossClass(num_classes=num_classes, margin=0.0,
                               scale=arcface_scale, elastic=elastic,
                               name=f"{loss_type}_head")
    
    logits = loss_layer([emb, lbl_in])
    
    training_model = Model(
        inputs=[img_in, lbl_in], outputs=logits,
        name=f"{loss_type.capitalize()}_Training_iResNet50"
    )
    return training_model, backbone, loss_layer
