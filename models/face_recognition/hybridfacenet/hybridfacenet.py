import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model


BN_MOMENTUM = 0.9
BN_EPSILON = 2e-5


class ArcFaceLayer(layers.Layer):
    """ArcFace additive angular margin head."""

    def __init__(self, num_classes, margin=0.0, scale=64.0, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale

    def build(self, input_shape):
        embedding_shape, _ = input_shape
        self.W = self.add_weight(
            name="weights",
            shape=(int(embedding_shape[-1]), self.num_classes),
            initializer="glorot_normal",
            trainable=True,
        )
        super().build(input_shape)

    def _update_margin_constants(self, margin):
        self.margin = margin

    def call(self, inputs):
        embeddings, labels = inputs
        embeddings = tf.nn.l2_normalize(embeddings, axis=1)
        weights = tf.nn.l2_normalize(self.W, axis=0)
        cos_theta = tf.matmul(embeddings, weights)
        cos_theta = tf.clip_by_value(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
        theta = tf.acos(cos_theta)
        cos_theta_m = tf.cos(theta + self.margin)
        one_hot = tf.one_hot(tf.cast(labels, tf.int32), depth=self.num_classes)
        logits = tf.where(tf.cast(one_hot, tf.bool), cos_theta_m, cos_theta)
        return logits * self.scale

    def get_config(self):
        config = super().get_config()
        config.update(
            {"num_classes": self.num_classes, "margin": self.margin, "scale": self.scale}
        )
        return config


class DropPath(layers.Layer):
    """Stochastic depth per sample."""

    def __init__(self, drop_prob=0.0, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = float(drop_prob)

    def call(self, x, training=None):
        if not training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (tf.shape(x)[0],) + (1,) * (len(x.shape) - 1)
        random_tensor = keep_prob + tf.random.uniform(shape, dtype=x.dtype)
        random_tensor = tf.floor(random_tensor)
        return (x / keep_prob) * random_tensor

    def get_config(self):
        config = super().get_config()
        config.update({"drop_prob": self.drop_prob})
        return config


class ClassToken(layers.Layer):
    def __init__(self, embed_dim=512, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.cls = self.add_weight(
            name="cls",
            shape=(1, 1, self.embed_dim),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, tokens):
        batch_size = tf.shape(tokens)[0]
        cls = tf.tile(self.cls, [batch_size, 1, 1])
        return tf.concat([cls, tokens], axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim})
        return config


class PositionalEmbedding(layers.Layer):
    def __init__(self, num_tokens=65, embed_dim=512, **kwargs):
        super().__init__(**kwargs)
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.pos = self.add_weight(
            name="pos_embedding",
            shape=(1, self.num_tokens, self.embed_dim),
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        return x + self.pos

    def get_config(self):
        config = super().get_config()
        config.update({"num_tokens": self.num_tokens, "embed_dim": self.embed_dim})
        return config


class CouplingAdd(layers.Layer):
    """Add source into target with a switchable gate for two-phase training."""

    def build(self, input_shape):
        self.gate = self.add_weight(
            name="gate",
            shape=(),
            initializer="ones",
            trainable=False,
        )
        super().build(input_shape)

    def call(self, inputs):
        target, source = inputs
        return target + self.gate * source


class FusionLayer(layers.Layer):
    """Learnable weighted fusion of CNN and Transformer embeddings."""

    def build(self, input_shape):
        self.raw_alpha = self.add_weight(
            name="raw_alpha",
            shape=(),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        cnn_emb, trans_emb = inputs
        cnn_emb = tf.nn.l2_normalize(cnn_emb, axis=1)
        trans_emb = tf.nn.l2_normalize(trans_emb, axis=1)
        alpha = tf.clip_by_value(tf.sigmoid(self.raw_alpha), 0.1, 0.9)
        fused = alpha * cnn_emb + (1.0 - alpha) * trans_emb
        return tf.nn.l2_normalize(fused, axis=1)

    @property
    def alpha(self):
        return tf.clip_by_value(tf.sigmoid(self.raw_alpha), 0.1, 0.9)


def preprocess_input(x):
    return (tf.cast(x, tf.float32) - 127.5) / 128.0


def shared_stem(x, name="stem"):
    x = layers.Conv2D(
        64,
        3,
        strides=2,
        padding="same",
        use_bias=False,
        kernel_initializer="glorot_normal",
        name=f"{name}_conv",
    )(x)
    x = layers.BatchNormalization(momentum=BN_MOMENTUM, epsilon=BN_EPSILON, name=f"{name}_bn")(x)
    return layers.PReLU(shared_axes=[1, 2], name=f"{name}_prelu")(x)


def ibasic_block(x, filters, stride, name):
    shortcut = x
    if stride != 1 or int(shortcut.shape[-1]) != filters:
        shortcut = layers.Conv2D(
            filters,
            1,
            strides=stride,
            use_bias=False,
            kernel_initializer="glorot_normal",
            name=f"{name}_shortcut_conv",
        )(shortcut)
        shortcut = layers.BatchNormalization(
            momentum=BN_MOMENTUM, epsilon=BN_EPSILON, name=f"{name}_shortcut_bn"
        )(shortcut)

    x = layers.BatchNormalization(momentum=BN_MOMENTUM, epsilon=BN_EPSILON, name=f"{name}_bn1")(x)
    x = layers.Conv2D(
        filters,
        3,
        strides=stride,
        padding="same",
        use_bias=False,
        kernel_initializer="glorot_normal",
        name=f"{name}_conv1",
    )(x)
    x = layers.BatchNormalization(momentum=BN_MOMENTUM, epsilon=BN_EPSILON, name=f"{name}_bn2")(x)
    x = layers.PReLU(shared_axes=[1, 2], name=f"{name}_prelu")(x)
    x = layers.Conv2D(
        filters,
        3,
        strides=1,
        padding="same",
        use_bias=False,
        kernel_initializer="glorot_normal",
        name=f"{name}_conv2",
    )(x)
    x = layers.BatchNormalization(momentum=BN_MOMENTUM, epsilon=BN_EPSILON, name=f"{name}_bn3")(x)
    return layers.Add(name=f"{name}_add")([shortcut, x])


def cnn_stage(x, filters, blocks, first_stride, name):
    for block_idx in range(blocks):
        stride = first_stride if block_idx == 0 else 1
        x = ibasic_block(x, filters, stride, f"{name}_block{block_idx + 1}")
    return x


def patch_tokenize(stage1, embed_dim=512):
    x = layers.Conv2D(
        embed_dim,
        7,
        strides=7,
        padding="same",
        use_bias=False,
        kernel_initializer="glorot_normal",
        name="patch_tokenizer_conv",
    )(stage1)
    return layers.Reshape((64, embed_dim), name="patch_tokenizer_reshape")(x)


def fcu_down(x, embed_dim=512, name="fcu_down"):
    x = layers.Conv2D(
        embed_dim,
        1,
        use_bias=False,
        kernel_initializer="glorot_normal",
        name=f"{name}_conv",
    )(x)
    x = layers.BatchNormalization(momentum=BN_MOMENTUM, epsilon=BN_EPSILON, name=f"{name}_bn")(x)
    x = layers.Resizing(8, 8, interpolation="bilinear", name=f"{name}_spatial_align")(x)
    x = layers.Reshape((64, embed_dim), name=f"{name}_flatten")(x)
    return layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln")(x)


def add_fcu_to_tokens(tokens, fcu_tokens, name):
    cls_token = layers.Lambda(lambda t: t[:, :1, :], name=f"{name}_take_cls")(tokens)
    patch_tokens = layers.Lambda(lambda t: t[:, 1:, :], name=f"{name}_take_patches")(tokens)
    patch_tokens = CouplingAdd(name=f"{name}_coupling_add")([patch_tokens, fcu_tokens])
    return layers.Concatenate(axis=1, name=f"{name}_concat")([cls_token, patch_tokens])


def fcu_up(tokens, cnn_feature, out_channels, name):
    x = layers.Lambda(lambda t: t[:, 1:, :], name=f"{name}_drop_cls")(tokens)
    x = layers.Reshape((8, 8, 512), name=f"{name}_tokens_to_map")(x)
    x = layers.Conv2D(
        out_channels,
        1,
        use_bias=False,
        kernel_initializer="glorot_normal",
        name=f"{name}_conv",
    )(x)
    x = layers.BatchNormalization(momentum=BN_MOMENTUM, epsilon=BN_EPSILON, name=f"{name}_bn")(x)
    height, width = int(cnn_feature.shape[1]), int(cnn_feature.shape[2])
    x = layers.Resizing(height, width, interpolation="bilinear", name=f"{name}_upsample")(x)
    return CouplingAdd(name=f"{name}_coupling_add")([cnn_feature, x])


def transformer_block(x, embed_dim, num_heads, mlp_ratio, attn_dropout, mlp_dropout, drop_path, name):
    shortcut = x
    x = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln1")(x)
    x = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=embed_dim // num_heads,
        dropout=attn_dropout,
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
        name=f"{name}_mha",
    )(x, x)
    x = DropPath(drop_path, name=f"{name}_attn_drop_path")(x)
    x = layers.Add(name=f"{name}_attn_add")([shortcut, x])

    shortcut = x
    x = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln2")(x)
    x = layers.Dense(
        int(embed_dim * mlp_ratio),
        activation=tf.keras.activations.gelu,
        kernel_initializer="glorot_normal",
        name=f"{name}_mlp_fc1",
    )(x)
    x = layers.Dropout(mlp_dropout, name=f"{name}_mlp_dropout1")(x)
    x = layers.Dense(embed_dim, kernel_initializer="glorot_normal", name=f"{name}_mlp_fc2")(x)
    x = layers.Dropout(mlp_dropout, name=f"{name}_mlp_dropout2")(x)
    x = DropPath(drop_path, name=f"{name}_mlp_drop_path")(x)
    return layers.Add(name=f"{name}_mlp_add")([shortcut, x])


def cnn_embedding_head(x, embedding_dim=512, dropout_rate=0.3):
    x = layers.BatchNormalization(momentum=BN_MOMENTUM, epsilon=BN_EPSILON, name="cnn_output_bn")(x)
    x = layers.Dropout(dropout_rate, name="cnn_output_dropout")(x)
    x = layers.Flatten(name="cnn_flatten")(x)
    x = layers.Dense(
        embedding_dim,
        use_bias=False,
        kernel_initializer="glorot_normal",
        name="cnn_fc_embedding",
    )(x)
    return layers.BatchNormalization(
        momentum=BN_MOMENTUM, epsilon=BN_EPSILON, name="cnn_bn_embedding"
    )(x)


def HybridFaceNet_Backbone(
    input_shape=(112, 112, 3),
    embedding_dim=512,
    num_transformer_layers=6,
    num_heads=8,
    mlp_ratio=2.0,
    drop_path_rate=0.1,
    dropout_rate=0.3,
):
    """Conformer-style dual-branch CNN/Transformer face embedding backbone."""

    image_input = layers.Input(shape=input_shape, name="image_input")
    x = layers.Lambda(preprocess_input, name="preprocess")(image_input)
    stem = shared_stem(x)

    stage1 = cnn_stage(stem, 64, blocks=2, first_stride=1, name="cnn_stage1")

    tokens = patch_tokenize(stage1, embedding_dim)
    tokens = ClassToken(embedding_dim, name="cls_token")(tokens)
    tokens = PositionalEmbedding(65, embedding_dim, name="pos_embedding")(tokens)

    stage2 = cnn_stage(stage1, 128, blocks=3, first_stride=2, name="cnn_stage2")
    tokens = add_fcu_to_tokens(tokens, fcu_down(stage2, embedding_dim, "fcu_down_1"), "fcu_down_1")
    for i in range(2):
        rate = drop_path_rate * i / max(num_transformer_layers - 1, 1)
        tokens = transformer_block(
            tokens, embedding_dim, num_heads, mlp_ratio, 0.1, 0.1, rate, f"trans_block_{i + 1}"
        )
    stage2 = fcu_up(tokens, stage2, 128, "fcu_up_1")

    stage3 = cnn_stage(stage2, 256, blocks=4, first_stride=2, name="cnn_stage3")
    tokens = add_fcu_to_tokens(tokens, fcu_down(stage3, embedding_dim, "fcu_down_2"), "fcu_down_2")
    for i in range(2, 4):
        rate = drop_path_rate * i / max(num_transformer_layers - 1, 1)
        tokens = transformer_block(
            tokens, embedding_dim, num_heads, mlp_ratio, 0.1, 0.1, rate, f"trans_block_{i + 1}"
        )
    stage3 = fcu_up(tokens, stage3, 256, "fcu_up_2")

    stage4 = cnn_stage(stage3, 512, blocks=2, first_stride=2, name="cnn_stage4")
    tokens = add_fcu_to_tokens(tokens, fcu_down(stage4, embedding_dim, "fcu_down_3"), "fcu_down_3")
    for i in range(4, num_transformer_layers):
        rate = drop_path_rate * i / max(num_transformer_layers - 1, 1)
        tokens = transformer_block(
            tokens, embedding_dim, num_heads, mlp_ratio, 0.1, 0.1, rate, f"trans_block_{i + 1}"
        )
    stage4 = fcu_up(tokens, stage4, 512, "fcu_up_3")

    cnn_embedding = cnn_embedding_head(stage4, embedding_dim, dropout_rate)

    cls = layers.Lambda(lambda t: t[:, 0, :], name="trans_take_cls")(tokens)
    trans_embedding = layers.LayerNormalization(epsilon=1e-6, name="trans_output_ln")(cls)

    cnn_norm = layers.Lambda(lambda t: tf.nn.l2_normalize(t, axis=1), name="cnn_l2_norm")(cnn_embedding)
    trans_norm = layers.Lambda(lambda t: tf.nn.l2_normalize(t, axis=1), name="trans_l2_norm")(trans_embedding)
    final_embedding = FusionLayer(name="fusion")([cnn_embedding, trans_embedding])

    return Model(
        inputs=image_input,
        outputs=[final_embedding, cnn_norm, trans_norm],
        name="HybridFaceNet_Backbone",
    )


def build_training_model(
    input_shape=(112, 112, 3),
    num_classes=432,
    embedding_dim=512,
    margin=0.0,
    arcface_scale=64.0,
    dropout_rate=0.3,
):
    backbone = HybridFaceNet_Backbone(
        input_shape=input_shape,
        embedding_dim=embedding_dim,
        dropout_rate=dropout_rate,
    )
    face_input = layers.Input(shape=input_shape, name="face_input")
    label_input = layers.Input(shape=(), dtype=tf.int32, name="label_input")
    fused_embedding, _, _ = backbone(face_input)
    arcface = ArcFaceLayer(num_classes, margin=margin, scale=arcface_scale, name="arcface_head")
    logits = arcface([fused_embedding, label_input])
    training_model = Model([face_input, label_input], logits, name="HybridFaceNet_Training")
    return training_model, backbone, arcface


def build_inference_model(input_shape=(112, 112, 3), embedding_dim=512, dropout_rate=0.3):
    backbone = HybridFaceNet_Backbone(input_shape, embedding_dim, dropout_rate=dropout_rate)
    face_input = layers.Input(shape=input_shape, name="face_input")
    fused_embedding, _, _ = backbone(face_input)
    return Model(face_input, fused_embedding, name="HybridFaceNet_Inference")


def build_ablation_models(backbone):
    face_input = layers.Input(shape=backbone.input_shape[1:], name="face_input")
    fused_embedding, cnn_embedding, trans_embedding = backbone(face_input)
    return {
        "cnn_only": Model(face_input, cnn_embedding, name="HybridFaceNet_CNN_Only"),
        "trans_only": Model(face_input, trans_embedding, name="HybridFaceNet_Trans_Only"),
        "fused": Model(face_input, fused_embedding, name="HybridFaceNet_Fused"),
    }

