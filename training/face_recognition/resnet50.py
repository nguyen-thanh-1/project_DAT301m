import math

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model


def configure_gpu_memory_growth():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if not gpus:
        return False
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        return True
    except RuntimeError:
        return False


class LearnablePositionalEncoding(layers.Layer):
    def build(self, input_shape):
        _, num_tokens, d_model = input_shape
        self.pos_emb = self.add_weight(
            "pos_emb",
            shape=(1, num_tokens, d_model),
            initializer="truncated_normal",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        return x + self.pos_emb


class TransformerEncoderBlock(layers.Layer):
    def __init__(self, d_model, num_heads, dff, drop_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.drop_rate = drop_rate

        self.ln1 = layers.LayerNormalization(epsilon=1e-6, name="ln1")
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=drop_rate,
            name="mha",
        )
        self.ln2 = layers.LayerNormalization(epsilon=1e-6, name="ln2")
        self.ffn_dense1 = layers.Dense(dff, activation="relu", kernel_initializer="he_normal", name="ffn_dense1")
        self.ffn_drop1 = layers.Dropout(drop_rate, name="ffn_drop1")
        self.ffn_dense2 = layers.Dense(d_model, kernel_initializer="he_normal", name="ffn_dense2")
        self.ffn_drop2 = layers.Dropout(drop_rate, name="ffn_drop2")

    def call(self, x, training=False):
        x_n = self.ln1(x)
        x = x + self.mha(x_n, x_n, training=training)
        x_n = self.ln2(x)
        ffn_out = self.ffn_dense1(x_n)
        ffn_out = self.ffn_drop1(ffn_out, training=training)
        ffn_out = self.ffn_dense2(ffn_out)
        ffn_out = self.ffn_drop2(ffn_out, training=training)
        return x + ffn_out


def identity_block(x, kernel_size, filters, stage, block):
    f1, f2, f3 = filters
    cb = f"res{stage}{block}_branch"
    bb = f"bn{stage}{block}_branch"
    shortcut = x

    x = layers.Conv2D(f1, (1, 1), kernel_initializer="he_normal", name=cb + "2a")(x)
    x = layers.BatchNormalization(name=bb + "2a")(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(f2, kernel_size, padding="same", kernel_initializer="he_normal", name=cb + "2b")(x)
    x = layers.BatchNormalization(name=bb + "2b")(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(f3, (1, 1), kernel_initializer="he_normal", name=cb + "2c")(x)
    x = layers.BatchNormalization(name=bb + "2c")(x)

    x = layers.Add()([x, shortcut])
    return layers.Activation("relu")(x)


def conv_block(x, kernel_size, filters, stage, block, strides=(2, 2)):
    f1, f2, f3 = filters
    cb = f"res{stage}{block}_branch"
    bb = f"bn{stage}{block}_branch"
    shortcut = x

    x = layers.Conv2D(f1, (1, 1), strides=strides, kernel_initializer="he_normal", name=cb + "2a")(x)
    x = layers.BatchNormalization(name=bb + "2a")(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(f2, kernel_size, padding="same", kernel_initializer="he_normal", name=cb + "2b")(x)
    x = layers.BatchNormalization(name=bb + "2b")(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(f3, (1, 1), kernel_initializer="he_normal", name=cb + "2c")(x)
    x = layers.BatchNormalization(name=bb + "2c")(x)

    shortcut = layers.Conv2D(f3, (1, 1), strides=strides, kernel_initializer="he_normal", name=cb + "1")(shortcut)
    shortcut = layers.BatchNormalization(name=bb + "1")(shortcut)

    x = layers.Add()([x, shortcut])
    return layers.Activation("relu")(x)


def ResNet50_Backbone(
    input_shape=(112, 112, 3),
    embedding_dim=512,
    use_transformer=True,
    num_transformer_layers=2,
    num_heads=8,
    transformer_dff=1024,
    drop_rate=0.1,
):
    img_input = layers.Input(shape=input_shape, name="image_input")

    x = layers.ZeroPadding2D((3, 3))(img_input)
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), kernel_initializer="he_normal", name="conv1")(x)
    x = layers.BatchNormalization(name="bn_conv1")(x)
    x = layers.Activation("relu")(x)
    x = layers.ZeroPadding2D((1, 1))(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block="a", strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block="b")
    x = identity_block(x, 3, [64, 64, 256], stage=2, block="c")

    x = conv_block(x, 3, [128, 128, 512], stage=3, block="a")
    x = identity_block(x, 3, [128, 128, 512], stage=3, block="b")
    x = identity_block(x, 3, [128, 128, 512], stage=3, block="c")
    x = identity_block(x, 3, [128, 128, 512], stage=3, block="d")

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block="a")
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block="b")
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block="c")
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block="d")
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block="e")
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block="f")

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block="a")
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block="b")
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block="c")

    if use_transformer:
        x = layers.Conv2D(embedding_dim, (1, 1), kernel_initializer="he_normal", name="trans_proj")(x)
        x = layers.BatchNormalization(name="bn_trans_proj")(x)
        x = layers.Activation("relu")(x)

        x = layers.Reshape((-1, embedding_dim), name="spatial_to_tokens")(x)
        x = LearnablePositionalEncoding(name="pos_encoding")(x)
        for i in range(num_transformer_layers):
            x = TransformerEncoderBlock(
                d_model=embedding_dim,
                num_heads=num_heads,
                dff=transformer_dff,
                drop_rate=drop_rate,
                name=f"transformer_{i}",
            )(x)
        x = layers.LayerNormalization(epsilon=1e-6, name="trans_final_norm")(x)
        embeddings = layers.GlobalAveragePooling1D(name="token_pool")(x)
    else:
        x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        embeddings = layers.Dense(embedding_dim, activation=None, kernel_initializer="he_normal", name="embedding_dense")(x)

    embeddings = layers.BatchNormalization(name="bn_embeddings")(embeddings)
    return Model(inputs=img_input, outputs=embeddings, name="ResNet50T_Backbone")


class ArcFace(layers.Layer):
    def __init__(self, num_classes, scale=64.0, margin=0.50, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        self._update_margin_constants(margin)

    def _update_margin_constants(self, margin):
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def set_margin(self, margin: float):
        self.margin = float(margin)
        self._update_margin_constants(self.margin)

    def build(self, input_shape):
        self.W = self.add_weight(
            "W",
            shape=(input_shape[0][-1], self.num_classes),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        embeddings, labels = inputs
        labels = tf.reshape(labels, [-1])

        norm_e = tf.nn.l2_normalize(embeddings, axis=1)
        norm_w = tf.nn.l2_normalize(self.W, axis=0)
        cos_t = tf.clip_by_value(tf.matmul(norm_e, norm_w), -1.0 + 1e-7, 1.0 - 1e-7)
        sin_t = tf.sqrt(1.0 - tf.square(cos_t) + 1e-7)
        cos_tm = cos_t * self.cos_m - sin_t * self.sin_m
        cos_tm = tf.where(cos_t > self.threshold, cos_tm, cos_t - self.mm)
        one_hot = tf.one_hot(tf.cast(labels, tf.int32), depth=self.num_classes)
        output = (one_hot * cos_tm) + ((1.0 - one_hot) * cos_t)
        return output * self.scale


def build_training_model(input_shape, num_classes, embedding_dim=512, use_transformer=True, **backbone_kwargs):
    backbone = ResNet50_Backbone(
        input_shape=input_shape,
        embedding_dim=embedding_dim,
        use_transformer=use_transformer,
        **backbone_kwargs,
    )
    img_in = layers.Input(shape=input_shape, name="image_input")
    lbl_in = layers.Input(shape=(), name="label_input")
    emb = backbone(img_in)
    arcface = ArcFace(num_classes=num_classes, margin=0.10, name="arcface_head")
    logits = arcface([emb, lbl_in])
    training_model = Model(inputs=[img_in, lbl_in], outputs=logits, name="ArcFace_Training")
    return training_model, backbone, arcface

