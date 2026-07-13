import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import math

from models.face_recognition.iresnet.iresnet50 import LOSS_LAYERS, DEFAULT_MARGINS

# =====================================================================
# MobileFaceNet Components
# =====================================================================

def conv_block(x, filters, kernel_size, strides, padding='same', name=None):
    """Standard Conv + BN + PReLU block."""
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding,
                      use_bias=False, name=f"{name}_conv")(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.PReLU(shared_axes=[1, 2], name=f"{name}_prelu")(x)
    return x

def dw_conv_block(x, kernel_size, strides, padding='same', name=None):
    """Depthwise Conv + BN + PReLU block."""
    x = layers.DepthwiseConv2D(kernel_size, strides=strides, padding=padding,
                               use_bias=False, name=f"{name}_dwconv")(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.PReLU(shared_axes=[1, 2], name=f"{name}_prelu")(x)
    return x

def bottleneck(x, filters, expansion, stride, name=None):
    """Inverted Residual Bottleneck for MobileFaceNet."""
    in_channels = x.shape[-1]
    expanded_channels = in_channels * expansion
    shortcut = x

    # 1. Expansion (1x1 Conv)
    if expansion != 1:
        x = conv_block(x, expanded_channels, (1, 1), strides=1, name=f"{name}_expand")
    else:
        x = x

    # 2. Depthwise Convolution
    x = layers.DepthwiseConv2D((3, 3), strides=stride, padding='same',
                               use_bias=False, name=f"{name}_depthwise")(x)
    x = layers.BatchNormalization(name=f"{name}_dw_bn")(x)
    x = layers.PReLU(shared_axes=[1, 2], name=f"{name}_dw_prelu")(x)

    # 3. Projection (1x1 Conv) - Linear activation
    x = layers.Conv2D(filters, (1, 1), strides=1, padding='same',
                      use_bias=False, name=f"{name}_project")(x)
    x = layers.BatchNormalization(name=f"{name}_proj_bn")(x)

    # Shortcut connection
    if stride == 1 and in_channels == filters:
        x = layers.Add(name=f"{name}_add")([shortcut, x])
    
    return x


# =====================================================================
# MobileFaceNet Backbone
# =====================================================================

def MobileFaceNet_Backbone(input_shape=(112, 112, 3), embedding_dim=256,
                           normalize_embeddings=True, name="MobileFaceNet"):
    """MobileFaceNet backbone implementation."""
    img_input = layers.Input(shape=input_shape, name="image_input")

    # Stem block
    x = conv_block(img_input, filters=64, kernel_size=(3, 3), strides=2, name="stem") # 56x56
    x = dw_conv_block(x, kernel_size=(3, 3), strides=1, name="stem_dw") # 56x56

    # Bottleneck configurations: [t (expansion), c (channels), n (num repeats), s (stride)]
    bottleneck_settings = [
        [2, 64, 5, 2],  # downsamples to 28x28
        [4, 128, 1, 2], # downsamples to 14x14
        [2, 128, 6, 1],
        [4, 256, 1, 2], # downsamples to 7x7
        [2, 256, 2, 1]
    ]

    block_id = 1
    for t, c, n, s in bottleneck_settings:
        for i in range(n):
            stride = s if i == 0 else 1
            x = bottleneck(x, filters=c, expansion=t, stride=stride, name=f"bottleneck_{block_id}")
            block_id += 1

    # Linear 1x1 Conv before GDC
    x = conv_block(x, filters=512, kernel_size=(1, 1), strides=1, name="conv_before_gdc") # 7x7x512

    # Global Depthwise Convolution (GDC)
    x = layers.DepthwiseConv2D((7, 7), strides=1, padding='valid', use_bias=False, name="gdc")(x) # 1x1x512
    
    # Flatten before BN to avoid cuDNN 4D spatial BN instability on 1x1 features
    x = layers.Flatten(name="flatten")(x)
    x = layers.BatchNormalization(name="gdc_bn")(x) # No activation

    # Embedding Head (Dense 256 + BN + L2)
    x = layers.Dense(embedding_dim, use_bias=False, kernel_initializer="glorot_normal", name="fc_embedding")(x)
    embeddings = layers.BatchNormalization(scale=False, momentum=0.9, epsilon=2e-5, name="bn_embedding")(x)

    if normalize_embeddings:
        embeddings = layers.Lambda(lambda t: tf.nn.l2_normalize(t, axis=1), name="l2_norm")(embeddings)

    return Model(inputs=img_input, outputs=embeddings, name=name)


# =====================================================================
# Build Training Model
# =====================================================================

def build_mobilefacenet_training_model(input_shape, num_classes, embedding_dim=256,
                                       arcface_scale=32.0, loss_type="arcface",
                                       margin=None):
    """Builds the full model for training MobileFaceNet with configurable margin loss."""
    backbone = MobileFaceNet_Backbone(
        input_shape=input_shape,
        embedding_dim=embedding_dim,
        normalize_embeddings=False,  # Loss layer normalizes internally
    )

    img_in = layers.Input(shape=input_shape, name="image_input")
    lbl_in = layers.Input(shape=(), name="label_input")
    emb = backbone(img_in)

    if margin is None:
        margin = DEFAULT_MARGINS.get(loss_type, 0.5)

    LossClass = LOSS_LAYERS.get(loss_type)
    
    if loss_type == "softmax":
        loss_layer = LossClass(num_classes=num_classes, scale=arcface_scale,
                               name=f"{loss_type}_head")
    else:
        # Start with margin=0, will be increased slowly by ProgressiveMarginCallback
        loss_layer = LossClass(num_classes=num_classes, margin=0.0,
                               scale=arcface_scale, name=f"{loss_type}_head")

    logits = loss_layer([emb, lbl_in])

    training_model = Model(
        inputs=[img_in, lbl_in], outputs=logits,
        name=f"{loss_type.capitalize()}_Training_MobileFaceNet"
    )
    return training_model, backbone, loss_layer
