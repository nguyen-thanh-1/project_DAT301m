import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from anti_spoofing.config import Config

_cfg = Config()
WD = _cfg.weight_decay

class CentralDifferenceConv2D(layers.Layer):
    """Conv2D thuong + Central Difference term, theo Yu et al. CVPR2020 (CDCN)."""
    def __init__(self, filters, kernel_size=3, strides=1, theta=0.7, padding="same", **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.theta = theta
        self.padding = padding

    def build(self, input_shape):
        self.conv = layers.Conv2D(self.filters, self.kernel_size, strides=self.strides,
                                   padding=self.padding, use_bias=False,
                                   kernel_regularizer=regularizers.l2(WD))
        super().build(input_shape)

    def call(self, x):
        out_normal = self.conv(x)
        if abs(self.theta) < 1e-8:
            return out_normal
        # central difference term: dung 1x1 conv voi cung kernel de xap xi sum_pn w(pn) * x(p0)
        kernel_sum = tf.reduce_sum(self.conv.kernel, axis=[0, 1], keepdims=True)  # 1x1xCinxCout
        out_diff = tf.nn.conv2d(x, kernel_sum, strides=[1, self.strides, self.strides, 1],
                                 padding=self.padding.upper())
        return out_normal - self.theta * out_diff

    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(filters=self.filters, kernel_size=self.kernel_size,
                         strides=self.strides, theta=self.theta, padding=self.padding))
        return cfg


def conv_bn_act(x, filters, kernel_size, strides=1, groups=1, act=True):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding="same", groups=groups,
                       use_bias=False, kernel_regularizer=regularizers.l2(WD))(x)
    x = layers.BatchNormalization()(x)
    if act:
        x = layers.Activation("swish")(x)
    return x


def maybe_cdc_conv_bn_act(x, filters, kernel_size, strides=1, groups=1, act=True, theta=0.0):
    """CDC conv hoac Conv thuong tuy thuoc vao theta."""
    if theta > 0.0:
        x = CentralDifferenceConv2D(filters, kernel_size, strides=strides, theta=theta)(x)
        x = layers.BatchNormalization()(x)
        if act:
            x = layers.Activation("swish")(x)
        return x
    else:
        return conv_bn_act(x, filters, kernel_size, strides=strides, groups=groups, act=act)


def squeeze_excite(x, se_ratio=0.25):
    filters = x.shape[-1]
    se = layers.GlobalAveragePooling2D(keepdims=True)(x)
    se = layers.Conv2D(max(1, int(filters * se_ratio)), 1, activation="swish")(se)
    se = layers.Conv2D(filters, 1, activation="sigmoid")(se)
    return layers.Multiply()([x, se])


def fused_mbconv(x, out_ch, expand_ratio, strides, se=False, theta=0.0):
    in_ch = x.shape[-1]
    exp_ch = in_ch * expand_ratio
    h = maybe_cdc_conv_bn_act(x, exp_ch, 3, strides=strides, theta=theta)
    if se:
        h = squeeze_excite(h)
    h = conv_bn_act(h, out_ch, 1, act=False)
    if strides == 1 and in_ch == out_ch:
        h = layers.Add()([x, h])
    return h


def mbconv(x, out_ch, expand_ratio, strides, se=True):
    in_ch = x.shape[-1]
    exp_ch = in_ch * expand_ratio
    h = conv_bn_act(x, exp_ch, 1) if expand_ratio != 1 else x
    h = conv_bn_act(h, exp_ch, 3, strides=strides, groups=exp_ch)  # depthwise
    if se:
        h = squeeze_excite(h)
    h = conv_bn_act(h, out_ch, 1, act=False)
    if strides == 1 and in_ch == out_ch:
        h = layers.Add()([x, h])
    return h


def build_efficientnet_style(img_size, num_classes=2, width_mult=0.75, theta=0.7):
    def c(ch):
        return max(8, int(ch * width_mult))

    inputs = layers.Input(shape=(img_size, img_size, 3), name="rgb")
    
    # 1. Stem & Early blocks dung CDC de trich xuat texture live gradient
    x = maybe_cdc_conv_bn_act(inputs, c(32), 3, strides=2, theta=theta)          # stem: 224 -> 112

    x = fused_mbconv(x, c(16), expand_ratio=1, strides=1, theta=theta)
    x = fused_mbconv(x, c(32), expand_ratio=4, strides=2, theta=theta)  # 112 -> 56
    x = fused_mbconv(x, c(32), expand_ratio=4, strides=1, theta=theta)
    
    # Lay dac trung lop early (texture gia/that ro net nhat)
    x_early = x

    x = fused_mbconv(x, c(48), expand_ratio=4, strides=2, theta=theta)  # 56 -> 28
    x = fused_mbconv(x, c(48), expand_ratio=4, strides=1, theta=theta)

    x = mbconv(x, c(96), expand_ratio=4, strides=2, se=True)   # 28 -> 14
    x = mbconv(x, c(96), expand_ratio=4, strides=1, se=True)
    x = mbconv(x, c(96), expand_ratio=4, strides=1, se=True)
    
    # Lay dac trung trung binh
    x_mid = x

    x = mbconv(x, c(136), expand_ratio=6, strides=1, se=True)
    x = mbconv(x, c(136), expand_ratio=6, strides=1, se=True)

    x = mbconv(x, c(192), expand_ratio=6, strides=2, se=True)  # 14 -> 7
    x = mbconv(x, c(192), expand_ratio=6, strides=1, se=True)

    x = conv_bn_act(x, c(768), 1)
    x_late = x

    # 2. Multi-Scale Feature Pooling ( Pooling lay trung binh tren cac dac trung khoang cach )
    feat_early = layers.GlobalAveragePooling2D()(x_early)
    feat_mid = layers.GlobalAveragePooling2D()(x_mid)
    feat_late = layers.GlobalAveragePooling2D()(x_late)

    # 3. Concatenate dung hop dac trung (Texture + Shapes)
    fused = layers.Concatenate()([feat_early, feat_mid, feat_late])
    fused = layers.Dropout(0.3)(fused)
    
    logits = layers.Dense(num_classes, dtype="float32", name="logits")(fused)
    return Model(inputs={"rgb": inputs}, outputs=logits, name="EffNetStyle_FromScratch")


def cdc_bn_act(x, filters, kernel_size=3, strides=1, theta=0.7):
    x = CentralDifferenceConv2D(filters, kernel_size, strides, theta)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


def build_cdcn_lbp(img_size, num_classes=2, theta=0.7):
    rgb_in = layers.Input(shape=(img_size, img_size, 3), name="rgb")
    lbp_in = layers.Input(shape=(img_size, img_size, 1), name="lbp")

    # ---- RGB branch: CDC stem + 3 CDC blocks ----
    x = cdc_bn_act(rgb_in, 32, 3, strides=2, theta=theta)      # 224 -> 112
    x = cdc_bn_act(x, 64, 3, strides=1, theta=theta)
    x = layers.MaxPooling2D(2)(x)                               # 112 -> 56

    x = cdc_bn_act(x, 96, 3, strides=1, theta=theta)
    x = cdc_bn_act(x, 96, 3, strides=1, theta=theta)
    x = layers.MaxPooling2D(2)(x)                               # 56 -> 28

    x = cdc_bn_act(x, 128, 3, strides=1, theta=theta)
    x = cdc_bn_act(x, 128, 3, strides=1, theta=theta)
    x = layers.MaxPooling2D(2)(x)                               # 28 -> 14

    rgb_feat = layers.GlobalAveragePooling2D()(x)

    # ---- LBP branch: nhe hon, conv thuong (LBP da la texture-encoded roi) ----
    y = conv_bn_act(lbp_in, 16, 3, strides=2)                   # 224 -> 112
    y = conv_bn_act(y, 32, 3, strides=2)                        # 112 -> 56
    y = conv_bn_act(y, 64, 3, strides=2)                        # 56 -> 28
    y = layers.GlobalAveragePooling2D()(y)

    # ---- fusion ----
    fused = layers.Concatenate()([rgb_feat, y])
    fused = layers.Dense(128, activation="swish",
                          kernel_regularizer=regularizers.l2(WD))(fused)
    fused = layers.Dropout(0.4)(fused)
    logits = layers.Dense(num_classes, dtype="float32", name="logits")(fused)

    return Model(inputs={"rgb": rgb_in, "lbp": lbp_in}, outputs=logits, name="CDCN_LBP_FromScratch")


def se_block(x, channels, reduction=8):
    """Squeeze-and-Excitation (SE) block (Channel-only Attention to prevent spatial layout overfitting)"""
    gap = layers.GlobalAveragePooling2D()(x)
    fc1 = layers.Dense(max(1, channels // reduction), activation="swish")(gap)
    fc2 = layers.Dense(channels, activation="sigmoid")(fc1)
    fc2 = layers.Reshape((1, 1, channels))(fc2)
    return layers.Multiply()([x, fc2])


def resnet_basic_block(x, filters, strides=1, use_se=True):
    """ResNet Basic Block integrated with SE attention"""
    in_ch = x.shape[-1]
    
    h = layers.Conv2D(filters, 3, strides=strides, padding="same", use_bias=False,
                      kernel_regularizer=regularizers.l2(WD))(x)
    h = layers.BatchNormalization()(h)
    h = layers.Activation("swish")(h)
    
    h = layers.Conv2D(filters, 3, strides=1, padding="same", use_bias=False,
                      kernel_regularizer=regularizers.l2(WD))(h)
    h = layers.BatchNormalization()(h)
    
    if use_se:
        h = se_block(h, filters)
        
    if strides != 1 or in_ch != filters:
        shortcut = layers.Conv2D(filters, 1, strides=strides, padding="same", use_bias=False,
                                 kernel_regularizer=regularizers.l2(WD))(x)
        shortcut = layers.BatchNormalization()(shortcut)
    else:
        shortcut = x
        
    out = layers.Add()([shortcut, h])
    out = layers.Activation("swish")(out)
    return out


def build_resnet18_cbam(img_size, num_classes=2, theta=0.7):
    """Upgraded Custom ResNet-18 with CDC Stem + Channel-only SE Attention + Multi-Scale Feature Fusion"""
    inputs = layers.Input(shape=(img_size, img_size, 3), name="rgb")
    
    # CDC Stem block (224 -> 112 -> 56): uses CDC for gradient texture extraction robust to webcam lighting
    x = CentralDifferenceConv2D(32, 7, strides=2, theta=theta)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
    
    # Stage 1: channels=32, shape=56
    x = resnet_basic_block(x, 32, strides=1, use_se=True)
    x = resnet_basic_block(x, 32, strides=1, use_se=True)
    
    # Stage 2: channels=64, shape=28
    x = resnet_basic_block(x, 64, strides=2, use_se=True)
    x = resnet_basic_block(x, 64, strides=1, use_se=True)
    x_stage2 = x
    
    # Stage 3: channels=128, shape=14
    x = resnet_basic_block(x, 128, strides=2, use_se=True)
    x = resnet_basic_block(x, 128, strides=1, use_se=True)
    x_stage3 = x
    
    # Stage 4: channels=256, shape=7
    x = resnet_basic_block(x, 256, strides=2, use_se=True)
    x = resnet_basic_block(x, 256, strides=1, use_se=True)
    x_stage4 = x
    
    # Multi-Scale Feature Pooling (Fusing early textures, mid structures, and late geometry)
    feat_s2 = layers.GlobalAveragePooling2D()(x_stage2)
    feat_s3 = layers.GlobalAveragePooling2D()(x_stage3)
    feat_s4 = layers.GlobalAveragePooling2D()(x_stage4)
    
    # Concatenate features (64 + 128 + 256 = 448 channels)
    fused = layers.Concatenate()([feat_s2, feat_s3, feat_s4])
    fused = layers.Dropout(0.4)(fused)
    
    logits = layers.Dense(num_classes, dtype="float32", name="logits")(fused)
    
    return Model(inputs={"rgb": inputs}, outputs=logits, name="ResNet18_SE_CDC_MultiScale")


def build_resnet34_se_cdc_multiscale(img_size, num_classes=2, theta=0.7):
    """Ultimate ResNet-34 with CDC Stem + Channel-only SE Attention + Multi-Scale Feature Fusion"""
    inputs = layers.Input(shape=(img_size, img_size, 3), name="rgb")
    
    # CDC Stem block (224 -> 112 -> 56)
    x = CentralDifferenceConv2D(32, 7, strides=2, theta=theta)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
    
    # Stage 1: channels=32, blocks=3
    x = resnet_basic_block(x, 32, strides=1, use_se=True)
    x = resnet_basic_block(x, 32, strides=1, use_se=True)
    x = resnet_basic_block(x, 32, strides=1, use_se=True)
    
    # Stage 2: channels=64, blocks=4
    x = resnet_basic_block(x, 64, strides=2, use_se=True)
    x = resnet_basic_block(x, 64, strides=1, use_se=True)
    x = resnet_basic_block(x, 64, strides=1, use_se=True)
    x = resnet_basic_block(x, 64, strides=1, use_se=True)
    x_stage2 = x
    
    # Stage 3: channels=128, blocks=6
    x = resnet_basic_block(x, 128, strides=2, use_se=True)
    x = resnet_basic_block(x, 128, strides=1, use_se=True)
    x = resnet_basic_block(x, 128, strides=1, use_se=True)
    x = resnet_basic_block(x, 128, strides=1, use_se=True)
    x = resnet_basic_block(x, 128, strides=1, use_se=True)
    x = resnet_basic_block(x, 128, strides=1, use_se=True)
    x_stage3 = x
    
    # Stage 4: channels=256, blocks=3
    x = resnet_basic_block(x, 256, strides=2, use_se=True)
    x = resnet_basic_block(x, 256, strides=1, use_se=True)
    x = resnet_basic_block(x, 256, strides=1, use_se=True)
    x_stage4 = x
    
    # Multi-Scale Feature Pooling (Fusing early textures, mid structures, and late geometry)
    feat_s2 = layers.GlobalAveragePooling2D()(x_stage2)
    feat_s3 = layers.GlobalAveragePooling2D()(x_stage3)
    feat_s4 = layers.GlobalAveragePooling2D()(x_stage4)
    
    # Concatenate features (64 + 128 + 256 = 448 channels)
    fused = layers.Concatenate()([feat_s2, feat_s3, feat_s4])
    fused = layers.Dropout(0.4)(fused)
    
    logits = layers.Dense(num_classes, dtype="float32", name="logits")(fused)
    
    return Model(inputs={"rgb": inputs}, outputs=logits, name="ResNet34_SE_CDC_MultiScale")




