import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os, math, argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from training.face_recognition.resnet50 import build_training_model

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
# 2. Transformer Encoder Components
# =====================================================================
class LearnablePositionalEncoding(layers.Layer):
    """Adds learnable positional embeddings to token sequences."""
    def build(self, input_shape):
        _, num_tokens, d_model = input_shape
        self.pos_emb = self.add_weight(
            'pos_emb', shape=(1, num_tokens, d_model),
            initializer='truncated_normal', trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        return x + self.pos_emb

    def get_config(self):
        return super().get_config()


class TransformerEncoderBlock(layers.Layer):
    """
    Pre-LayerNorm Transformer Encoder with Multi-Head Self-Attention + FFN.
    Placed after ResNet-50 Stage 5 to capture global spatial relationships
    between facial feature patches (eyes↔nose↔mouth interactions).
    """
    def __init__(self, d_model, num_heads, dff, drop_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.drop_rate = drop_rate
        if self.d_model % self.num_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})")

        self.ln1 = layers.LayerNormalization(epsilon=1e-6, name='ln1')
        self.mha = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model // self.num_heads,
            dropout=self.drop_rate,
            name='mha'
        )
        self.ln2 = layers.LayerNormalization(epsilon=1e-6, name='ln2')
        self.ffn_dense1 = layers.Dense(self.dff, activation='relu', kernel_initializer='he_normal', name='ffn_dense1')
        self.ffn_drop1 = layers.Dropout(self.drop_rate, name='ffn_drop1')
        self.ffn_dense2 = layers.Dense(self.d_model, kernel_initializer='he_normal', name='ffn_dense2')
        self.ffn_drop2 = layers.Dropout(self.drop_rate, name='ffn_drop2')

    def call(self, x, training=False):
        x_n = self.ln1(x)
        x = x + self.mha(x_n, x_n, training=training)
        x_n = self.ln2(x)
        ffn_out = self.ffn_dense1(x_n)
        ffn_out = self.ffn_drop1(ffn_out, training=training)
        ffn_out = self.ffn_dense2(ffn_out)
        ffn_out = self.ffn_drop2(ffn_out, training=training)
        x = x + ffn_out
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d_model": self.d_model, "num_heads": self.num_heads,
                     "dff": self.dff, "drop_rate": self.drop_rate})
        return cfg

# =====================================================================
# 3. ResNet-50 + Transformer Hybrid Backbone
# =====================================================================
def identity_block(x, kernel_size, filters, stage, block):
    f1, f2, f3 = filters
    cb = f'res{stage}{block}_branch'
    bb = f'bn{stage}{block}_branch'
    shortcut = x
    x = layers.Conv2D(f1, (1,1), kernel_initializer='he_normal', name=cb+'2a')(x)
    x = layers.BatchNormalization(name=bb+'2a')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(f2, kernel_size, padding='same', kernel_initializer='he_normal', name=cb+'2b')(x)
    x = layers.BatchNormalization(name=bb+'2b')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(f3, (1,1), kernel_initializer='he_normal', name=cb+'2c')(x)
    x = layers.BatchNormalization(name=bb+'2c')(x)
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def conv_block(x, kernel_size, filters, stage, block, strides=(2,2)):
    f1, f2, f3 = filters
    cb = f'res{stage}{block}_branch'
    bb = f'bn{stage}{block}_branch'
    shortcut = x
    x = layers.Conv2D(f1, (1,1), strides=strides, kernel_initializer='he_normal', name=cb+'2a')(x)
    x = layers.BatchNormalization(name=bb+'2a')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(f2, kernel_size, padding='same', kernel_initializer='he_normal', name=cb+'2b')(x)
    x = layers.BatchNormalization(name=bb+'2b')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(f3, (1,1), kernel_initializer='he_normal', name=cb+'2c')(x)
    x = layers.BatchNormalization(name=bb+'2c')(x)
    shortcut = layers.Conv2D(f3, (1,1), strides=strides, kernel_initializer='he_normal', name=cb+'1')(shortcut)
    shortcut = layers.BatchNormalization(name=bb+'1')(shortcut)
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def ResNet50_Backbone(input_shape=(112,112,3), embedding_dim=512,
                      use_transformer=True, num_transformer_layers=2,
                      num_heads=8, transformer_dff=1024, drop_rate=0.1):
    """
    Hybrid CNN-Transformer backbone for face recognition.
    
    Architecture flow:
        Image (112x112x3)
          → ResNet-50 Stages 1-5 → Feature Map (4x4x2048)
          → 1x1 Conv Projection → (4x4x512)
          → Reshape to token sequence → (16 tokens x 512 dims)
          → Positional Encoding
          → 2x Transformer Encoder Blocks (global attention)
          → Global Average Pooling over tokens → (512,)
          → BatchNorm → Embedding Vector (512,)
    """
    img_input = layers.Input(shape=input_shape, name='image_input')

    # === ResNet-50 CNN Feature Extractor ===
    # Stage 1: (112,112,3) → (28,28,64)
    x = layers.ZeroPadding2D((3,3))(img_input)
    x = layers.Conv2D(64, (7,7), strides=(2,2), kernel_initializer='he_normal', name='conv1')(x)
    x = layers.BatchNormalization(name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D((1,1))(x)
    x = layers.MaxPooling2D((3,3), strides=(2,2))(x)

    # Stage 2: (28,28,64) → (28,28,256)
    x = conv_block(x, 3, [64,64,256], stage=2, block='a', strides=(1,1))
    x = identity_block(x, 3, [64,64,256], stage=2, block='b')
    x = identity_block(x, 3, [64,64,256], stage=2, block='c')

    # Stage 3: (28,28,256) → (14,14,512)
    x = conv_block(x, 3, [128,128,512], stage=3, block='a')
    x = identity_block(x, 3, [128,128,512], stage=3, block='b')
    x = identity_block(x, 3, [128,128,512], stage=3, block='c')
    x = identity_block(x, 3, [128,128,512], stage=3, block='d')

    # Stage 4: (14,14,512) → (7,7,1024)
    x = conv_block(x, 3, [256,256,1024], stage=4, block='a')
    x = identity_block(x, 3, [256,256,1024], stage=4, block='b')
    x = identity_block(x, 3, [256,256,1024], stage=4, block='c')
    x = identity_block(x, 3, [256,256,1024], stage=4, block='d')
    x = identity_block(x, 3, [256,256,1024], stage=4, block='e')
    x = identity_block(x, 3, [256,256,1024], stage=4, block='f')

    # Stage 5: (7,7,1024) → (4,4,2048)
    x = conv_block(x, 3, [512,512,2048], stage=5, block='a')
    x = identity_block(x, 3, [512,512,2048], stage=5, block='b')
    x = identity_block(x, 3, [512,512,2048], stage=5, block='c')

    if use_transformer:
        # === Transformer Feature Enhancer ===
        # Project 2048 channels → embedding_dim via 1x1 Conv
        x = layers.Conv2D(embedding_dim, (1,1), kernel_initializer='he_normal', name='trans_proj')(x)
        x = layers.BatchNormalization(name='bn_trans_proj')(x)
        x = layers.Activation('relu')(x)
        # (4,4,512) → (16 tokens, 512 dims)

        # Flatten spatial grid into token sequence
        x = layers.Reshape((-1, embedding_dim), name='spatial_to_tokens')(x)

        # Learnable positional encoding
        x = LearnablePositionalEncoding(name='pos_encoding')(x)

        # Transformer encoder blocks (global self-attention)
        for i in range(num_transformer_layers):
            x = TransformerEncoderBlock(
                d_model=embedding_dim, num_heads=num_heads,
                dff=transformer_dff, drop_rate=drop_rate,
                name=f'transformer_{i}'
            )(x)

        # Final LayerNorm + Global Average Pooling over tokens
        x = layers.LayerNormalization(epsilon=1e-6, name='trans_final_norm')(x)
        embeddings = layers.GlobalAveragePooling1D(name='token_pool')(x)
    else:
        # Standard CNN-only path
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        embeddings = layers.Dense(embedding_dim, activation=None, kernel_initializer='he_normal', name='embedding_dense')(x)

    # BatchNorm on embeddings (stabilizes L2 normalization in ArcFace)
    embeddings = layers.BatchNormalization(name='bn_embeddings')(embeddings)

    model = Model(inputs=img_input, outputs=embeddings, name='ResNet50T_Backbone')
    return model

# =====================================================================
# 4. ArcFace Custom Layer
# =====================================================================
class ArcFace(layers.Layer):
    """ArcFace (Additive Angular Margin Loss) classification head."""
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

    def build(self, input_shape):
        self.W = self.add_weight(
            'W', shape=(input_shape[0][-1], self.num_classes),
            initializer='glorot_uniform', trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        embeddings, labels = inputs
        labels = tf.reshape(labels, [-1])
        norm_e = tf.nn.l2_normalize(embeddings, axis=1)
        norm_w = tf.nn.l2_normalize(self.W, axis=0)
        cos_t = tf.clip_by_value(tf.matmul(norm_e, norm_w), -1.0+1e-7, 1.0-1e-7)
        sin_t = tf.sqrt(1.0 - tf.square(cos_t) + 1e-7)
        cos_tm = cos_t * self.cos_m - sin_t * self.sin_m
        cos_tm = tf.where(cos_t > self.threshold, cos_tm, cos_t - self.mm)
        one_hot = tf.one_hot(tf.cast(labels, tf.int32), depth=self.num_classes)
        output = (one_hot * cos_tm) + ((1.0 - one_hot) * cos_t)
        return output * self.scale

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"num_classes": self.num_classes, "scale": self.scale, "margin": self.margin})
        return cfg

# =====================================================================
# 5. Progressive Margin Callback
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

# =====================================================================
# 6. Learning Rate Schedule (Warmup + Cosine Decay)
# =====================================================================
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, peak_lr, warmup_steps, total_steps, min_lr=1e-6):
        super().__init__()
        self.peak_lr = peak_lr
        self.warmup_steps = float(warmup_steps)
        self.total_steps = float(total_steps)
        self.min_lr = min_lr

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup = self.peak_lr * (step / tf.maximum(self.warmup_steps, 1.0))
        progress = (step - self.warmup_steps) / tf.maximum(self.total_steps - self.warmup_steps, 1.0)
        cosine = self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (1.0 + tf.cos(3.14159265 * progress))
        return tf.where(step < self.warmup_steps, warmup, cosine)

    def get_config(self):
        return {"peak_lr": self.peak_lr, "warmup_steps": self.warmup_steps,
                "total_steps": self.total_steps, "min_lr": self.min_lr}

# =====================================================================
# 7. Data Pipeline
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
    shuffle_buffer=10000,
    crop_to_aspect_ratio=True,
):
    print(f"\n--- Building Data Pipeline ---")
    print(f"Train: {train_dir}")

    cache_train_path = os.path.join(cache_dir, 'train.cache')
    cache_val_path = os.path.join(cache_dir, 'val.cache')

    use_val_dir = False
    if val_dir and os.path.isdir(val_dir):
        train_classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
        val_classes = sorted([d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))])
        if train_classes and train_classes == val_classes:
            use_val_dir = True
            print(f"Val:   {val_dir} (matched classes)")
        else:
            print(f"Val:   {val_dir} (ignored; classes do not match train)")

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
            class_names=train_ds.class_names,
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
    print(f"Classes: {num_classes} | Train batches/epoch: {num_train_batches}")

    def preprocess_train(image, label):
        image = (tf.cast(image, tf.float32) - 127.5) / 128.0
        image = tf.image.random_flip_left_right(image)  # Data Augmentation
        return (image, label), label

    def preprocess_val(image, label):
        image = (tf.cast(image, tf.float32) - 127.5) / 128.0
        return (image, label), label

    # Optimized pipeline: cache raw → augment → shuffle batches → prefetch
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
# 8. Model Builder
# =====================================================================
def build_model(input_shape, num_classes, embedding_dim=512, use_transformer=True):
    return build_training_model(
        input_shape=input_shape,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        use_transformer=use_transformer,
    )

# =====================================================================
# 9. Plot History
# =====================================================================
def plot_history(history, save_path):
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    epochs_range = range(1, len(loss) + 1)

    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, loss, label='Train Loss', color='#e74c3c', lw=2.5)
    if val_loss:
        plt.plot(epochs_range, val_loss, label='Val Loss', color='#3498db', lw=2.5)
    plt.title('Loss (ArcFace + Progressive Margin)', fontsize=13, fontweight='bold')
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend(); plt.grid(True, ls='--', alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, acc, label='Train Accuracy', color='#2ecc71', lw=2.5)
    if val_acc:
        plt.plot(epochs_range, val_acc, label='Val Accuracy', color='#9b59b6', lw=2.5)
    plt.title('Accuracy', fontsize=13, fontweight='bold')
    plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True, ls='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"\n[INFO] Plot saved to: {save_path}")

# =====================================================================
# 10. Custom Callback: Save Backbone at Best Epoch
# =====================================================================
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

# =====================================================================
# 11. Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="ResNet-50+Transformer ArcFace Trainer")
    parser.add_argument('--dataset_dir', type=str, default='dataset/train', help='Training directory (class subfolders)')
    parser.add_argument('--val_dir', type=str, default='dataset/val', help='Validation directory (optional)')
    parser.add_argument('--val_split', type=float, default=0.2, help='Used only when val_dir is missing')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--img_size', type=int, default=112)
    parser.add_argument('--embedding_dim', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--no_transformer', action='store_true', help='Disable transformer module')
    parser.add_argument('--cache', type=str, default='disk', choices=['disk', 'memory', 'none'],
                        help='Dataset cache mode (disk recommended for large datasets)')
    parser.add_argument('--cache_dir', type=str, default='results/tf_cache', help='Directory for disk cache files')
    parser.add_argument('--shuffle_buffer', type=int, default=10000, help='Shuffle buffer size')
    parser.add_argument('--crop_to_aspect_ratio', action=argparse.BooleanOptionalAction, default=True,
                        help='Crop to preserve aspect ratio before resizing.')
    args = parser.parse_args()

    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    # Hardware
    configure_gpu()

    # Data
    img_shape = (args.img_size, args.img_size, 3)
    train_ds, val_ds, num_classes, num_train_batches = build_dataset(
        args.dataset_dir,
        args.val_dir,
        args.batch_size,
        (args.img_size, args.img_size),
        val_split=args.val_split,
        cache_mode=args.cache,
        cache_dir=args.cache_dir,
        shuffle_buffer=args.shuffle_buffer,
        crop_to_aspect_ratio=args.crop_to_aspect_ratio,
    )

    # Model
    use_transformer = not args.no_transformer
    training_model, backbone, arcface_layer = build_model(
        img_shape, num_classes, args.embedding_dim, use_transformer
    )
    backbone.summary()

    # LR Schedule: 1 epoch warmup + cosine decay
    warmup_steps = num_train_batches  # 1 epoch warmup
    total_steps = num_train_batches * args.epochs
    lr_schedule = WarmupCosineDecay(
        peak_lr=args.lr, warmup_steps=warmup_steps, total_steps=total_steps
    )

    # Compile
    training_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            os.path.join(results_dir, 'best_resnet50t_arcface.h5'),
            save_best_only=True, save_weights_only=True, monitor='val_loss', mode='min', verbose=1
        ),
        SaveBestBackbone(backbone, os.path.join(results_dir, 'best_backbone_weights.h5')),
        ProgressiveMarginCallback(
            arcface_layer, start_margin=0.0, end_margin=0.50, total_epochs=args.epochs
        ),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ]

    # Train
    print(f"\n{'='*60}")
    print(f"Training Config:")
    print(f"  Backbone:     ResNet-50 {'+ Transformer' if use_transformer else '(CNN only)'}")
    print(f"  Classes:      {num_classes}")
    print(f"  Batch Size:   {args.batch_size}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Peak LR:      {args.lr}")
    print(f"  ArcFace:      margin 0.10→0.50 (progressive), scale=64")
    print(f"  Warmup:       1 epoch ({warmup_steps} steps)")
    print(f"{'='*60}\n")

    history = training_model.fit(
        train_ds, validation_data=val_ds,
        epochs=args.epochs, callbacks=callbacks
    )

    # Plot
    plot_history(history, os.path.join(results_dir, 'training_history.png'))
    print("\n[SUCCESS] Training complete. Results saved in results/")

if __name__ == '__main__':
    main()
