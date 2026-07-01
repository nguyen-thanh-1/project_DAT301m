import argparse
import os
import sys
import math
from pathlib import Path

# Add project root to path to allow importing shared modules
sys.path.append(str(Path(__file__).resolve().parents[3]))

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from shared.utils import (
    SaveBestBackbone,
    VerificationCallback,
    WarmupCosineDecay,
    configure_gpu,
    plot_history,
    HistoryPlotterCallback,
)
from models.face_recognition.mobilefacenet.mobile_face_net import build_mobilefacenet_training_model
from models.face_recognition.iresnet.iresnet50 import DEFAULT_MARGINS
from data_processing.prepare_dataset import build_training_dataset


class SlowProgressiveMarginCallback(tf.keras.callbacks.Callback):
    """Slowly increases margin by a fixed step per epoch up to max_margin.
    
    Starts epoch 1 (index 0) at margin = 0.0, and increases by step each epoch.
    """
    def __init__(self, margin_layer, step=0.01, max_margin=0.50):
        super().__init__()
        self.margin_layer = margin_layer
        self.step = step
        self.max_margin = max_margin

    def on_epoch_begin(self, epoch, logs=None):
        new_m = min(epoch * self.step, self.max_margin)
        self.margin_layer.margin = new_m
        if hasattr(self.margin_layer, "_update_margin_constants"):
            self.margin_layer._update_margin_constants(new_m)
        print(f"\n[Margin Schedule] Epoch {epoch+1}: margin = {new_m:.4f} rad ({math.degrees(new_m):.2f} deg)")


class SaveTrainingMetaCallback(tf.keras.callbacks.Callback):
    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath

    def on_epoch_end(self, epoch, logs=None):
        import json
        meta = {
            "last_epoch": epoch,
            "val_loss": float(logs.get("val_loss", 0.0)) if logs else 0.0
        }
        with open(self.filepath, "w") as f:
            json.dump(meta, f)


def get_paths_and_labels(dataset_dir, class_to_idx=None):
    """Scans dataset directory and returns sorted image paths and label index lists."""
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        return [], [], 0 if class_to_idx is None else len(class_to_idx)
    
    if class_to_idx is None:
        class_names = sorted([d.name for d in dataset_dir.iterdir() if d.is_dir()])
        class_to_idx = {name: i for i, name in enumerate(class_names)}
    
    image_paths = []
    labels = []
    
    for class_name, idx in class_to_idx.items():
        class_dir = dataset_dir / class_name
        if not class_dir.is_dir():
            continue
        for img_path in class_dir.iterdir():
            if img_path.is_file() and img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                image_paths.append(str(img_path))
                labels.append(idx)
                
    return image_paths, labels, len(class_to_idx)


def main():
    parser = argparse.ArgumentParser(description="MobileFaceNet ArcFace Trainer")
    parser.add_argument("--dataset_dir", type=str, default="dataset_final/train",
                        help="Training directory (class subfolders)")
    parser.add_argument("--val_dir", type=str, default="dataset_final/val",
                        help="Validation directory (SAME classes as train)")
    parser.add_argument("--test_dir", type=str, default="dataset_final/test",
                        help="Test directory (UNSEEN classes, for verification eval)")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size (256/512 recommended for GPU 16GB VRAM)")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--img_size", type=int, default=112)
    parser.add_argument("--embedding_dim", type=int, default=256,
                        help="Embedding dimension (256 for MobileFaceNet)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Peak learning rate for AdamW")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay for AdamW")
    parser.add_argument("--arcface_scale", type=float, default=32.0,
                        help="ArcFace scale factor (s)")
    parser.add_argument("--loss_type", type=str, default="arcface",
                        choices=["arcface", "cosface", "sphereface", "softmax"],
                        help="Loss function: arcface | cosface | sphereface | softmax")
    parser.add_argument("--margin_step", type=float, default=0.01,
                        help="Incremental step for margin progressive scheduling")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--verify_every", type=int, default=3,
                        help="Run verification eval every N epochs")
    parser.add_argument("--verify_pairs", type=int, default=3000,
                        help="Number of pairs for verification eval")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from previous checkpoint if available")
    args, unknown = parser.parse_known_args()

    # Results go into the model's local results folder, organized by loss type
    results_dir = os.path.join(os.path.dirname(__file__), "results", args.loss_type)
    os.makedirs(results_dir, exist_ok=True)

    configure_gpu()

    # Load paths
    train_dir = Path(args.dataset_dir)
    train_class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    train_class_to_idx = {name: i for i, name in enumerate(train_class_names)}

    train_paths, train_labels, num_classes = get_paths_and_labels(args.dataset_dir, class_to_idx=train_class_to_idx)
    val_paths, val_labels, _ = get_paths_and_labels(args.val_dir, class_to_idx=train_class_to_idx)
    
    print(f"[Data] Loaded {len(train_paths)} train images, {len(val_paths)} val images across {num_classes} classes.")

    if len(train_paths) == 0:
        raise SystemExit(f"Error: dataset_dir {args.dataset_dir} does not contain any images or classes.")

    # Build tf.data pipeline
    train_ds = build_training_dataset(train_paths, train_labels, batch_size=args.batch_size, is_training=True)
    val_ds = build_training_dataset(val_paths, val_labels, batch_size=args.batch_size, is_training=False)

    num_train_batches = math.ceil(len(train_paths) / args.batch_size)

    # Check for resume option
    initial_epoch = 0
    start_step = 0.0
    checkpoint_path = os.path.join(results_dir, "best_mobilefacenet.h5")
    meta_path = os.path.join(results_dir, "training_meta.json")

    if args.resume:
        import json
        if os.path.exists(meta_path) and os.path.exists(checkpoint_path):
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                initial_epoch = meta.get("last_epoch", -1) + 1
                start_step = float(initial_epoch * num_train_batches)
                print(f"[Resume] Loaded training metadata. Resuming from epoch {initial_epoch + 1}")
            except Exception as e:
                print(f"[Resume Warning] Could not parse metadata: {e}. Starting from scratch.")
                initial_epoch = 0
                start_step = 0.0
        else:
            print("[Resume Warning] Checkpoint or metadata files not found. Starting from scratch.")

    if initial_epoch >= args.epochs:
        print(f"[Resume] Training already completed up to {args.epochs} epochs. Nothing to do!")
        return

    img_shape = (args.img_size, args.img_size, 3)
    training_model, backbone, loss_layer = build_mobilefacenet_training_model(
        input_shape=img_shape,
        num_classes=num_classes,
        embedding_dim=args.embedding_dim,
        arcface_scale=args.arcface_scale,
        loss_type=args.loss_type,
    )
    backbone.summary()

    # Load weights if resuming
    if args.resume and initial_epoch > 0:
        if os.path.exists(checkpoint_path):
            try:
                training_model.load_weights(checkpoint_path)
                print(f"[Resume] Loaded model weights from: {checkpoint_path}")
            except Exception as e:
                print(f"[Resume Error] Could not load model weights: {e}. Starting from scratch.")
                initial_epoch = 0
                start_step = 0.0

    # Learning Rate Schedule (warmup 5 epochs -> cosine decay)
    warmup_steps = num_train_batches * 5
    total_steps = num_train_batches * args.epochs
    lr_schedule = WarmupCosineDecay(
        peak_lr=args.lr, warmup_steps=warmup_steps, total_steps=total_steps, start_step=start_step
    )

    # Use AdamW if available (TF 2.11+), otherwise fallback to standard Adam
    try:
        optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=args.weight_decay)
        print("[Optimizer] Initialized AdamW optimizer.")
    except AttributeError:
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        print("[Optimizer Warning] AdamW not available, falling back to standard Adam.")

    training_model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    # Determine end margin for progressive scheduling
    end_margin = DEFAULT_MARGINS.get(args.loss_type, 0.5)

    callbacks = [
        ModelCheckpoint(
            checkpoint_path,
            save_best_only=True,
            save_weights_only=True,
            monitor="val_loss",
            mode="min",
            verbose=1,
        ),
        SaveBestBackbone(
            backbone,
            os.path.join(results_dir, "best_mobilefacenet_backbone.h5"),
        ),
        SlowProgressiveMarginCallback(
            loss_layer,
            step=args.margin_step,
            max_margin=end_margin,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            restore_best_weights=True,
            verbose=1,
        ),
        HistoryPlotterCallback(
            os.path.join(results_dir, "training_history.png")
        ),
        SaveTrainingMetaCallback(meta_path),
    ]

    # Add verification callback if test_dir exists
    if args.test_dir and os.path.isdir(args.test_dir):
        callbacks.append(
            VerificationCallback(
                backbone,
                test_dir=args.test_dir,
                img_size=args.img_size,
                num_pairs=args.verify_pairs,
                eval_every=args.verify_every,
                batch_size=args.batch_size,
                save_dir=results_dir,
                file_prefix="",  # Save as verification_roc.png cleanly inside loss folder
            )
        )
        print(f"[Verification] Enabled on {args.test_dir} (every {args.verify_every} epochs)")

    print("\n" + "=" * 60)
    print(f"Training Config (MobileFaceNet + {args.loss_type.upper()} Head)")
    print(f"  Loss Type:   {args.loss_type}")
    print(f"  Max Margin:  {end_margin}")
    print(f"  Classes:     {num_classes}")
    print(f"  Img Size:    {args.img_size}")
    print(f"  Batch Size:  {args.batch_size}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Peak LR:     {args.lr}")
    print(f"  WeightDecay: {args.weight_decay}")
    print(f"  Scale (s):   {args.arcface_scale}")
    print(f"  Patience:    {args.patience}")
    print(f"  Params:      {training_model.count_params():,}")
    print("=" * 60 + "\n")

    history = training_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        initial_epoch=initial_epoch,
    )
    plot_history(history, os.path.join(results_dir, "training_history.png"))
    print(f"\n[SUCCESS] MobileFaceNet Training complete. Results saved in {results_dir}/")


if __name__ == "__main__":
    main()
