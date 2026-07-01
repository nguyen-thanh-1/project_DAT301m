"""
iResNet50 Training Script — ArcFace Paper Configuration (Deng et al., 2019)

Training configuration follows the ArcFace paper on VGGFace2:
- Optimizer: SGD(lr=0.01, momentum=0.9, nesterov=True, weight_decay=5e-4)
- LR Schedule: Step Decay at milestones [16, 22, 26, 30], gamma=0.1
- Progressive Margin (step-wise per 5 epochs):
    Epoch 1-5:  m=0.1, Epoch 6-10: m=0.2, Epoch 11-15: m=0.3,
    Epoch 16-20: m=0.4, Epoch 21+: m=0.5
- Scale: s=64.0 (fixed)
- Dropout: 0.5 in backbone output head (BN-Dropout-FC-BN per ArcFace paper)
- Loss: CategoricalCrossentropy(label_smoothing=0.1) for better generalization
- Data: Random horizontal flip, brightness/contrast jitter

Usage (from Jupyter Notebook):
    sys.argv = [
        'train.py',
        '--dataset_dir', 'dataset_final/train',
        '--val_dir', 'dataset_final/val',
        '--test_dir', 'dataset_final/test',
        '--batch_size', '128',
        '--epochs', '30',
        '--loss_type', 'arcface',
        '--verify_every', '3',
    ]
    train_main()
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

# Add project root to path to allow importing shared modules
sys.path.append(str(Path(__file__).resolve().parents[3]))

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from shared.utils import (
    SaveBestBackbone,
    VerificationCallback,
    configure_gpu,
    plot_history,
    HistoryPlotterCallback,
)
from models.face_recognition.iresnet.iresnet50 import build_training_model, DEFAULT_MARGINS
from data_processing.prepare_dataset import build_training_dataset


# =====================================================================
# Callbacks
# =====================================================================

class ProgressiveMarginCallback(tf.keras.callbacks.Callback):
    """Step-wise Progressive Margin Schedule (per ArcFace paper spec).
    
    Schedule (using ABSOLUTE 1-indexed epochs):
      Epoch  1-5:   margin = 0.1  (warm-up, avoid divergence)
      Epoch  6-10:  margin = 0.2
      Epoch 11-15:  margin = 0.3
      Epoch 16-20:  margin = 0.4
      Epoch 21+:    margin = 0.5  (final value from paper)
    
    Scale s = 64 remains fixed throughout training.
    
    VERIFIED: Keras passes ABSOLUTE 0-indexed epoch to callbacks when using
    model.fit(initial_epoch=N). E.g., initial_epoch=10 → epoch=10,11,12...
    So epoch+1 gives the correct 1-indexed absolute epoch.
    """
    def __init__(self, margin_layer, max_margin=0.50, step_size=0.1, step_every=5):
        super().__init__()
        self.margin_layer = margin_layer
        self.max_margin = max_margin
        self.step_size = step_size    # margin increment per step
        self.step_every = step_every  # epochs per step

    def on_epoch_begin(self, epoch, logs=None):
        # epoch is ABSOLUTE 0-indexed (Keras passes 10,11,12... with initial_epoch=10)
        # Convert to 1-indexed: epoch 0→1, epoch 10→11, etc.
        abs_epoch_1idx = epoch + 1
        # step_num: epoch 1-5 → step 1, epoch 6-10 → step 2, etc.
        step_num = ((abs_epoch_1idx - 1) // self.step_every) + 1
        new_m = min(step_num * self.step_size, self.max_margin)
        self.margin_layer._update_margin_constants(new_m)
        print(f"\n[Margin Schedule] Epoch {abs_epoch_1idx}: margin = {new_m:.2f} rad ({math.degrees(new_m):.1f} deg)")


class StepDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Step Decay LR Schedule per ArcFace paper on VGGFace2.
    
    Community-verified configuration (safe for smaller datasets):
      - Epoch  1-16: LR = 0.01
      - Epoch 17-22: LR = 0.001
      - Epoch 23-26: LR = 0.0001
      - Epoch 27-30: LR = 0.00001
    
    Milestones = [16, 22, 26, 30], gamma = 0.1
    
    Optional linear warmup: LR increases from 0 to initial_lr over warmup_epochs.
    """
    def __init__(self, initial_lr=0.01, milestones_epochs=None, gamma=0.1,
                 steps_per_epoch=1, start_step=0.0, warmup_epochs=0):
        super().__init__()
        self.initial_lr = initial_lr
        self.milestones_epochs = milestones_epochs or [16, 22, 26, 30]
        self.gamma = gamma
        self.steps_per_epoch = steps_per_epoch
        self.start_step = start_step
        self.warmup_epochs = warmup_epochs
        self.warmup_steps = warmup_epochs * steps_per_epoch
        # Convert epoch milestones to step milestones
        self.milestone_steps = [e * steps_per_epoch for e in self.milestones_epochs]

    def __call__(self, step):
        # NOTE: Do NOT add self.start_step here.
        # When resuming from a full checkpoint, optimizer.iterations is restored
        # to the correct step count automatically by tf.train.Checkpoint.restore().
        step = tf.cast(step, tf.float32)
        # Step decay
        lr = self.initial_lr
        for ms in self.milestone_steps:
            lr = tf.where(step >= float(ms), lr * self.gamma, lr)
        # Linear warmup: ramp from 0 to initial_lr over warmup_steps
        if self.warmup_steps > 0:
            warmup_lr = self.initial_lr * (step / float(self.warmup_steps))
            lr = tf.where(step < float(self.warmup_steps), warmup_lr, lr)
        return lr

    def get_config(self):
        return {
            "initial_lr": self.initial_lr,
            "milestones_epochs": self.milestones_epochs,
            "gamma": self.gamma,
            "steps_per_epoch": self.steps_per_epoch,
            "start_step": self.start_step,
            "warmup_epochs": self.warmup_epochs,
        }


class TrainingCheckpointCallback(tf.keras.callbacks.Callback):
    """Unified checkpoint callback: saves tf.train.Checkpoint + training_meta.json.
    
    Saves FULL training state every epoch:
      - Model weights (backbone + loss head) via tf.train.Checkpoint
      - Optimizer state (momentum buffers, LR scheduler position)
      - Epoch counter (absolute)
      - Current margin value
      - Current learning rate
      - Best val_accuracy tracker
      - Full epoch-by-epoch history
    
    Manages two checkpoint sets:
      - latest/ : last 2 checkpoints (for resume)
      - best/   : 1 checkpoint with highest val_accuracy
    """
    def __init__(self, latest_manager, best_manager, epoch_var,
                 meta_path, loss_layer, optimizer, num_classes):
        super().__init__()
        self.latest_manager = latest_manager
        self.best_manager = best_manager
        self.epoch_var = epoch_var
        self.meta_path = meta_path
        self.loss_layer = loss_layer
        self.optimizer = optimizer
        self.num_classes = num_classes
        self.best_val_accuracy = 0.0
        # History accumulator — loaded from existing meta on init
        self.history = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "lr": [], "margin": [],
        }
        self._load_existing_history()

    def _load_existing_history(self):
        """Load history from existing meta file for seamless resume."""
        if not os.path.exists(self.meta_path):
            return
        try:
            with open(self.meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if "history" in meta:
                for k in self.history:
                    if k in meta["history"]:
                        self.history[k] = meta["history"][k]
            self.best_val_accuracy = float(meta.get("best_val_accuracy", 0.0))
        except Exception:
            pass

    def _get_current_lr(self):
        """Get current learning rate from optimizer."""
        try:
            lr = self.optimizer.learning_rate
            if callable(lr):
                return float(lr(self.optimizer.iterations).numpy())
            return float(lr.numpy() if hasattr(lr, 'numpy') else lr)
        except Exception:
            return 0.0

    def _save_with_retry(self, manager, ckpt_number, max_retries=3, delay=2):
        """Save checkpoint with retry on Windows file-locking errors."""
        import time
        last_error = None
        for attempt in range(max_retries):
            try:
                return manager.save(checkpoint_number=ckpt_number)
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    print(f"[Checkpoint] Save failed (attempt {attempt+1}/{max_retries}), "
                          f"retrying in {delay}s... [{type(e).__name__}]")
                    time.sleep(delay)
        raise last_error

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # epoch is ABSOLUTE 0-indexed from Keras
        self.epoch_var.assign(epoch + 1)

        # Save latest checkpoint (retry on Windows file-locking)
        latest_path = self._save_with_retry(self.latest_manager, epoch + 1)

        # Track best val_accuracy and save best checkpoint
        val_acc = float(logs.get("val_accuracy",
                        logs.get("val_sparse_categorical_accuracy", 0.0)))
        best_path = None
        if val_acc > self.best_val_accuracy:
            self.best_val_accuracy = val_acc
            best_path = self._save_with_retry(self.best_manager, epoch + 1)
            print(f"[Best Checkpoint] New best val_accuracy={val_acc:.4f}, saved: {best_path}")

        # Get current margin and lr
        current_margin = float(self.loss_layer.margin_var.numpy())
        current_lr = self._get_current_lr()

        # Truncate history to current epoch position (handle resume overlaps)
        abs_epoch_idx = epoch  # 0-indexed position in history
        for k in self.history:
            if len(self.history[k]) > abs_epoch_idx:
                self.history[k] = self.history[k][:abs_epoch_idx]

        # Append new epoch data
        acc = float(logs.get("accuracy",
                    logs.get("sparse_categorical_accuracy", 0.0)))
        self.history["train_loss"].append(float(logs.get("loss", 0.0)))
        self.history["val_loss"].append(float(logs.get("val_loss", 0.0)))
        self.history["train_acc"].append(acc)
        self.history["val_acc"].append(val_acc)
        self.history["lr"].append(current_lr)
        self.history["margin"].append(current_margin)

        # Build meta dict
        meta = {
            "last_epoch": epoch,
            "next_epoch": epoch + 1,
            "latest_checkpoint": latest_path,
            "best_checkpoint": best_path or self.best_manager.latest_checkpoint,
            "loss": float(logs.get("loss", 0.0)),
            "accuracy": acc,
            "val_loss": float(logs.get("val_loss", 0.0)),
            "val_accuracy": val_acc,
            "best_val_accuracy": self.best_val_accuracy,
            "current_margin": current_margin,
            "current_lr": current_lr,
            "num_classes": self.num_classes,
            "history": self.history,
        }
        # Append any extra metrics (e.g., val_eer from VerificationCallback)
        for k, v in logs.items():
            if k not in meta and isinstance(v, (float, int)):
                meta[k] = float(v)

        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[Checkpoint] Epoch {epoch+1}: saved to {latest_path} "
              f"(margin={current_margin:.2f}, lr={current_lr:.6f})")


def read_resume_meta(meta_path):
    if not os.path.exists(meta_path):
        return {"next_epoch": 0}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        print(f"[Resume Warning] Could not parse metadata: {e}. Starting from scratch.")
        return {"next_epoch": 0}
    next_epoch = int(meta.get("next_epoch", int(meta.get("last_epoch", -1)) + 1))
    meta["next_epoch"] = max(next_epoch, 0)
    return meta


def verify_checkpoint_integrity(checkpoint_path):
    """Verify checkpoint files exist before attempting restore."""
    required_files = [
        checkpoint_path + '.index',
        checkpoint_path + '.data-00000-of-00001',
    ]
    for f in required_files:
        if not os.path.exists(f):
            raise FileNotFoundError(
                f"Checkpoint file missing: {f}\n"
                f"Cannot resume — need to retrain from scratch."
            )
    print(f"\u2713 Checkpoint files verified: {checkpoint_path}")


def make_full_checkpoint(training_model, backbone, loss_layer, optimizer):
    epoch_var = tf.Variable(0, dtype=tf.int64, trainable=False, name="resume_epoch")
    checkpoint = tf.train.Checkpoint(
        model=training_model,
        backbone=backbone,
        loss_layer=loss_layer,
        optimizer=optimizer,
        epoch=epoch_var,
        # margin_var is inside loss_layer, tracked via loss_layer checkpoint
    )
    return checkpoint, epoch_var


# =====================================================================
# Data Loading Helper
# =====================================================================

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


# =====================================================================
# Main Training Function
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="iResNet50 ArcFace Trainer (Paper Config)")
    parser.add_argument("--dataset_dir", type=str, default="dataset_final/train",
                        help="Training directory (class subfolders)")
    parser.add_argument("--val_dir", type=str, default="dataset_final/val",
                        help="Validation directory (SAME classes as train)")
    parser.add_argument("--test_dir", type=str, default="dataset_final/test",
                        help="Test directory (UNSEEN classes, for verification eval)")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size (128 recommended for VRAM >= 8GB, 64 if < 8GB)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Total epochs (30 per ArcFace paper on VGGFace2)")
    parser.add_argument("--img_size", type=int, default=112)
    parser.add_argument("--embedding_dim", type=int, default=512,
                        help="Embedding dimension (512 per ArcFace paper)")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout rate (0.5 per ArcFace paper BN-Dropout-FC-BN spec)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Initial learning rate for SGD (0.01 safe start)")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="L2 weight decay for SGD (5e-4 per ArcFace paper)")
    parser.add_argument("--arcface_scale", type=float, default=64.0,
                        help="ArcFace scale factor (s=64.0 per paper, fixed)")
    parser.add_argument("--loss_type", type=str, default="arcface",
                        choices=["arcface", "cosface", "sphereface", "softmax"],
                        help="Loss function: arcface | cosface | sphereface | softmax")
    parser.add_argument("--margin_step", type=float, default=0.1,
                        help="Margin increment per step (0.1 per spec)")
    parser.add_argument("--margin_step_every", type=int, default=5,
                        help="Epochs per margin step (5 per spec)")
    parser.add_argument("--milestones", type=str, default="16,22,26,30",
                        help="LR step decay milestones (comma-separated epoch numbers)")
    parser.add_argument("--gamma", type=float, default=0.1,
                        help="LR decay factor at each milestone")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--verify_every", type=int, default=3,
                        help="Run verification eval every N epochs")
    parser.add_argument("--verify_pairs", type=int, default=25000,
                        help="Number of pairs for verification eval (25K per spec)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from previous checkpoint if available")
    parser.add_argument("--elastic_face", action="store_true", default=False,
                        help="Use ElasticFace stochastic margin (Boutros et al. 2021)")
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Label smoothing factor (0.1 recommended, 0.0 to disable)")
    parser.add_argument("--warmup_epochs", type=int, default=2,
                        help="Linear LR warmup epochs (0 to disable)")
    parser.add_argument("--clipnorm", type=float, default=0.0,
                        help="Gradient clipping by norm (0 to disable, 1.0 recommended)")
    parser.add_argument("--use_erasing", action="store_true", default=False,
                        help="Enable Random Erasing / Cutout augmentation")
    parser.add_argument("--use_blur", action="store_true", default=False,
                        help="Enable random blur augmentation (p=0.2)")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Custom results directory (default: results/{loss_type}/ under model dir)")
    args, unknown = parser.parse_known_args()

    # Parse milestones
    milestones = [int(m.strip()) for m in args.milestones.split(",")]

    # Results go into the model's local results folder, organized by loss type
    if args.results_dir:
        results_dir = args.results_dir
    else:
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

    if len(val_paths) == 0:
        print("[Data] WARNING: val_dir has 0 overlapping classes with train_dir.")
        print("[Data] This happens when train/val identities are disjoint (e.g. MS1M dataset).")
        print("[Data] Will re-split: using 10% of train identities as val (shared classes).")
        # Auto-split: take last 10% of train classes as val (they share the class_to_idx)
        rng = __import__('random')
        rng.seed(42)
        train_ids = sorted(train_class_to_idx.keys())
        rng.shuffle(train_ids)
        split_n = max(1, int(len(train_ids) * 0.1))
        val_ids = set(train_ids[:split_n])
        # Rebuild train_paths without val_ids
        new_train_paths, new_train_labels = [], []
        new_val_paths, new_val_labels = [], []
        for p, lbl in zip(train_paths, train_labels):
            # Find which class name this label belongs to
            cls_name = train_class_names[lbl] if lbl < len(train_class_names) else None
            if cls_name in val_ids:
                new_val_paths.append(p)
                new_val_labels.append(lbl)
            else:
                new_train_paths.append(p)
                new_train_labels.append(lbl)
        train_paths, train_labels = new_train_paths, new_train_labels
        val_paths, val_labels = new_val_paths, new_val_labels
        print(f"[Data] After re-split: {len(train_paths)} train, {len(val_paths)} val "
              f"({len(val_ids)} shared classes)")

    # Build tf.data pipeline (with heavy augmentation + Random Erasing on train only)
    train_ds = build_training_dataset(train_paths, train_labels, batch_size=args.batch_size,
                                      is_training=True, use_erasing=args.use_erasing,
                                      use_blur=args.use_blur)
    val_ds = build_training_dataset(val_paths, val_labels, batch_size=args.batch_size,
                                    is_training=False) if val_paths else None

    num_train_batches = math.ceil(len(train_paths) / args.batch_size)

    # Check for resume option
    initial_epoch = 0
    start_step = 0.0
    resume_meta = {}
    checkpoint_path = os.path.join(results_dir, "best_iresnet50.h5")
    latest_checkpoint_dir = os.path.join(results_dir, "checkpoints", "latest")
    best_checkpoint_dir = os.path.join(results_dir, "checkpoints", "best")
    os.makedirs(best_checkpoint_dir, exist_ok=True)
    meta_path = os.path.join(results_dir, "training_meta.json")
    latest_full_checkpoint = tf.train.latest_checkpoint(latest_checkpoint_dir)

    if args.resume:
        resume_meta = read_resume_meta(meta_path)
        initial_epoch = int(resume_meta.get("next_epoch", 0))
        if latest_full_checkpoint:
            # Verify checkpoint integrity before attempting restore
            verify_checkpoint_integrity(latest_full_checkpoint)
            start_step = 0.0
            print(f"[Resume] Full checkpoint found: {latest_full_checkpoint}")
        elif os.path.exists(checkpoint_path):
            start_step = float(initial_epoch * num_train_batches)
            print(f"[Resume Fallback] No full checkpoint. Will load weights from {checkpoint_path}")
        else:
            print("[Resume Warning] No checkpoint found. Starting from scratch.")
            initial_epoch = 0
            start_step = 0.0

    if initial_epoch >= args.epochs:
        print(f"[Resume] Training already completed up to {args.epochs} epochs. Nothing to do!")
        return

    # Build model
    img_shape = (args.img_size, args.img_size, 3)
    training_model, backbone, loss_layer = build_training_model(
        input_shape=img_shape,
        num_classes=num_classes,
        embedding_dim=args.embedding_dim,
        dropout_rate=args.dropout,
        arcface_scale=args.arcface_scale,
        loss_type=args.loss_type,
        elastic=args.elastic_face,
    )
    backbone.summary()

    # Learning Rate Schedule — Step Decay (per ArcFace paper) with optional warmup
    lr_schedule = StepDecaySchedule(
        initial_lr=args.lr,
        milestones_epochs=milestones,
        gamma=args.gamma,
        steps_per_epoch=num_train_batches,
        start_step=start_step,
        warmup_epochs=args.warmup_epochs,
    )

    # Optimizer — SGD with Nesterov momentum + weight decay (per ArcFace paper, NOT Adam)
    sgd_kwargs = {
        "learning_rate": lr_schedule,
        "momentum": 0.9,
        "nesterov": True,
    }
    if args.clipnorm > 0.0:
        sgd_kwargs["clipnorm"] = args.clipnorm
    try:
        # TF 2.11+ experimental SGD supports weight_decay natively
        optimizer = tf.keras.optimizers.experimental.SGD(
            weight_decay=args.weight_decay, **sgd_kwargs)
    except (AttributeError, TypeError):
        # Fallback: legacy SGD (no weight_decay param)
        optimizer = tf.keras.optimizers.SGD(**sgd_kwargs)
        print("[Optimizer] weight_decay not supported by this Keras version — "
              "applied via kernel_regularizer in model layers instead.")
    print(f"[Optimizer] SGD(lr={args.lr}, momentum=0.9, nesterov=True, weight_decay={args.weight_decay})")
    if args.clipnorm > 0.0:
        print(f"[Gradient Clip] clipnorm={args.clipnorm}")
    if args.warmup_epochs > 0:
        print(f"[LR Warmup] {args.warmup_epochs} epochs")
    print(f"[LR Schedule] Step Decay: milestones={milestones}, gamma={args.gamma}")

    # Loss: CategoricalCrossentropy with label_smoothing for better generalization
    # Requires one-hot labels — the data pipeline already provides labels as integers,
    # so we convert to one-hot inside a custom loss wrapper.
    cce_loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, label_smoothing=args.label_smoothing
    )
    def label_smoothed_loss(y_true, y_pred):
        """Wraps CategoricalCrossentropy: converts sparse labels to one-hot."""
        y_true_onehot = tf.one_hot(tf.cast(tf.squeeze(y_true), tf.int32), depth=num_classes)
        return cce_loss(y_true_onehot, y_pred)

    training_model.compile(
        optimizer=optimizer,
        loss=label_smoothed_loss,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    full_checkpoint, checkpoint_epoch = make_full_checkpoint(
        training_model, backbone, loss_layer, optimizer,
    )
    latest_manager = tf.train.CheckpointManager(
        full_checkpoint, latest_checkpoint_dir, max_to_keep=2,
    )
    best_manager = tf.train.CheckpointManager(
        full_checkpoint, best_checkpoint_dir, max_to_keep=1,
    )

    # Restore after compile so optimizer state can be matched/deferred correctly.
    if args.resume and (initial_epoch > 0 or latest_full_checkpoint):
        if latest_full_checkpoint:
            full_checkpoint.restore(latest_full_checkpoint).expect_partial()
            restored_epoch = int(checkpoint_epoch.numpy())
            if restored_epoch > 0:
                initial_epoch = min(restored_epoch, args.epochs)
            # Restore margin from meta if available
            if 'current_margin' in resume_meta:
                loss_layer._update_margin_constants(resume_meta['current_margin'])
            # Verification prints
            print("\n" + "-" * 50)
            print("[Resume] Verification after restore:")
            try:
                cur_lr = optimizer.learning_rate(optimizer.iterations).numpy()
            except Exception:
                cur_lr = float(optimizer.learning_rate)
            print(f"  \u2713 Next epoch:          {initial_epoch + 1}")
            print(f"  \u2713 Margin restored to:  {float(loss_layer.margin_var.numpy()):.2f}")
            print(f"  \u2713 Current LR:          {cur_lr:.6f}")
            print(f"  \u2713 Optimizer iters:     {int(optimizer.iterations.numpy())}")
            print("-" * 50)
        elif os.path.exists(checkpoint_path):
            try:
                training_model.load_weights(checkpoint_path)
                print("[Resume Fallback] Loaded legacy weights only. No optimizer state.")
                # Manually advance optimizer step to match resumed epoch
                if start_step > 0:
                    optimizer.iterations.assign(int(start_step))
                    print(f"  Set optimizer.iterations to {int(start_step)} "
                          f"(epoch {initial_epoch}, lr={optimizer.learning_rate(optimizer.iterations).numpy():.6f})")
            except Exception as e:
                print(f"[Resume Error] {e}. Starting from scratch.")
                initial_epoch = 0
                start_step = 0.0

    # Determine end margin for progressive scheduling
    end_margin = DEFAULT_MARGINS.get(args.loss_type, 0.5)

    callbacks = [
        ModelCheckpoint(
            checkpoint_path,
            save_best_only=True,
            save_weights_only=True,
            monitor="val_accuracy",
            mode="max",
            verbose=1,
        ),
        SaveBestBackbone(
            backbone,
            os.path.join(results_dir, "best_iresnet50_backbone.h5"),
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
        TrainingCheckpointCallback(
            latest_manager=latest_manager,
            best_manager=best_manager,
            epoch_var=checkpoint_epoch,
            meta_path=meta_path,
            loss_layer=loss_layer,
            optimizer=optimizer,
            num_classes=num_classes,
        ),
    ]

    # Progressive margin only for ArcFace and CosFace (additive margins)
    # SphereFace uses fixed multiplicative margin, Softmax has no margin
    if args.loss_type in ("arcface", "cosface"):
        callbacks.insert(2, ProgressiveMarginCallback(
            loss_layer,
            max_margin=end_margin,
            step_size=args.margin_step,
            step_every=args.margin_step_every,
        ))

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
            )
        )
        print(f"[Verification] Enabled on {args.test_dir} (every {args.verify_every} epochs, {args.verify_pairs} pairs)")

    print("\n" + "=" * 60)
    print(f"Training Config (iResNet50 + {args.loss_type.upper()} Head)")
    print(f"  Loss Type:     {args.loss_type}")
    print(f"  Label Smooth:  {args.label_smoothing}")
    print(f"  ElasticFace:   {args.elastic_face}")
    print(f"  Max Margin:    {end_margin}")
    print(f"  Margin Step:   +{args.margin_step} every {args.margin_step_every} epochs")
    print(f"  Classes:       {num_classes}")
    print(f"  Img Size:      {args.img_size}")
    print(f"  Batch Size:    {args.batch_size}")
    print(f"  Epochs:        {args.epochs} (starting from {initial_epoch + 1})")
    print(f"  Optimizer:     SGD (momentum=0.9, nesterov=True)")
    print(f"  Initial LR:    {args.lr}")
    print(f"  LR Warmup:     {args.warmup_epochs} epochs" if args.warmup_epochs > 0 else "  LR Warmup:     disabled")
    print(f"  LR Milestones: {milestones}")
    print(f"  Weight Decay:  {args.weight_decay}")
    print(f"  Clip Norm:     {args.clipnorm}" if args.clipnorm > 0 else "  Clip Norm:     disabled")
    print(f"  Scale (s):     {args.arcface_scale}")
    print(f"  Dropout:       {args.dropout}")
    print(f"  Patience:      {args.patience}")
    print(f"  Embedding:     {args.embedding_dim}")
    print(f"  Params:        {training_model.count_params():,}")
    print("=" * 60 + "\n")

    history = training_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        initial_epoch=initial_epoch,
    )
    plot_history(history, os.path.join(results_dir, "training_history.png"))
    print(f"\n[SUCCESS] iResNet50 Training complete. Results saved in {results_dir}/")


if __name__ == "__main__":
    main()
