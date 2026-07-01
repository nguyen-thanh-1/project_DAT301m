"""
HybridFaceNet training script.

Two-phase strategy:
  Phase 1: epochs 1-5 train CNN branch + ArcFace head with FCU gates disabled.
  Phase 2: epochs 6-30 train the full CNN/Transformer/FCU/Fusion model.
"""

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from models.face_recognition.hybridfacenet.hybridfacenet import (
    ArcFaceLayer,
    CouplingAdd,
    build_ablation_models,
    build_training_model,
)
from shared.utils import (
    HistoryPlotterCallback,
    SaveBestBackbone,
    VerificationCallback,
    configure_gpu,
    plot_history,
)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ProgressiveMarginCallback(tf.keras.callbacks.Callback):
    """Step schedule: 0.1, 0.2, 0.3, 0.4, 0.5 every 5 epochs."""

    def __init__(self, margin_layer, max_margin=0.5, step_size=0.1, step_every=5):
        super().__init__()
        self.margin_layer = margin_layer
        self.max_margin = max_margin
        self.step_size = step_size
        self.step_every = step_every

    def on_epoch_begin(self, epoch, logs=None):
        step_num = (epoch // self.step_every) + 1
        margin = min(step_num * self.step_size, self.max_margin)
        self.margin_layer._update_margin_constants(margin)
        print(f"\n[Margin] epoch={epoch + 1} margin={margin:.2f}")


class WarmupCosineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warm-up followed by cosine annealing."""

    def __init__(self, warmup_epochs, total_epochs, steps_per_epoch, start_lr, peak_lr, min_lr, start_epoch=0):
        super().__init__()
        self.warmup_steps = float(warmup_epochs * steps_per_epoch)
        self.total_steps = float(total_epochs * steps_per_epoch)
        self.steps_per_epoch = float(steps_per_epoch)
        self.start_lr = float(start_lr)
        self.peak_lr = float(peak_lr)
        self.min_lr = float(min_lr)
        self.start_step = float(start_epoch * steps_per_epoch)

    def __call__(self, step):
        step = tf.cast(step, tf.float32) + self.start_step
        warmup_progress = step / tf.maximum(self.warmup_steps, 1.0)
        warmup_lr = self.start_lr + (self.peak_lr - self.start_lr) * warmup_progress

        cosine_step = tf.maximum(step - self.warmup_steps, 0.0)
        cosine_total = tf.maximum(self.total_steps - self.warmup_steps, 1.0)
        progress = tf.minimum(cosine_step / cosine_total, 1.0)
        cosine_lr = self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (
            1.0 + tf.cos(math.pi * progress)
        )
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "start_lr": self.start_lr,
            "peak_lr": self.peak_lr,
            "min_lr": self.min_lr,
            "start_step": self.start_step,
        }


class SaveTrainingMetaCallback(tf.keras.callbacks.Callback):
    def __init__(self, filepath, phase, checkpoint_path=None):
        super().__init__()
        self.filepath = filepath
        self.phase = phase
        self.checkpoint_path = checkpoint_path

    def on_epoch_end(self, epoch, logs=None):
        meta = {
            "last_epoch": epoch,
            "next_epoch": epoch + 1,
            "phase": self.phase,
            "checkpoint_path": self.checkpoint_path,
        }
        if logs:
            meta.update({k: float(v) for k, v in logs.items() if isinstance(v, (float, int, np.floating))})
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)


class LatestCheckpointCallback(tf.keras.callbacks.Callback):
    """Saves a full resumable checkpoint after every completed epoch."""

    def __init__(self, checkpoint_manager, epoch_var, meta_path, phase):
        super().__init__()
        self.checkpoint_manager = checkpoint_manager
        self.epoch_var = epoch_var
        self.meta_path = meta_path
        self.phase = phase

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_var.assign(epoch + 1)
        checkpoint_path = self.checkpoint_manager.save(checkpoint_number=epoch + 1)
        meta = {
            "last_epoch": epoch,
            "next_epoch": epoch + 1,
            "phase": self.phase,
            "latest_checkpoint": checkpoint_path,
        }
        if logs:
            meta.update({k: float(v) for k, v in logs.items() if isinstance(v, (float, int, np.floating))})
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[Checkpoint] Saved resumable checkpoint: {checkpoint_path}")


class FusionAlphaLogger(tf.keras.callbacks.Callback):
    def __init__(self, backbone, save_path):
        super().__init__()
        self.backbone = backbone
        self.save_path = save_path
        self.history = []
        if os.path.exists(save_path):
            try:
                with open(save_path, "r", encoding="utf-8") as f:
                    self.history = json.load(f)
            except Exception:
                self.history = []

    def on_epoch_end(self, epoch, logs=None):
        fusion = self.backbone.get_layer("fusion")
        alpha = float(fusion.alpha.numpy())
        self.history = [item for item in self.history if int(item.get("epoch", 0)) <= epoch]
        self.history.append({"epoch": epoch + 1, "alpha": alpha})
        if logs is not None:
            logs["fusion_alpha"] = alpha
        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
        print(f"[Fusion] alpha={alpha:.4f}")


class BranchAccuracyCallback(tf.keras.callbacks.Callback):
    """Logs CNN-only, Trans-only and Fused validation accuracy on a bounded sample."""

    def __init__(self, backbone, arcface_layer, val_ds, max_batches=20):
        super().__init__()
        self.backbone = backbone
        self.arcface_layer = arcface_layer
        self.val_ds = val_ds
        self.max_batches = max_batches

    def on_epoch_end(self, epoch, logs=None):
        if self.max_batches <= 0:
            return
        correct = {"fused": 0, "cnn": 0, "trans": 0}
        total = 0
        for batch_idx, ((images, labels), _) in enumerate(self.val_ds.take(self.max_batches)):
            fused, cnn, trans = self.backbone(images, training=False)
            for name, embedding in [("fused", fused), ("cnn", cnn), ("trans", trans)]:
                logits = self.arcface_layer([embedding, labels])
                pred = tf.argmax(logits, axis=1, output_type=tf.int32)
                correct[name] += int(tf.reduce_sum(tf.cast(pred == tf.cast(labels, tf.int32), tf.int32)).numpy())
            total += int(labels.shape[0])
        if total == 0:
            return
        values = {name: correct[name] / total for name in correct}
        if logs is not None:
            logs.update({f"val_{name}_accuracy": value for name, value in values.items()})
        print(
            "[Branch Val] "
            f"fused={values['fused']:.4f} cnn={values['cnn']:.4f} trans={values['trans']:.4f}"
        )


def set_fcu_gates(backbone, value):
    for layer in backbone.layers:
        if isinstance(layer, CouplingAdd):
            layer.gate.assign(float(value))


def set_phase_trainability(backbone, phase):
    """Freeze Transformer/FCU/Fusion during CNN warm-up; unfreeze all for joint training."""
    if phase == "cnn_warmup":
        for layer in backbone.layers:
            name = layer.name
            layer.trainable = not (
                name.startswith("trans_")
                or name.startswith("fcu_")
                or name in {"cls_token", "pos_embedding", "patch_tokenizer_conv", "fusion"}
            )
        set_fcu_gates(backbone, 0.0)
    else:
        for layer in backbone.layers:
            layer.trainable = True
        set_fcu_gates(backbone, 1.0)


def read_resume_state(meta_path):
    if not os.path.exists(meta_path):
        return {"next_epoch": 0, "phase": None}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as exc:
        print(f"[Resume Warning] Could not read {meta_path}: {exc}")
        return {"next_epoch": 0, "phase": None}
    next_epoch = int(meta.get("next_epoch", int(meta.get("last_epoch", -1)) + 1))
    meta["next_epoch"] = max(next_epoch, 0)
    return meta


def make_checkpoint(backbone, arcface_layer, optimizer=None):
    epoch_var = tf.Variable(0, dtype=tf.int64, trainable=False, name="resume_epoch")
    if optimizer is None:
        checkpoint = tf.train.Checkpoint(backbone=backbone, arcface=arcface_layer, epoch=epoch_var)
    else:
        checkpoint = tf.train.Checkpoint(
            backbone=backbone,
            arcface=arcface_layer,
            optimizer=optimizer,
            epoch=epoch_var,
        )
    return checkpoint, epoch_var


def restore_latest_checkpoint(
    checkpoint_dir,
    backbone,
    arcface_layer,
    optimizer=None,
    include_optimizer=True,
    required=False,
):
    latest = tf.train.latest_checkpoint(str(checkpoint_dir))
    if not latest:
        if required:
            raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
        return None, 0

    checkpoint, epoch_var = make_checkpoint(
        backbone,
        arcface_layer,
        optimizer=optimizer if include_optimizer else None,
    )
    checkpoint.restore(latest).expect_partial()
    restored_epoch = int(epoch_var.numpy())
    print(
        f"[Resume] Restored {'full' if include_optimizer else 'weights-only'} "
        f"checkpoint from {latest} (next_epoch={restored_epoch})"
    )
    return latest, restored_epoch


def build_phase_callbacks(
    backbone,
    arcface_layer,
    val_ds,
    args,
    results_dir,
    phase,
    checkpoint_manager,
    epoch_var,
    include_best=False,
):
    callbacks = [
        ProgressiveMarginCallback(arcface_layer),
        HistoryPlotterCallback(str(results_dir / "training_history.png")),
        FusionAlphaLogger(backbone, str(results_dir / "fusion_alpha_history.json")),
        BranchAccuracyCallback(backbone, arcface_layer, val_ds, max_batches=args.branch_val_batches),
        LatestCheckpointCallback(
            checkpoint_manager,
            epoch_var,
            str(results_dir / "training_meta.json"),
            phase=phase,
        ),
    ]

    if include_best:
        callbacks = [
            ModelCheckpoint(
                str(results_dir / "best_hybridfacenet.h5"),
                monitor="val_accuracy",
                mode="max",
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
            ),
            SaveBestBackbone(backbone, str(results_dir / "best_hybridfacenet_backbone.h5")),
            EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True, verbose=1),
        ] + callbacks

    if args.test_dir and os.path.isdir(args.test_dir):
        ablations = build_ablation_models(backbone)
        callbacks.append(
            VerificationCallback(
                ablations["fused"],
                test_dir=args.test_dir,
                num_pairs=args.verify_pairs,
                eval_every=args.verify_every,
                batch_size=args.batch_size,
                save_dir=str(results_dir),
                file_prefix="fused",
            )
        )

    return callbacks


def get_paths_and_labels(dataset_dir, class_to_idx=None):
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        return [], [], 0 if class_to_idx is None else len(class_to_idx)
    if class_to_idx is None:
        class_names = sorted(p.name for p in dataset_dir.iterdir() if p.is_dir())
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    image_paths, labels = [], []
    for class_name, idx in class_to_idx.items():
        class_dir = dataset_dir / class_name
        if not class_dir.is_dir():
            continue
        for img_path in class_dir.iterdir():
            if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTS:
                image_paths.append(str(img_path))
                labels.append(idx)
    return image_paths, labels, len(class_to_idx)


def build_raw_dataset(image_paths, labels, batch_size, is_training):
    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    if is_training:
        ds = ds.shuffle(len(image_paths), reshuffle_each_iteration=True)

    def load_img(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, [112, 112])
        img.set_shape([112, 112, 3])
        img = tf.cast(img, tf.float32)
        return img, tf.cast(label, tf.int32)

    def augment(img, label):
        img = img / 255.0
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.2)
        img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
        img = tf.image.random_hue(img, max_delta=0.05)
        img = tf.clip_by_value(img, 0.0, 1.0) * 255.0
        return (img, label), label

    def format_val(img, label):
        return (img, label), label

    ds = ds.map(load_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()
    ds = ds.map(augment if is_training else format_val, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=is_training)
    return ds.prefetch(tf.data.AUTOTUNE)


def make_sgd(learning_rate, weight_decay):
    try:
        return tf.keras.optimizers.experimental.SGD(
            learning_rate=learning_rate,
            momentum=0.9,
            nesterov=True,
            weight_decay=weight_decay,
        )
    except (AttributeError, TypeError):
        return tf.keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=0.9,
            nesterov=True,
        )


def build_branch_training_model(backbone, arcface_layer, input_shape, branch):
    face_input = tf.keras.Input(shape=input_shape, name=f"{branch}_face_input")
    label_input = tf.keras.Input(shape=(), dtype=tf.int32, name=f"{branch}_label_input")
    fused, cnn, trans = backbone(face_input)
    embedding = {"fused": fused, "cnn": cnn, "trans": trans}[branch]
    logits = arcface_layer([embedding, label_input])
    return tf.keras.Model([face_input, label_input], logits, name=f"HybridFaceNet_{branch}_Training")


def main():
    parser = argparse.ArgumentParser(description="HybridFaceNet Trainer")
    parser.add_argument("--dataset_dir", type=str, default="dataset_final/train")
    parser.add_argument("--val_dir", type=str, default="dataset_final/val")
    parser.add_argument("--test_dir", type=str, default="dataset_final/test")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--warmup_lr", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--lr_min", type=float, default=1e-5)
    parser.add_argument("--lr_start", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--arcface_scale", type=float, default=64.0)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--verify_every", type=int, default=5)
    parser.add_argument("--verify_pairs", type=int, default=5000)
    parser.add_argument("--branch_val_batches", type=int, default=20)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest full checkpoint, including optimizer state and epoch.",
    )
    args, _ = parser.parse_known_args()

    configure_gpu()

    train_root = Path(args.dataset_dir)
    class_names = sorted(p.name for p in train_root.iterdir() if p.is_dir())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    train_paths, train_labels, num_classes = get_paths_and_labels(args.dataset_dir, class_to_idx)
    val_paths, val_labels, _ = get_paths_and_labels(args.val_dir, class_to_idx)
    if not train_paths:
        raise SystemExit(f"No training images found in {args.dataset_dir}")

    print(f"[Data] train={len(train_paths)} val={len(val_paths)} classes={num_classes}")
    train_ds = build_raw_dataset(train_paths, train_labels, args.batch_size, is_training=True)
    val_ds = build_raw_dataset(val_paths, val_labels, args.batch_size, is_training=False)
    steps_per_epoch = math.ceil(len(train_paths) / args.batch_size)

    results_dir = Path(__file__).resolve().parent / "results" / "arcface"
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root = results_dir / "checkpoints"
    warmup_checkpoint_dir = checkpoint_root / "warmup"
    joint_checkpoint_dir = checkpoint_root / "joint"
    meta_path = results_dir / "training_meta.json"

    training_model, backbone, arcface_layer = build_training_model(
        input_shape=(112, 112, 3),
        num_classes=num_classes,
        embedding_dim=args.embedding_dim,
        arcface_scale=args.arcface_scale,
        dropout_rate=args.dropout,
    )
    cnn_training_model = build_branch_training_model(backbone, arcface_layer, (112, 112, 3), "cnn")

    print(f"[Model] params={training_model.count_params():,}")
    backbone.summary()

    resume_state = read_resume_state(str(meta_path)) if args.resume else {"next_epoch": 0, "phase": None}
    resume_epoch = min(int(resume_state.get("next_epoch", 0)), args.epochs)
    if args.resume:
        print(
            f"[Resume] requested. phase={resume_state.get('phase')} "
            f"next_epoch={resume_epoch}"
        )
        if resume_epoch >= args.epochs:
            print(f"[Resume] Training already reached requested epochs={args.epochs}. Nothing to do.")
            return

    if args.warmup_epochs > 0 and resume_epoch < min(args.warmup_epochs, args.epochs):
        print("\n[Phase 1] CNN warm-up")
        set_phase_trainability(backbone, "cnn_warmup")
        warmup_optimizer = make_sgd(args.warmup_lr, args.weight_decay)
        cnn_training_model.compile(
            optimizer=warmup_optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
        warmup_checkpoint, warmup_epoch_var = make_checkpoint(
            backbone,
            arcface_layer,
            optimizer=warmup_optimizer,
        )
        warmup_manager = tf.train.CheckpointManager(
            warmup_checkpoint,
            str(warmup_checkpoint_dir),
            max_to_keep=3,
        )
        if args.resume:
            _, restored_epoch = restore_latest_checkpoint(
                warmup_checkpoint_dir,
                backbone,
                arcface_layer,
                optimizer=warmup_optimizer,
                include_optimizer=True,
            )
            if restored_epoch:
                resume_epoch = min(restored_epoch, args.warmup_epochs)

        cnn_training_model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=resume_epoch,
            epochs=min(args.warmup_epochs, args.epochs),
            callbacks=build_phase_callbacks(
                backbone,
                arcface_layer,
                val_ds,
                args,
                results_dir,
                phase="cnn_warmup",
                checkpoint_manager=warmup_manager,
                epoch_var=warmup_epoch_var,
                include_best=False,
            ),
        )
        resume_epoch = min(args.warmup_epochs, args.epochs)

    if args.epochs > args.warmup_epochs and resume_epoch < args.epochs:
        print("\n[Phase 2] Joint training")
        if args.resume and not tf.train.latest_checkpoint(str(joint_checkpoint_dir)):
            restore_latest_checkpoint(
                warmup_checkpoint_dir,
                backbone,
                arcface_layer,
                optimizer=None,
                include_optimizer=False,
            )

        set_phase_trainability(backbone, "joint")
        lr_schedule = WarmupCosineSchedule(
            warmup_epochs=3,
            total_epochs=max(args.epochs - args.warmup_epochs, 1),
            steps_per_epoch=steps_per_epoch,
            start_lr=args.lr_start,
            peak_lr=args.lr,
            min_lr=args.lr_min,
        )
        joint_optimizer = make_sgd(lr_schedule, args.weight_decay)
        training_model.compile(
            optimizer=joint_optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
        joint_checkpoint, joint_epoch_var = make_checkpoint(
            backbone,
            arcface_layer,
            optimizer=joint_optimizer,
        )
        joint_manager = tf.train.CheckpointManager(
            joint_checkpoint,
            str(joint_checkpoint_dir),
            max_to_keep=3,
        )
        if args.resume:
            _, restored_epoch = restore_latest_checkpoint(
                joint_checkpoint_dir,
                backbone,
                arcface_layer,
                optimizer=joint_optimizer,
                include_optimizer=True,
            )
            if restored_epoch:
                resume_epoch = max(restored_epoch, args.warmup_epochs)
        initial_epoch = max(resume_epoch, args.warmup_epochs)

        history = training_model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=initial_epoch,
            epochs=args.epochs,
            callbacks=build_phase_callbacks(
                backbone,
                arcface_layer,
                val_ds,
                args,
                results_dir,
                phase="joint",
                checkpoint_manager=joint_manager,
                epoch_var=joint_epoch_var,
                include_best=True,
            ),
        )
        plot_history(history, str(results_dir / "training_history.png"))

    print(f"[Done] HybridFaceNet results saved to {results_dir}")


if __name__ == "__main__":
    main()
