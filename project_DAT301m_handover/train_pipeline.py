import os
import time
import json
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path

from anti_spoofing.config import Config, setup_dirs
from anti_spoofing.dataset import discover_dataset, build_dataset
from anti_spoofing.face_detector import build_crop_cache
from anti_spoofing.models import (
    build_efficientnet_style, 
    build_cdcn_lbp, 
    build_resnet18_cbam, 
    build_resnet34_se_cdc_multiscale
)
from anti_spoofing.loss import focal_loss_from_logits
from anti_spoofing.metrics import full_benchmark

# Set seed for reproducibility
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
import random
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def configure_gpu_and_precision():
    print("TensorFlow:", tf.__version__)
    gpus = tf.config.list_physical_devices("GPU")
    print("GPUs found:", gpus)
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
            print(f"Memory growth enabled for GPU: {g}")
        except RuntimeError as e:
            print(e)

    USE_MIXED_PRECISION = True
    if USE_MIXED_PRECISION:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision enabled:", tf.keras.mixed_precision.global_policy())

def cosine_warmup_lr(step, total_steps, warmup_steps, base_lr, min_lr):
    step = tf.cast(step, tf.float32)
    warmup_steps = tf.cast(max(1, warmup_steps), tf.float32)
    total_steps = tf.cast(max(1, total_steps), tf.float32)

    warmup_lr = base_lr * (step / warmup_steps)
    progress = (step - warmup_steps) / tf.maximum(1.0, total_steps - warmup_steps)
    cosine_lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + tf.cos(np.pi * tf.clip_by_value(progress, 0, 1)))
    return tf.where(step < warmup_steps, warmup_lr, cosine_lr)

def make_train_step(model, optimizer, alpha_weights, cfg, uses_lbp):
    @tf.function
    def train_step(batch_inputs, batch_labels):
        with tf.GradientTape() as tape:
            inputs = batch_inputs if uses_lbp else {"rgb": batch_inputs["rgb"]}
            logits = model(inputs, training=True)
            loss = focal_loss_from_logits(batch_labels, logits, alpha_weights,
                                           gamma=cfg.focal_gamma,
                                           label_smoothing=cfg.label_smoothing,
                                           num_classes=cfg.num_classes)
            scaled_loss = loss
        grads = tape.gradient(scaled_loss, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, cfg.grad_clip_norm)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        probs = tf.nn.softmax(tf.cast(logits, tf.float32), axis=-1)
        return loss, probs
    return train_step

def run_eval(model, dataset, uses_lbp):
    all_scores, all_labels = [], []
    for batch_inputs, batch_labels in dataset:
        inputs = batch_inputs if uses_lbp else {"rgb": batch_inputs["rgb"]}
        logits = model(inputs, training=False)
        probs = tf.nn.softmax(tf.cast(logits, tf.float32), axis=-1)
        all_scores.append(probs[:, 1].numpy())   # P(real)
        all_labels.append(batch_labels.numpy())
    return np.concatenate(all_scores), np.concatenate(all_labels)

def train_model(model, model_name, train_ds, dev_ds, cfg: Config, alpha_weights, uses_lbp: bool, steps_per_epoch: int):
    ckpt_path = Path(cfg.ckpt_dir) / model_name
    ckpt_path.mkdir(parents=True, exist_ok=True)
    log_csv_path = Path(cfg.log_dir) / f"{model_name}_train_log.csv"
    tb_writer = tf.summary.create_file_writer(str(Path(cfg.log_dir) / model_name / "tb"))

    optimizer = tf.keras.optimizers.AdamW(learning_rate=cfg.lr, weight_decay=cfg.weight_decay)
    
    # Check if mixed precision is enabled and scale optimizer accordingly
    if tf.keras.mixed_precision.global_policy().name.startswith("mixed"):
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

    train_step_fn = make_train_step(model, optimizer, alpha_weights, cfg, uses_lbp)

    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = steps_per_epoch * cfg.warmup_epochs

    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer, step=tf.Variable(0))
    ckpt_manager = tf.train.CheckpointManager(ckpt, str(ckpt_path), max_to_keep=3)

    history = {"epoch": [], "train_loss": [], "lr": [],
               "dev_APCER": [], "dev_BPCER": [], "dev_ACER": [], "dev_AUC": [], "dev_EER": []}

    best_acer = float("inf")
    patience_left = cfg.early_stop_patience

    # Write CSV header
    with open(log_csv_path, "w") as f:
        f.write("epoch,loss,lr,dev_APCER,dev_BPCER,dev_ACER,dev_AUC,dev_EER\n")

    print(f"\n---> Starting training for {model_name}...")
    for epoch in range(1, cfg.epochs + 1):
        start_time = time.time()
        epoch_losses = []

        for step_idx, (batch_inputs, batch_labels) in enumerate(train_ds):
            step = (epoch - 1) * steps_per_epoch + step_idx
            
            # Custom Learning Rate Scheduling
            lr_val = cosine_warmup_lr(step, total_steps, warmup_steps, cfg.lr, cfg.min_lr)
            optimizer.learning_rate.assign(lr_val)

            loss_val, _ = train_step_fn(batch_inputs, batch_labels)
            epoch_losses.append(loss_val.numpy())

        avg_loss = np.mean(epoch_losses)

        # Run Dev evaluation
        scores, labels_arr = run_eval(model, dev_ds, uses_lbp)
        metrics = full_benchmark(scores, labels_arr, threshold=0.5)

        # Log details
        current_lr = optimizer.learning_rate.numpy()
        print(f"Epoch {epoch:02d}/{cfg.epochs:02d} | Loss: {avg_loss:.4f} | LR: {current_lr:.6f} | "
              f"dev ACER: {metrics['ACER']*100:.2f}% (APCER: {metrics['APCER']*100:.2f}%, BPCER: {metrics['BPCER']*100:.2f}%)")

        with tb_writer.as_default():
            tf.summary.scalar("train/loss", avg_loss, step=epoch)
            tf.summary.scalar("train/learning_rate", current_lr, step=epoch)
            tf.summary.scalar("dev/ACER", metrics["ACER"], step=epoch)
            tf.summary.scalar("dev/APCER", metrics["APCER"], step=epoch)
            tf.summary.scalar("dev/BPCER", metrics["BPCER"], step=epoch)
            tf.summary.scalar("dev/AUC", metrics["AUC"], step=epoch)
            tf.summary.scalar("dev/EER", metrics["EER"], step=epoch)

        history["epoch"].append(epoch)
        history["train_loss"].append(float(avg_loss))
        history["lr"].append(float(current_lr))
        history["dev_APCER"].append(float(metrics["APCER"]))
        history["dev_BPCER"].append(float(metrics["BPCER"]))
        history["dev_ACER"].append(float(metrics["ACER"]))
        history["dev_AUC"].append(float(metrics["AUC"]))
        history["dev_EER"].append(float(metrics["EER"]))

        # Append to CSV log
        with open(log_csv_path, "a") as f:
            f.write(f"{epoch},{avg_loss:.6f},{current_lr:.6f},{metrics['APCER']:.6f},"
                    f"{metrics['BPCER']:.6f},{metrics['ACER']:.6f},{metrics['AUC']:.6f},{metrics['EER']:.6f}\n")

        if metrics["ACER"] < best_acer - 1e-4:
            best_acer = metrics["ACER"]
            patience_left = cfg.early_stop_patience
            ckpt_manager.save()
            model.save_weights(str(ckpt_path / "best_weights.weights.h5"))
            print(f"  -> new best dev ACER={best_acer*100:.2f}% (checkpoint saved)")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"  -> early stopping at epoch {epoch} (no ACER improvement for "
                      f"{cfg.early_stop_patience} epochs)")
                break

    return history, best_acer, str(ckpt_path / "best_weights.weights.h5")

def main():
    parser = argparse.ArgumentParser(description="Unified Face Anti-Spoofing Training Script")
    parser.add_argument("--model", type=str, default="all", choices=["a", "b", "c", "d", "all"],
                        help="Choose which model to train: a (EffNet), b (CDCN+LBP), c (ResNet18-CBAM), d (ResNet34-SE-CDC), or all")
    args = parser.parse_args()

    configure_gpu_and_precision()
    cfg = Config()
    setup_dirs(cfg)
    
    print("Discovering dataset structure...")
    manifest = discover_dataset(cfg.dataset_root)
    print("Building face crop cache (Haar Cascade)...")
    cached_manifest = build_crop_cache(manifest, cfg.cache_dir, cfg.img_size, cfg.use_face_crop)
    
    print("Building tf.data pipelines...")
    train_ds, n_train, train_labels = build_dataset(cached_manifest, "train", cfg, training=True)
    dev_ds,   n_dev,   dev_labels   = build_dataset(cached_manifest, "dev",   cfg, training=False)

    steps_per_epoch = n_train // cfg.batch_size
    print(f"train={n_train} dev={n_dev} | steps/epoch={steps_per_epoch}")

    # Compute class weights for imbalanced loss
    n_real = sum(train_labels)
    n_spoof = len(train_labels) - n_real
    total = len(train_labels)
    class_weight = {
        0: total / (2.0 * n_spoof),
        1: total / (2.0 * n_real),
    }
    alpha_weights = (class_weight[0] / (class_weight[0] + class_weight[1]),
                     class_weight[1] / (class_weight[0] + class_weight[1]))
    print("normalized focal alpha (spoof, real):", alpha_weights)

    MODEL_META = {
        "a": {
            "name": "model_a_effnet_style",
            "build_fn": build_efficientnet_style,
            "uses_lbp": False,
            "output_dir": "outputs_upgraded",
            "results_file": "results_a.json"
        },
        "b": {
            "name": "model_b_cdcn_lbp",
            "build_fn": build_cdcn_lbp,
            "uses_lbp": True,
            "output_dir": "outputs_upgraded",
            "results_file": "results_b.json"
        },
        "c": {
            "name": "model_c_resnet_cbam",
            "build_fn": build_resnet18_cbam,
            "uses_lbp": False,
            "output_dir": "outputs_model_c",
            "results_file": "results_c.json"
        },
        "d": {
            "name": "model_d_resnet_se",
            "build_fn": build_resnet34_se_cdc_multiscale,
            "uses_lbp": False,
            "output_dir": "outputs_model_d",
            "results_file": "results_d.json"
        }
    }

    models_to_train = ["a", "b", "c", "d"] if args.model == "all" else [args.model]

    for m_key in models_to_train:
        meta = MODEL_META[m_key]
        print(f"\n==================================================")
        print(f" TRAINING MODEL {m_key.upper()}: {meta['name']}")
        print(f"==================================================")

        # Dynamic output directory override
        cfg.output_dir = meta["output_dir"]
        os.makedirs(cfg.output_dir, exist_ok=True)

        model = meta["build_fn"](cfg.img_size, cfg.num_classes)
        model.summary()

        history, best_acer, best_ckpt = train_model(
            model, meta["name"], train_ds, dev_ds, cfg, alpha_weights, 
            uses_lbp=meta["uses_lbp"], steps_per_epoch=steps_per_epoch
        )

        # Save metadata/results
        results = {
            "best_acer": best_acer,
            "best_ckpt": best_ckpt,
            "history": history
        }
        out_path = Path(cfg.output_dir) / meta["results_file"]
        with open(out_path, "w") as f:
            json.dump(results, f)
        print(f"Results saved to {out_path}")

    print("\nAll training completed successfully!")

if __name__ == "__main__":
    main()
