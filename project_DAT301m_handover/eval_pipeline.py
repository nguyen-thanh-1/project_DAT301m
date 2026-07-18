import os
import csv
import json
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

from anti_spoofing.config import Config
from anti_spoofing.dataset import discover_dataset, build_dataset
from anti_spoofing.face_detector import build_crop_cache
from anti_spoofing.models import (
    build_efficientnet_style, 
    build_cdcn_lbp, 
    build_resnet18_cbam, 
    build_resnet34_se_cdc_multiscale
)
from anti_spoofing.metrics import full_benchmark, compute_roc_manual, compute_auc_manual
from train_pipeline import run_eval, configure_gpu_and_precision

def load_best_and_eval(model, best_ckpt_path, test_ds, uses_lbp, model_name):
    model.load_weights(best_ckpt_path)
    scores, labels_arr = run_eval(model, test_ds, uses_lbp)
    metrics = full_benchmark(scores, labels_arr, threshold=0.5)
    print(f"\n=== {model_name} — EVALUATION (test) SET ===")
    for k in ["APCER", "BPCER", "ACER", "AUC", "EER"]:
        v = metrics[k]
        print(f"  {k:6s}: {v*100:6.2f}%" if k != "AUC" else f"  {k:6s}: {v:.4f}")
    return metrics, scores, labels_arr

def plot_history(history, title, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="train loss")
    axes[0].set_title(f"{title} - Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["epoch"], np.array(history["dev_ACER"]) * 100, label="dev ACER%", color="crimson")
    axes[1].plot(history["epoch"], np.array(history["dev_APCER"]) * 100, "--", label="dev APCER%", alpha=0.7)
    axes[1].plot(history["epoch"], np.array(history["dev_BPCER"]) * 100, "--", label="dev BPCER%", alpha=0.7)
    axes[1].set_title(f"{title} - Error Rates")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(history["epoch"], history["dev_AUC"], label="dev AUC", color="green")
    axes[2].plot(history["epoch"], np.array(history["dev_EER"]) * 100, label="dev EER%", color="orange")
    axes[2].set_title(f"{title} - AUC / EER")
    axes[2].set_xlabel("epoch")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_roc(scores_list, labels_list, names, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for scores, labels_arr, name in zip(scores_list, labels_list, names):
        thresholds, apcer_curve, bpcer_curve = compute_roc_manual(scores, labels_arr)
        tpr = 1 - bpcer_curve  # recall on real class
        fpr = apcer_curve      # spoof accepted as real
        order = np.argsort(fpr)
        auc = compute_auc_manual(scores, labels_arr)
        axes[0].plot(fpr[order], tpr[order], label=f"{name} (AUC={auc:.3f})")

        axes[1].plot(apcer_curve * 100, bpcer_curve * 100, label=name)

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[0].set_xlabel("FPR (APCER)")
    axes[0].set_ylabel("TPR (1-BPCER)")
    axes[0].set_title("ROC Curve")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("APCER (%)")
    axes[1].set_ylabel("BPCER (%)")
    axes[1].set_title("DET Curve")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Unified Face Anti-Spoofing Evaluation Script")
    parser.add_argument("--model", type=str, default="all", choices=["a", "b", "c", "d", "all"],
                        help="Choose which model to evaluate: a, b, c, d, or all")
    args = parser.parse_args()

    configure_gpu_and_precision()
    cfg = Config()

    print("Discovering dataset structure...")
    manifest = discover_dataset(cfg.dataset_root)
    cached_manifest = build_crop_cache(manifest, cfg.cache_dir, cfg.img_size, cfg.use_face_crop)
    
    print("Building tf.data test pipeline...")
    test_ds, n_test, test_labels  = build_dataset(cached_manifest, "test",  cfg, training=False)

    MODEL_META = {
        "a": {
            "name": "model_a_effnet_style",
            "build_fn": build_efficientnet_style,
            "uses_lbp": False,
            "output_dir": "outputs_upgraded",
            "results_file": "results_a.json",
            "ckpt_path": "checkpoints/model_a_effnet_style/best_weights.weights.h5",
            "display_name": "Model A - EffNet-style (ours, RGB)"
        },
        "b": {
            "name": "model_b_cdcn_lbp",
            "build_fn": build_cdcn_lbp,
            "uses_lbp": True,
            "output_dir": "outputs_upgraded",
            "results_file": "results_b.json",
            "ckpt_path": "checkpoints/model_b_cdcn_lbp/best_weights.weights.h5",
            "display_name": "Model B - CDCN+LBP (ours, RGB+texture)"
        },
        "c": {
            "name": "model_c_resnet_cbam",
            "build_fn": build_resnet18_cbam,
            "uses_lbp": False,
            "output_dir": "outputs_model_c",
            "results_file": "results_c.json",
            "ckpt_path": "checkpoints/model_c_resnet_cbam/best_weights.weights.h5",
            "display_name": "Model C - ResNet18-CBAM (ours, RGB+CBAM)"
        },
        "d": {
            "name": "model_d_resnet_se",
            "build_fn": build_resnet34_se_cdc_multiscale,
            "uses_lbp": False,
            "output_dir": "outputs_model_d",
            "results_file": "results_d.json",
            "ckpt_path": "checkpoints/model_d_resnet_se/best_weights.weights.h5",
            "display_name": "Model D - ResNet34-SE-CDC-MS (ours)"
        }
    }

    literature = [
        ("MobileNetV3-Large (light-weight-FAS repo)", 0.921, 16.1, 17.3, 15.4, 16.3, "3.0M"),
        ("MobileNetV3-Small (light-weight-FAS repo)", 0.889, 18.7, 14.8, 24.6, 19.7, "1.0M"),
    ]

    if args.model == "all":
        print("\n=== Evaluating All Models + Computing Ensemble ===")
        scores_list = []
        labels_list = []
        names = []
        rows = []

        # Model A
        meta_a = MODEL_META["a"]
        model_a = meta_a["build_fn"](cfg.img_size, cfg.num_classes)
        metrics_a, scores_a, labels_a = load_best_and_eval(model_a, meta_a["ckpt_path"], test_ds, meta_a["uses_lbp"], meta_a["name"])
        scores_list.append(scores_a)
        labels_list.append(labels_a)
        names.append(meta_a["display_name"])
        rows.append((meta_a["display_name"], metrics_a["AUC"], metrics_a["EER"]*100, metrics_a["APCER"]*100, metrics_a["BPCER"]*100, metrics_a["ACER"]*100, f"{model_a.count_params():,}"))

        # Model B
        meta_b = MODEL_META["b"]
        model_b = meta_b["build_fn"](cfg.img_size, cfg.num_classes)
        metrics_b, scores_b, labels_b = load_best_and_eval(model_b, meta_b["ckpt_path"], test_ds, meta_b["uses_lbp"], meta_b["name"])
        scores_list.append(scores_b)
        labels_list.append(labels_b)
        names.append(meta_b["display_name"])
        rows.append((meta_b["display_name"], metrics_b["AUC"], metrics_b["EER"]*100, metrics_b["APCER"]*100, metrics_b["BPCER"]*100, metrics_b["ACER"]*100, f"{model_b.count_params():,}"))

        # Model C
        meta_c = MODEL_META["c"]
        model_c = meta_c["build_fn"](cfg.img_size, cfg.num_classes)
        metrics_c, scores_c, labels_c = load_best_and_eval(model_c, meta_c["ckpt_path"], test_ds, meta_c["uses_lbp"], meta_c["name"])
        scores_list.append(scores_c)
        labels_list.append(labels_c)
        names.append(meta_c["display_name"])
        rows.append((meta_c["display_name"], metrics_c["AUC"], metrics_c["EER"]*100, metrics_c["APCER"]*100, metrics_c["BPCER"]*100, metrics_c["ACER"]*100, f"{model_c.count_params():,}"))

        # Model D
        meta_d = MODEL_META["d"]
        model_d = meta_d["build_fn"](cfg.img_size, cfg.num_classes)
        metrics_d, scores_d, labels_d = load_best_and_eval(model_d, meta_d["ckpt_path"], test_ds, meta_d["uses_lbp"], meta_d["name"])
        scores_list.append(scores_d)
        labels_list.append(labels_d)
        names.append(meta_d["display_name"])
        rows.append((meta_d["display_name"], metrics_d["AUC"], metrics_d["EER"]*100, metrics_d["APCER"]*100, metrics_d["BPCER"]*100, metrics_d["ACER"]*100, f"{model_d.count_params():,}"))

        # Compute Ensemble Model A & C (40% A, 60% C)
        print("\nComputing Ensemble Mode A & C...")
        scores_ens = 0.40 * scores_a + 0.60 * scores_c
        metrics_ens = full_benchmark(scores_ens, labels_a, threshold=0.5)
        scores_list.append(scores_ens)
        labels_list.append(labels_a)
        names.append("Ensemble Mode A & C (ours)")
        rows.append(("Ensemble Mode A & C (ours, 40% A + 60% C)", metrics_ens["AUC"], metrics_ens["EER"]*100, metrics_ens["APCER"]*100, metrics_ens["BPCER"]*100, metrics_ens["ACER"]*100, f"{(model_a.count_params() + model_c.count_params()):,}"))

        print("\n=== Ensemble Mode A & C — EVALUATION SET ===")
        for k in ["APCER", "BPCER", "ACER", "AUC", "EER"]:
            v = metrics_ens[k]
            print(f"  {k:6s}: {v*100:6.2f}%" if k != "AUC" else f"  {k:6s}: {v:.4f}")

        # Plot master comparison
        output_dir = "outputs_upgraded"
        os.makedirs(output_dir, exist_ok=True)
        plot_roc(scores_list, labels_list, names, os.path.join(output_dir, "roc_det_comparison.png"))
        print(f"Saved master ROC DET curve to {output_dir}/roc_det_comparison.png")

        # Save master benchmark CSV
        csv_path = os.path.join(output_dir, "final_benchmark.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model", "AUC", "EER%", "APCER%", "BPCER%", "ACER%", "params"])
            for row in rows:
                writer.writerow(row)
            for row in literature:
                writer.writerow(["[literature] " + row[0]] + list(row[1:]))
        print(f"Saved master benchmark CSV to {csv_path}")

    else:
        meta = MODEL_META[args.model]
        print(f"\n=== Evaluating Model {args.model.upper()}: {meta['name']} ===")
        
        output_dir = meta["output_dir"]
        os.makedirs(output_dir, exist_ok=True)

        model = meta["build_fn"](cfg.img_size, cfg.num_classes)
        metrics, scores, labels = load_best_and_eval(model, meta["ckpt_path"], test_ds, meta["uses_lbp"], meta["name"])

        # Plot individual ROC
        plot_roc([scores], [labels], [meta["display_name"]], os.path.join(output_dir, f"{args.model}_roc_det.png"))
        print(f"Saved ROC plot to {output_dir}/{args.model}_roc_det.png")

        # Plot training curves from JSON history
        history_path = os.path.join(output_dir, meta["results_file"])
        try:
            with open(history_path, "r") as f:
                res_data = json.load(f)
                plot_history(res_data["history"], meta["name"], os.path.join(output_dir, f"model_{args.model}_curves.png"))
                print(f"Saved training history curves to {output_dir}/model_{args.model}_curves.png")
        except FileNotFoundError:
            print(f"Warning: History file {history_path} not found. Skipping history curves plotting.")

        # Save single benchmark CSV
        csv_path = os.path.join(output_dir, "final_benchmark.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model", "AUC", "EER%", "APCER%", "BPCER%", "ACER%", "params"])
            writer.writerow((meta["display_name"], metrics["AUC"], metrics["EER"]*100, metrics["APCER"]*100, metrics["BPCER"]*100, metrics["ACER"]*100, f"{model.count_params():,}"))
            for row in literature:
                writer.writerow(["[literature] " + row[0]] + list(row[1:]))
        print(f"Saved benchmark CSV to {csv_path}")

if __name__ == "__main__":
    main()
