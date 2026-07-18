import numpy as np

def compute_roc_manual(scores, labels, n_thresholds=1000):
    """scores: P(real) trong [0,1]; labels: 1=real, 0=spoof. Tra ve threshold, far, frr (=apcer,bpcer)."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    thresholds = np.linspace(0.0, 1.0, n_thresholds)

    apcer_list, bpcer_list = [], []
    real_mask = labels == 1
    spoof_mask = labels == 0
    n_real = real_mask.sum()
    n_spoof = spoof_mask.sum()

    for t in thresholds:
        pred_real = scores >= t
        # APCER: spoof examples predicted as real
        apcer = np.sum(pred_real & spoof_mask) / max(1, n_spoof)
        # BPCER: real examples predicted as spoof
        bpcer = np.sum(~pred_real & real_mask) / max(1, n_real)
        apcer_list.append(apcer)
        bpcer_list.append(bpcer)

    return thresholds, np.array(apcer_list), np.array(bpcer_list)


def compute_eer(thresholds, apcer, bpcer):
    diff = apcer - bpcer
    idx = np.argmin(np.abs(diff))
    eer = (apcer[idx] + bpcer[idx]) / 2.0
    return eer, thresholds[idx]


def compute_auc_manual(scores, labels):
    """AUC qua Mann-Whitney U statistic (khong can sklearn)."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    rank_pos_sum = ranks[: len(pos)].sum()
    auc = (rank_pos_sum - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))
    return auc


def full_benchmark(scores, labels, threshold=0.5):
    """scores = P(real). Tra ve dict day du cac metric o 1 threshold co dinh + AUC/EER toan cuc."""
    thresholds, apcer_curve, bpcer_curve = compute_roc_manual(scores, labels)
    eer, eer_thresh = compute_eer(thresholds, apcer_curve, bpcer_curve)
    auc = compute_auc_manual(scores, labels)

    preds_real = np.asarray(scores) >= threshold
    labels_arr = np.asarray(labels)
    n_real = (labels_arr == 1).sum()
    n_spoof = (labels_arr == 0).sum()
    apcer = np.sum(preds_real & (labels_arr == 0)) / max(1, n_spoof)
    bpcer = np.sum(~preds_real & (labels_arr == 1)) / max(1, n_real)
    acer = (apcer + bpcer) / 2.0
    accuracy = np.mean(preds_real.astype(int) == labels_arr)

    return {
        "threshold": threshold, "APCER": apcer, "BPCER": bpcer, "ACER": acer,
        "AUC": auc, "EER": eer, "EER_threshold": eer_thresh, "accuracy_raw": accuracy,
        "_det_curve": (thresholds, apcer_curve, bpcer_curve),
    }
