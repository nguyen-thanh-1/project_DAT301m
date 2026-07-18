import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path

# Tự động dò đường dẫn LCC_FASD (trong cache kagglehub của WSL hoặc local)
_wsl_cache = "/home/hieu/.cache/kagglehub/datasets/faber24/lcc-fasd/versions/1/LCC_FASD"
_default_root = _wsl_cache if os.path.exists(_wsl_cache) else "./LCC_FASD"

@dataclass
class Config:
    # ---- paths ----
    dataset_root: str = _default_root          # chỉnh lại theo đường dẫn thật trên máy bạn
    cache_dir: str = "./cache_cropped"          # nơi lưu ảnh đã crop mặt (classical CV)
    ckpt_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    output_dir: str = "./outputs"

    # ---- image / data ----
    img_size: int = 224
    batch_size: int = 64
    val_batch_size: int = 128
    num_classes: int = 2                        # 0 = spoof, 1 = real
    use_face_crop: bool = True                  # False -> chỉ resize/center-crop, không detect mặt
    lbp_radius: int = 1
    lbp_points: int = 8

    # ---- training ----
    epochs: int = 60
    lr: float = 1e-3
    lr_min: float = 1e-6
    warmup_epochs: int = 3
    weight_decay: float = 1e-5
    focal_gamma: float = 2.0
    label_smoothing: float = 0.05
    early_stop_patience: int = 10
    grad_clip_norm: float = 5.0

    # ---- misc ----
    seed: int = 42

def setup_dirs(cfg: Config):
    for d in [cfg.cache_dir, cfg.ckpt_dir, cfg.log_dir, cfg.output_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)
