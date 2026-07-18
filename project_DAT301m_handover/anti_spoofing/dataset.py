import os
import random
import tensorflow as tf
from pathlib import Path
from anti_spoofing.config import Config
from anti_spoofing.lbp import lbp_numpy_wrapper

SPLIT_ALIASES = {
    "train": ["training", "train"],
    "dev":   ["development", "dev", "val", "validation"],
    "test":  ["evaluation", "eval", "test"],
}
CLASS_ALIASES = {
    "real":  ["real", "live", "genuine", "bonafide"],
    "spoof": ["spoof", "fake", "attack", "spoofing"],
}

def _find_dir(base: Path, aliases):
    if not base.exists():
        return None
    lower_map = {p.name.lower(): p for p in base.iterdir() if p.is_dir()}
    for alias in aliases:
        for name, path in lower_map.items():
            if alias in name:
                return path
    return None

def discover_dataset(root: str):
    root = Path(root)
    assert root.exists(), f"Dataset root not found: {root} -- sua CFG.dataset_root cho dung."

    manifest = {}  # split -> class -> list[Path]
    for split_key, split_aliases in SPLIT_ALIASES.items():
        split_dir = _find_dir(root, split_aliases)
        if split_dir is None:
            continue
        manifest[split_key] = {}
        for cls_key, cls_aliases in CLASS_ALIASES.items():
            cls_dir = _find_dir(split_dir, cls_aliases)
            if cls_dir is None:
                continue
            files = sorted(
                [p for p in cls_dir.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
            )
            manifest[split_key][cls_key] = files
    return manifest

AUTOTUNE = tf.data.AUTOTUNE

def _list_files_labels(cached_manifest, split):
    files, labels = [], []
    for cls, lbl in [("real", 1), ("spoof", 0)]:
        for f in cached_manifest[split].get(cls, []):
            files.append(str(f))
            labels.append(lbl)
    return files, labels

def _load_rgb(path, img_size):
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, [img_size, img_size], method="area")
    img = tf.cast(img, tf.float32) / 255.0
    return img

def _lbp_numpy_fn(img_uint8_np, P, R):
    return lbp_numpy_wrapper(img_uint8_np, int(P), int(R))

def _make_example(path, label, img_size, lbp_points, lbp_radius, augment):
    rgb = _load_rgb(path, img_size)

    if augment:
        # 1. Lật ảnh ngang ngẫu nhiên
        rgb = tf.image.random_flip_left_right(rgb)
        
        # 2. Dịch chuyển ngẫu nhiên bằng cách pad ảnh rồi crop ngẫu nhiên
        padding = 12
        rgb_padded = tf.image.pad_to_bounding_box(rgb, padding, padding, img_size + padding * 2, img_size + padding * 2)
        rgb = tf.image.random_crop(rgb_padded, size=[img_size, img_size, 3])
        
        # 3. Biến đổi ánh sáng, màu sắc để chống domain gap (webcam khác cảm biến camera)
        rgb = tf.image.random_brightness(rgb, max_delta=0.08)
        rgb = tf.image.random_contrast(rgb, lower=0.90, upper=1.10)
        rgb = tf.image.random_saturation(rgb, lower=0.85, upper=1.15)
        rgb = tf.image.random_hue(rgb, max_delta=0.03)
        rgb = tf.clip_by_value(rgb, 0.0, 1.0)

    rgb_uint8 = tf.cast(rgb * 255.0, tf.uint8)
    lbp = tf.numpy_function(
        func=lambda x: _lbp_numpy_fn(x, lbp_points, lbp_radius),
        inp=[rgb_uint8], Tout=tf.float32,
    )
    lbp.set_shape([img_size, img_size, 1])
    rgb.set_shape([img_size, img_size, 3])
    label = tf.cast(label, tf.int32)
    return {"rgb": rgb, "lbp": lbp}, label

def build_dataset(cached_manifest, split, cfg: Config, training: bool):
    files, labels = _list_files_labels(cached_manifest, split)
    print(f"[{split}] {len(files)} files | real={sum(labels)} spoof={len(labels)-sum(labels)}")
    ds = tf.data.Dataset.from_tensor_slices((files, labels))
    if training:
        ds = ds.shuffle(buffer_size=min(len(files), 20000), seed=cfg.seed, reshuffle_each_iteration=True)
    ds = ds.map(
        lambda p, l: _make_example(p, l, cfg.img_size, cfg.lbp_points, cfg.lbp_radius, augment=training),
        num_parallel_calls=AUTOTUNE,
    )
    batch_size = cfg.batch_size if training else cfg.val_batch_size
    ds = ds.batch(batch_size, drop_remainder=training)
    ds = ds.prefetch(AUTOTUNE)
    return ds, len(files), labels
