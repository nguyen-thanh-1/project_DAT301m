import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from training.face_recognition.resnet50 import ResNet50_Backbone


def load_and_preprocess(image_path: Path, img_size: int):
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        w, h = im.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        im = im.crop((left, top, left + side, top + side))
        im = im.resize((img_size, img_size), resample=Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32)
    arr = (arr - 127.5) / 128.0
    return arr


def l2_normalize(x: np.ndarray, axis=-1, eps=1e-12):
    denom = np.linalg.norm(x, axis=axis, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom


def cosine(a: np.ndarray, b: np.ndarray):
    a = l2_normalize(a)
    b = l2_normalize(b)
    return float(np.sum(a * b))


def main():
    ap = argparse.ArgumentParser(description="Extract embeddings with ResNet50+Transformer backbone.")
    ap.add_argument("--weights", default="results/best_backbone_weights.h5", help="Backbone weights file (.h5).")
    ap.add_argument("--img", required=True, help="Path to first image.")
    ap.add_argument("--img2", default="", help="Optional second image to compute cosine similarity.")
    ap.add_argument("--img_size", type=int, default=112)
    ap.add_argument("--embedding_dim", type=int, default=512)
    ap.add_argument("--no_transformer", action="store_true", help="Disable transformer module.")
    args = ap.parse_args()

    weights = Path(args.weights)
    img1 = Path(args.img)
    img2 = Path(args.img2) if args.img2 else None

    backbone = ResNet50_Backbone(
        input_shape=(args.img_size, args.img_size, 3),
        embedding_dim=args.embedding_dim,
        use_transformer=not args.no_transformer,
    )
    backbone.load_weights(str(weights))

    x1 = load_and_preprocess(img1, args.img_size)[None, ...]
    emb1 = backbone.predict(x1, verbose=0)[0]
    emb1 = l2_normalize(emb1)
    print("img1_embedding_norm", float(np.linalg.norm(emb1)))

    if img2:
        x2 = load_and_preprocess(img2, args.img_size)[None, ...]
        emb2 = backbone.predict(x2, verbose=0)[0]
        emb2 = l2_normalize(emb2)
        print("img2_embedding_norm", float(np.linalg.norm(emb2)))
        print("cosine_similarity", cosine(emb1, emb2))


if __name__ == "__main__":
    main()

