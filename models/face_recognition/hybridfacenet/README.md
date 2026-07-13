# HybridFaceNet

Conformer-style dual-branch face-recognition model implemented in TensorFlow 2.x.

## Architecture

- Input: `112x112x3`, normalized inside the model with `(pixel - 127.5) / 128.0`.
- Shared stem: `Conv2D(64, 3x3, stride=2) -> BN -> PReLU`.
- CNN branch: iResNet-Lite stages `[2, 3, 4, 2]` with `IBasicBlock`.
- Transformer branch: 64 CNN-derived patch tokens plus `[CLS]`, 6 Pre-LN Transformer blocks, 8 heads, MLP ratio 2.
- FCU coupling: CNN-to-Transformer token injection and Transformer-to-CNN spatial injection after every two Transformer blocks.
- Fusion: learnable clipped sigmoid alpha combining L2-normalized CNN and Transformer embeddings.
- Output: 512-D L2-normalized embedding for one-shot face verification.

## Training

```powershell
python models\face_recognition\hybridfacenet\train.py `
  --dataset_dir dataset_final\train `
  --val_dir dataset_final\val `
  --test_dir dataset_final\test `
  --batch_size 64 `
  --epochs 30 `
  --verify_every 5 `
  --resume
```

The train script uses:

- Phase 1: CNN warm-up for 5 epochs, FCU gates disabled, LR `0.01`.
- Phase 2: full joint training, SGD + Nesterov + weight decay, warm-up cosine LR from `1e-4 -> 0.05 -> 1e-5`.
- ArcFace margin schedule: `0.1, 0.2, 0.3, 0.4, 0.5`, stepping every 5 epochs.
- Augmentation: horizontal flip, brightness, contrast, hue; no crop/rotation/elastic transforms.
- Resume: `--resume` restores the latest phase checkpoint with backbone weights, ArcFace head, optimizer state, and next epoch.

Outputs are saved under:

```text
models/face_recognition/hybridfacenet/results/arcface/
```

Full resume checkpoints are saved under:

```text
models/face_recognition/hybridfacenet/results/arcface/checkpoints/warmup/
models/face_recognition/hybridfacenet/results/arcface/checkpoints/joint/
```

## Evaluation

```powershell
python models\face_recognition\hybridfacenet\evaluate.py `
  --test_dir dataset_final\test `
  --weights models\face_recognition\hybridfacenet\results\arcface\best_hybridfacenet_backbone.h5 `
  --weights_type backbone `
  --num_pairs 50000
```

The evaluation script reports CNN-only, Transformer-only and Fused metrics:

- EER
- TAR@FAR=0.001
- TAR@FAR=0.01
- AUC-ROC
- Optimal cosine threshold
- ROC curve
- Score distributions
- Optional fused t-SNE if `scikit-learn` is installed
