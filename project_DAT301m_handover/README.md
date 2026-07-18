# Face Recognition: ResNet-50 + Transformer & ArcFace (One-shot Learning)

Hệ thống nhận diện khuôn mặt sử dụng kiến trúc lai **ResNet-50 CNN + Transformer Encoder** kết hợp hàm mất mát **ArcFace (Additive Angular Margin Loss)** với chiến lược **Progressive Margin** để huấn luyện từ đầu (from scratch). Hệ thống phục vụ cho module điểm danh thông minh bằng khuôn mặt với khả năng One-shot Learning.

---

## 1. Thống kê Bộ Dữ Liệu (Dataset Statistics)

Bộ dữ liệu nằm trong thư mục `dataset/` được chia làm hai tập chính với danh tính độc lập (disjoint identities):

| Thuộc tính | Tập Huấn Luyện (`train`) | Tập Kiểm Thử (`val`) |
| :--- | :---: | :---: |
| **Tổng số danh tính (Classes/IDs)** | 658 | 106 |
| **Tổng số lượng ảnh** | 183,216 | 23,034 |
| **Số ảnh ít nhất / Danh tính** | 2 | 4 |
| **Số ảnh nhiều nhất / Danh tính** | 720 | 644 |
| **Số ảnh trung bình / Danh tính** | 278.44 | 217.30 |
| **Độ phân giải ảnh (Image Shape)** | Lai ghép: `160x160` (cắt sẵn) và `622x544` (gốc) | Lai ghép: `160x160` và `622x544` |

> [!NOTE]
> Bộ dữ liệu chứa danh tính dạng lai: ca sĩ/diễn viên Việt Nam (`160x160` pixel) và tập VGGFace2 mã hóa dạng `n000...`. Toàn bộ ảnh được chuẩn hóa kích thước tự động qua data pipeline.

---

## 2. Kiến Trúc Mạng (Architecture)

Mô hình sử dụng kiến trúc **Hybrid CNN-Transformer**, chia làm 3 phần:

### A. CNN Feature Extractor (ResNet-50 Stages 1-5)
Tự xây dựng bằng Keras Functional API, gồm `conv_block` và `identity_block`:

```
Image (112x112x3)
  → Stage 1: Conv 7x7 + MaxPool       → (28, 28, 64)
  → Stage 2: 3 Residual Blocks        → (28, 28, 256)
  → Stage 3: 4 Residual Blocks        → (14, 14, 512)
  → Stage 4: 6 Residual Blocks        → (7,  7,  1024)
  → Stage 5: 3 Residual Blocks        → (4,  4,  2048)
```

### B. Transformer Feature Enhancer (2 Encoder Blocks)
Đặt **sau Stage 5** của ResNet-50 để học **quan hệ toàn cục** giữa các vùng khuôn mặt (mắt↔mũi↔miệng):

```
Feature Map (4x4x2048)
  → 1x1 Conv Projection               → (4, 4, 512)
  → Reshape thành token sequence       → (16 tokens, 512 dims)
  → Learnable Positional Encoding
  → 2x Transformer Encoder Block:
      ├── Pre-LayerNorm
      ├── Multi-Head Self-Attention (8 heads, key_dim=64)
      └── Position-wise FFN (512→1024→512)
  → Final LayerNorm
  → Global Average Pooling (over tokens) → (512,)
  → BatchNormalization                    → Embedding Vector (512-d)
```

> [!TIP]
> Có thể tắt module Transformer để chạy chế độ CNN-only bằng flag `--no_transformer`.

### C. Classification Head (ArcFace Layer)
Custom `tf.keras.layers.Layer` thực hiện Additive Angular Margin Loss:
* Chuẩn hóa $L_2$ trên embedding và ma trận trọng số $W$.
* **Progressive Margin:** Biên góc tăng dần từ `m = 0.10` → `m = 0.50` qua các epoch (thay vì cố định `0.50` ngay từ đầu, giúp mô hình hội tụ nhanh hơn khi train from scratch).
* Hệ số tỉ lệ: $s = 64.0$.

### D. Số lượng tham số (Parameters)
* **ResNet-50 Backbone (CNN):** ~24.6M tham số
* **Transformer Encoder (2 blocks):** ~4.2M tham số bổ sung
* **ArcFace Head (658 classes):** ~336K tham số
* **Tổng cộng:** ~29.1M tham số

---

## 3. Siêu Tham Số (Hyperparameters)

| Tham số | Giá trị mặc định | Ghi chú |
| :--- | :---: | :--- |
| **Input Resolution** | `112 x 112 x 3` | Có thể đổi qua `--img_size` |
| **Batch Size** | `128` | Tối ưu cho RTX 5060 Ti 16GB VRAM |
| **Epochs** | `20` | Kết hợp EarlyStopping (patience=5) |
| **Embedding Dim** | `512` | Kích thước vector đặc trưng khuôn mặt |
| **ArcFace Scale ($s$)** | `64.0` | Hệ số phóng đại logits |
| **ArcFace Margin ($m$)** | `0.10 → 0.50` | Progressive margin (tăng tuyến tính qua epochs) |
| **Learning Rate** | `0.001` (peak) | Warmup 1 epoch + Cosine Decay |
| **Min Learning Rate** | `1e-6` | Giá trị LR tối thiểu cuối quá trình cosine decay |
| **Optimizer** | Adam | Kết hợp LR Schedule |
| **Chuẩn hóa pixel** | `[-1, 1]` | $(x - 127.5) / 128.0$ |
| **Transformer Heads** | `8` | Multi-Head Self-Attention |
| **Transformer FFN Dim** | `1024` | Feed-Forward Network hidden dim |
| **Transformer Layers** | `2` | Số lượng Encoder blocks |
| **Transformer Dropout** | `0.1` | Dropout trong attention và FFN |

---

## 4. Kỹ Thuật Tối Ưu (Optimization Techniques)

### A. Progressive Margin (ArcFace)
Thay vì áp dụng margin $m = 0.50$ ngay từ đầu (khiến accuracy = 0 trong nhiều epoch đầu khi train from scratch), hệ thống **tăng dần margin** từ `0.10 rad (5.7°)` đến `0.50 rad (28.6°)` qua các epoch, giúp backbone học được đặc trưng cơ bản trước khi siết chặt biên quyết định.

### B. Learning Rate Schedule (Warmup + Cosine Decay)
* **Epoch 1:** LR tăng tuyến tính từ 0 → `peak_lr` (warmup phase).
* **Epoch 2 trở đi:** LR giảm dần theo đường cong cosine về `min_lr = 1e-6`.

### C. Data Augmentation
* **Random Horizontal Flip:** Lật ngang ngẫu nhiên khuôn mặt mỗi epoch → tăng tính tổng quát hóa.

### D. Data Pipeline tối ưu
* Mặc định cache ra **disk** (`results/tf_cache`) để tránh đầy RAM khi dataset lớn.
* `.shuffle(10000)` xáo trộn thứ tự batch mỗi epoch.
* `.prefetch(AUTOTUNE)` nạp trước batch tiếp theo trong khi GPU đang tính toán.
* `crop_to_aspect_ratio=True` để giảm méo tỉ lệ khi ảnh có nhiều kích thước.

### E. Phần cứng (Hardware)
* **CPU:** Intel Core i7-14700F | **RAM:** 32GB | **GPU:** NVIDIA RTX 5060 Ti 16GB VRAM
* `tf.config.experimental.set_memory_growth = True` → phân bổ VRAM động.

---

## 5. Hướng Dẫn Chạy (Execution)

### Yêu cầu
* Môi trường đã cài TensorFlow 2.10.0, NumPy < 2.0.0, matplotlib.

### A. Kiểm tra GPU
```powershell
.\.venv\Scripts\python.exe .\check.py
```

### B. Huấn luyện mặc định (ResNet-50 + Transformer, 20 epochs)
```powershell
.\.venv\Scripts\python.exe .\train.py
```

### C. Huấn luyện chế độ CNN-only (không Transformer)
```powershell
.\.venv\Scripts\python.exe .\train.py --no_transformer
```

### D. Tùy chỉnh tham số
```powershell
.\.venv\Scripts\python.exe .\train.py --epochs 30 --batch_size 128 --img_size 112 --lr 0.001
```

### E. Xem tất cả tham số
```powershell
.\.venv\Scripts\python.exe .\train.py --help
```

### F. Inference / One-shot (trích xuất embedding + cosine similarity)
> Backbone ResNet-50+Transformer được định nghĩa trong `training/face_recognition/resnet50.py`. Inference không cần import `train.py`.

```powershell
.\.venv\Scripts\python.exe .\infer_embeddings.py --weights results\\best_backbone_weights.h5 --img path\\to\\a.jpg --img2 path\\to\\b.jpg
```

---

## 6. Kết Quả Đầu Ra (Outputs)

Thư mục `results/` chứa:

| File | Mô tả |
| :--- | :--- |
| `best_resnet50t_arcface.h5` | Trọng số toàn bộ mô hình huấn luyện tốt nhất (theo `val_loss`) |
| `best_backbone_weights.h5` | Trọng số backbone (ResNet-50 + Transformer) tại epoch tốt nhất, dùng cho inference trích xuất embedding 512-d |
| `training_history.png` | Biểu đồ Loss và Accuracy (Train/Val) qua các epoch |

---

## 7. Cấu Trúc Thư Mục (Project Structure)

```
Project_DAT301m/
├── dataset/
│   ├── train/          # 658 danh tính, 183,216 ảnh
│   └── val/            # 106 danh tính, 23,034 ảnh
├── training/
│   └── face_recognition/
│       └── resnet50.py # Kiến trúc mô hình (reference)
├── results/            # Kết quả huấn luyện (auto-generated)
├── infer_embeddings.py # Inference: trích xuất embedding / cosine similarity
├── train.py            # Script huấn luyện chính
├── check.py            # Script kiểm tra GPU/TensorFlow
└── README.md
```
