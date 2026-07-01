# Kiến trúc mô hình: iResNet-50 + ArcFace Head

> **Source file:** `training/face_recognition/iresnet50.py`
> **Tổng tham số:** ~43.7M | **Input:** 112×112×3 | **Output Embedding:** 512-D

---

## 1. Tổng quan kiến trúc (Full Pipeline)

```mermaid
flowchart TB
    subgraph PREPROCESSING["🖼️ Tiền Xử Lý (preprocess.py)"]
        RAW["Ảnh gốc\n(Kích thước bất kỳ)"]
        MTCNN["MTCNN\nDetect + Align"]
        CROP["Crop & Resize\n112×112"]
        RAW --> MTCNN --> CROP
    end

    subgraph TRAINING_MODEL["🏋️ Training Model (ArcFace_Training_iResNet50)"]
        direction TB
        IMG_IN["📥 image_input\n(batch, 112, 112, 3)"]
        LBL_IN["🏷️ label_input\n(batch,)"]
        
        subgraph BACKBONE["🧠 iResNet-50 Backbone"]
            STEM["Stem Block"]
            S1["Stage 1 (×3 blocks)\n64 filters"]
            S2["Stage 2 (×4 blocks)\n128 filters"]
            S3["Stage 3 (×14 blocks)\n256 filters"]
            S4["Stage 4 (×3 blocks)\n512 filters"]
            HEAD["ArcFace Head\n(Embedding Layer)"]
            STEM --> S1 --> S2 --> S3 --> S4 --> HEAD
        end

        ARCFACE["⭐ ArcFace Loss Layer\nMargin + Scale"]
        LOGITS["📊 Logits\n(batch, num_classes)"]
        
        IMG_IN --> BACKBONE
        BACKBONE -->|"emb (batch, 512)"| ARCFACE
        LBL_IN --> ARCFACE
        ARCFACE --> LOGITS
    end

    subgraph INFERENCE["🔍 Inference (One-Shot Learning)"]
        EMB["Embedding Vector\n512-D + L2 Norm"]
        COSINE["Cosine Similarity\nSo sánh 2 vector"]
        EMB --> COSINE
    end

    CROP --> IMG_IN
    BACKBONE -.->|"Chỉ dùng Backbone"| EMB

    style PREPROCESSING fill:#1a1a2e,stroke:#e94560,color:#eee
    style TRAINING_MODEL fill:#0f3460,stroke:#e94560,color:#eee
    style INFERENCE fill:#16213e,stroke:#0f3460,color:#eee
    style ARCFACE fill:#e94560,stroke:#fff,color:#fff
```

---

## 2. Chi tiết từng khối (Block-by-Block)

### 2.1 Stem Block

| Thứ tự | Layer | Kernel | Stride | Output Shape | Params |
|--------|-------|--------|--------|-------------|--------|
| 1 | `Conv2D` (stem_conv) | 3×3, 64 | 1 | 112×112×64 | 1,728 |
| 2 | `BatchNorm` (stem_bn) | - | - | 112×112×64 | 256 |
| 3 | `PReLU` (stem_prelu) | - | - | 112×112×64 | 1 |

---

### 2.2 IR Block (Improved Residual Block)

Đây là đơn vị cơ bản được lặp lại 24 lần trong toàn bộ mạng.

```mermaid
flowchart TB
    INPUT["Input\n(H × W × C_in)"]
    
    subgraph MAIN_PATH["Main Path"]
        BN1["BatchNorm"]
        CONV1["Conv2D 3×3\nstride=1, same"]
        BN2["BatchNorm"]
        PRELU["PReLU\n(shared_axes=1,2)"]
        CONV2["Conv2D 3×3\nstride=S, same"]
        BN3["BatchNorm"]
        BN1 --> CONV1 --> BN2 --> PRELU --> CONV2 --> BN3
    end

    subgraph SHORTCUT["Shortcut Path"]
        SC_CONV["Conv2D 1×1\nstride=S"]
        SC_BN["BatchNorm"]
        SC_CONV --> SC_BN
    end

    ADD["⊕ Add"]
    OUTPUT["Output\n(H/S × W/S × C_out)"]

    INPUT --> BN1
    INPUT --> SC_CONV
    BN3 --> ADD
    SC_BN --> ADD
    ADD --> OUTPUT

    style MAIN_PATH fill:#1a1a2e,stroke:#e94560,color:#eee
    style SHORTCUT fill:#16213e,stroke:#0f3460,color:#eee
    style ADD fill:#e94560,stroke:#fff,color:#fff
```

> **Lưu ý:** Shortcut path chỉ có Conv2D 1×1 khi `stride ≠ 1` hoặc `C_in ≠ C_out` (tức block đầu tiên của mỗi Stage). Các block tiếp theo dùng Identity shortcut (nối thẳng).

---

### 2.3 Bốn Stage chi tiết

#### Stage 1 — 3 blocks × 64 filters

| Block | Stride | Input Shape | Output Shape | Shortcut |
|-------|--------|-------------|-------------|----------|
| `stage1_block1` | **2** | 112×112×64 | **56×56×64** | Conv 1×1 (stride=2) |
| `stage1_block2` | 1 | 56×56×64 | 56×56×64 | Identity |
| `stage1_block3` | 1 | 56×56×64 | 56×56×64 | Identity |

#### Stage 2 — 4 blocks × 128 filters

| Block | Stride | Input Shape | Output Shape | Shortcut |
|-------|--------|-------------|-------------|----------|
| `stage2_block1` | **2** | 56×56×64 | **28×28×128** | Conv 1×1 (stride=2) |
| `stage2_block2` | 1 | 28×28×128 | 28×28×128 | Identity |
| `stage2_block3` | 1 | 28×28×128 | 28×28×128 | Identity |
| `stage2_block4` | 1 | 28×28×128 | 28×28×128 | Identity |

#### Stage 3 — 14 blocks × 256 filters

| Block | Stride | Input Shape | Output Shape | Shortcut |
|-------|--------|-------------|-------------|----------|
| `stage3_block1` | **2** | 28×28×128 | **14×14×256** | Conv 1×1 (stride=2) |
| `stage3_block2` → `block14` | 1 | 14×14×256 | 14×14×256 | Identity (×13) |

#### Stage 4 — 3 blocks × 512 filters

| Block | Stride | Input Shape | Output Shape | Shortcut |
|-------|--------|-------------|-------------|----------|
| `stage4_block1` | **2** | 14×14×256 | **7×7×512** | Conv 1×1 (stride=2) |
| `stage4_block2` | 1 | 7×7×512 | 7×7×512 | Identity |
| `stage4_block3` | 1 | 7×7×512 | 7×7×512 | Identity |

---

### 2.4 ArcFace Head (Embedding Layer)

Phần cuối của Backbone, chuyển đổi feature map thành vector embedding 512 chiều.

```mermaid
flowchart TB
    FM["Feature Map\n7 × 7 × 512\n= 25,088 values"]
    FLAT["Flatten\n(batch, 25088)"]
    BN_FLAT["BatchNorm\nmomentum=0.9\nepsilon=2e-5"]
    DROP["Dropout (0.3)\n(Training only)"]
    FC["Dense 512\nuse_bias=False\nglorot_normal"]
    BN_EMB["BatchNorm\nscale=False\nmomentum=0.9\nepsilon=2e-5"]
    EMB["📍 Embedding\n(batch, 512)"]

    FM --> FLAT --> BN_FLAT --> DROP --> FC --> BN_EMB --> EMB

    style FM fill:#16213e,stroke:#0f3460,color:#eee
    style EMB fill:#e94560,stroke:#fff,color:#fff
    style DROP fill:#533483,stroke:#e94560,color:#eee
```

| Thứ tự | Layer | Chi tiết | Output Shape | Params |
|--------|-------|----------|-------------|--------|
| 1 | `Flatten` | 7×7×512 → 25088 | (batch, 25088) | 0 |
| 2 | `BatchNorm` (bn_flatten) | momentum=0.9, ε=2e-5 | (batch, 25088) | 100,352 |
| 3 | `Dropout` | rate=0.3 (chỉ khi train) | (batch, 25088) | 0 |
| 4 | `Dense` (fc_embedding) | units=512, no bias | (batch, 512) | 12,845,056 |
| 5 | `BatchNorm` (bn_embedding) | scale=False, momentum=0.9 | (batch, 512) | 1,024 |

> **Tại sao `scale=False` ở BN cuối?**
> Vì ArcFace sẽ L2-normalize embedding, nên giá trị scale (γ) của BN là dư thừa. Bỏ đi giúp giảm tham số và tránh xung đột gradient.

---

## 3. ArcFace Loss Layer (Chỉ dùng khi Training)

```mermaid
flowchart LR
    subgraph INPUTS["Inputs"]
        EMB["Embedding\n(batch, 512)"]
        LBL["Label\n(batch,)"]
    end

    subgraph ARCFACE_COMPUTE["ArcFace Computation"]
        L2_EMB["L2 Normalize\nEmbedding"]
        L2_W["L2 Normalize\nWeights (512 × C)"]
        COS["cos(θ) = emb · W\n(batch, C)"]
        SIN["sin(θ) = √(1 - cos²θ)"]
        COS_M["cos(θ + m)\n= cosθ·cos(m) - sinθ·sin(m)"]
        ONE_HOT["One-Hot Labels\n(batch, C)"]
        MERGE["Chọn:\n✅ Đúng class → cos(θ+m)\n❌ Sai class → cos(θ)"]
        SCALE["× Scale (s=64)"]
    end

    LOGITS["Logits\n(batch, C)"]
    LOSS["SparseCategoricalCrossentropy\n(from_logits=True)"]

    EMB --> L2_EMB --> COS
    L2_W --> COS
    COS --> SIN
    COS --> COS_M
    SIN --> COS_M
    LBL --> ONE_HOT --> MERGE
    COS_M --> MERGE
    COS --> MERGE
    MERGE --> SCALE --> LOGITS --> LOSS

    style ARCFACE_COMPUTE fill:#1a1a2e,stroke:#e94560,color:#eee
    style SCALE fill:#e94560,stroke:#fff,color:#fff
```

### Công thức toán học:

$$\text{ArcFace}(x_i, y_i) = -\log \frac{e^{s \cdot \cos(\theta_{y_i} + m)}}{e^{s \cdot \cos(\theta_{y_i} + m)} + \sum_{j \neq y_i} e^{s \cdot \cos\theta_j}}$$

| Tham số | Giá trị | Mô tả |
|---------|---------|-------|
| `s` (scale) | 64 | Phóng đại gradient, giúp hội tụ nhanh |
| `m` (margin) | 0.0 → 0.5 | Tăng dần qua từng epoch (Progressive Margin) |
| `W` (weights) | 512 × num_classes | Ma trận trọng số đại diện cho từng class |

---

## 4. Tóm tắt luồng dữ liệu (Data Flow Summary)

```
Ảnh 112×112×3
    │
    ▼
┌─────────────────────────────────────────────┐
│  STEM: Conv2D(64, 3×3) → BN → PReLU        │  112×112×64
├─────────────────────────────────────────────┤
│  STAGE 1: IR Block × 3  (filters=64)       │  56×56×64
├─────────────────────────────────────────────┤
│  STAGE 2: IR Block × 4  (filters=128)      │  28×28×128
├─────────────────────────────────────────────┤
│  STAGE 3: IR Block × 14 (filters=256)      │  14×14×256
├─────────────────────────────────────────────┤
│  STAGE 4: IR Block × 3  (filters=512)      │  7×7×512
├─────────────────────────────────────────────┤
│  HEAD:                                      │
│    Flatten           → (25088)              │
│    BatchNorm         → (25088)              │
│    Dropout(0.3)      → (25088)              │
│    Dense(512)        → (512)                │
│    BatchNorm(γ=off)  → (512)   ← EMBEDDING │
└─────────────────────────────────────────────┘
    │
    ├── [Training] ──→ ArcFace(margin, scale=64) → Logits(num_classes) → Loss
    │
    └── [Inference] ──→ L2 Normalize → 512-D Vector → Cosine Similarity
```

---

## 5. So sánh với ResNet-50 gốc (ImageNet)

| Đặc điểm | ResNet-50 (ImageNet) | iResNet-50 (InsightFace) |
|-----------|---------------------|-------------------------|
| Input size | 224×224 | **112×112** |
| Stem | Conv 7×7 + MaxPool | **Conv 3×3 (no pool)** |
| Block type | Bottleneck (1×1→3×3→1×1) | **IR Block (3×3→3×3)** |
| Activation | ReLU | **PReLU** |
| Block structure | Conv → BN → ReLU | **BN → Conv → BN → PReLU → Conv → BN** |
| Stages (blocks) | [3, 4, 6, 3] = 16 | **[3, 4, 14, 3] = 24** |
| Output head | GAP → Dense(1000) | **Flatten → BN → Dropout → Dense(512) → BN** |
| Loss function | Softmax Cross-Entropy | **ArcFace (Angular Margin)** |
| Tối ưu cho | Phân loại tổng quát | **Trích xuất đặc trưng khuôn mặt** |

> **Tại sao không dùng MaxPool ở Stem?**
> Vì ảnh input chỉ 112×112, MaxPool sẽ mất quá nhiều thông tin spatial ngay từ đầu. iResNet dùng Conv stride=2 ở block đầu tiên của mỗi Stage để giảm kích thước một cách "mềm" hơn, giữ lại nhiều chi tiết khuôn mặt hơn.
