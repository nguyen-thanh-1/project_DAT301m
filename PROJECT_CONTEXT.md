# 🧠 Face Recognition Project Context

## 1. Project Overview & Ultimate Goals
**Tên dự án:** Tối ưu hóa kiến trúc Nhận diện khuôn mặt (Face Recognition) cho học One-shot 

**Mục tiêu bài báo (Dual-Contribution):**
1. **Đóng góp 1 (Architecture & Losses):** Xây dựng và so sánh hiệu năng của các hàm mất mát dựa trên lề góc (Angular Margin Losses: ArcFace, CosFace, SphereFace, Softmax) trên cùng một kiến trúc Backbone (iResNet).
2. **Đóng góp 2 (One-shot Learning):** Đánh giá khả năng tổng quát hóa của không gian nhúng (embedding space) sinh ra từ mô hình khi nhận diện các danh tính **hoàn toàn mới, chưa từng xuất hiện** trong quá trình huấn luyện.

---

## 2. Dataset Status (VGGFace2 Subset)
Dữ liệu đã được tiền xử lý (MTCNN crop & align 112×112) và phân tách ĐÚNG CHUẨN cho tác vụ Face Verification. Tổng cộng **318,822 ảnh** thuộc **540 identity** khác nhau:

- **`dataset_final/train/`**: **432 classes** — **248,400 ảnh**. Tập train được cân bằng hoàn toàn (perfectly balanced) với mỗi class đúng **575 ảnh** (CV = 0.0000). Dùng để train classification với ArcFace head.
- **`dataset_final/val/`**: **432 classes** (**CÙNG** danh tính với train) — **31,327 ảnh**. Số ảnh/class dao động từ 30–143 (mean = 72.5, std = 20.63, CV = 0.2845 — slightly imbalanced). Dùng để theo dõi độ hội tụ của mô hình (convergence monitoring).
- **`dataset_final/test/`**: **108 classes** (**HOÀN TOÀN MỚI / UNSEEN**, 0 identity trùng với train/val) — **39,095 ảnh**. Số ảnh/class dao động từ 101–642 (mean = 362.0, std = 115.28, CV = 0.3185). Dùng làm tập sinh các cặp Positive/Negative để đánh giá One-shot Verification.

**Tỷ lệ phân chia:** Train ~77.9% | Val ~9.8% | Test ~12.3%.
**Kích thước ảnh:** Toàn bộ đồng nhất **112×112 pixel**, sẵn sàng đưa vào mô hình.

---

## 3. Current Architecture & Pipeline
Toàn bộ mã nguồn được viết bằng **TensorFlow 2.x**. Pipeline đã được tối ưu hóa cho việc chạy thực nghiệm:

  Đầu ra là Embedding vector (kích thước 512) đã được chuẩn hóa L2.
- **Classification Head (Loss Layer):** Đầu ra của Backbone được đưa qua các lớp tùy chỉnh (Custom Layers) hỗ trợ 4 thuật toán: `ArcFace`, `CosFace`, `SphereFace`, `Softmax`. Margin được tăng dần trong lúc train qua `ProgressiveMarginCallback`.

**B. Đánh giá (Evaluation Metrics):**
- **Training Phase:** Dùng `SparseCategoricalCrossentropy` và `Accuracy`.
- **Verification Phase:** Thông qua `VerificationCallback` và script `eval_verification.py`. Đánh giá bằng **Cosine Similarity**. Các metric xuất ra chuẩn paper: **EER** (Equal Error Rate), **TAR@FAR**, **AUC**, kèm biểu đồ **ROC Curve** và phân bố điểm.
- **Visualization:** Có script `visualize_embeddings.py` để vẽ **t-SNE/PCA** không gian embedding và Heatmap khoảng cách Cosine.

**C. Thực nghiệm tự động:**
- Script `run_experiments.py` tự động chạy các cấu hình Loss khác nhau, lưu log, đánh giá và backup weights tốt nhất vào thư mục `experiments/`.

---

## 4. Project Directory Layout

Dự án được cấu trúc dạng module hóa chuyên nghiệp để quản lý nhiều cấu hình mô hình khác nhau:

- **`data_processing/`**:
  - `preprocess.py`: Cắt ảnh theo MTCNN và chuẩn hóa 112×112.
  - `prepare_dataset.py`: Phân chia dữ liệu train/val/test theo chuẩn one-shot.
  - `check_dataset_aligned.ipynb`: Notebook kiểm tra số lượng ảnh từng class.
  - `dataset_audit.py`: Công cụ kiểm tra tính toàn vẹn dữ liệu.
  - `dataset_statistics.py`: Script thống kê dataset toàn diện (phân bố, overlap, biểu đồ).
  - `dataset_description.md`: Mô tả thống kê dataset dạng văn bản.
- **`shared/`**:
  - `utils.py`: Các callbacks, learning rate scheduler, data generator dùng chung.
  - `eval_verification.py`: Đánh giá One-shot Verification (EER, ROC, TAR@FAR).
  - `visualize_embeddings.py`: Vẽ t-SNE/PCA và heatmap tương đồng.
- **`models/face_recognition/`**:
  - **`iresnet/`**:
    - `iresnet50.py`: Mạng iResNet50 Backbone.
    - `train.py`: Quy trình train riêng của iResNet.
    - `train_notebook.ipynb`: Notebook chạy thực nghiệm iResNet.
    - `results/`: Checkpoint, training history plot và verification curves.
  - **`hybridfacenet/`**:
    - `hybridfacenet.py`: Mạng lai CNN-Transformer.
    - `train.py`: Quy trình train riêng của HybridFaceNet.
    - `evaluate.py`: Script đánh giá riêng.
    - `train_hybridfacenet_notebook.ipynb`: Notebook chạy thực nghiệm.
    - `results/`: Checkpoint, training history plot và verification curves.
  - **`mobilefacenet/`**:
    - `mobile_face_net.py`: Mạng MobileFaceNet Backbone.
    - `train.py`: Quy trình train riêng của MobileFaceNet.
    - `train_mobilefacenet_notebook.ipynb`: Notebook chạy thực nghiệm.
    - `results/`: Checkpoint, training history plot và verification curves.
- **`run_experiments.py`**: Chạy thử nghiệm so sánh tự động tất cả các cấu hình từ root.
