# Tài Liệu Bàn Giao Dự Án Face Anti-Spoofing (LCC-FASD)

Tài liệu này được lập ra nhằm tổng hợp toàn bộ thông tin mã nguồn, trọng số mô hình đã huấn luyện, kết quả thực nghiệm và hướng dẫn chi tiết giúp người nhận bàn giao có thể tích hợp, sử dụng hoặc tiếp tục phát triển dự án này một cách dễ dàng nhất.

---

## 1. Thành Phần Mã Nguồn (Codebase Files)

Các tệp mã nguồn cốt lõi trong dự án được tổ chức như sau:

* **Thư mục mô-đun chính `anti_spoofing/`:**
  * [config.py](file:///c:/Users/Admin/Desktop/TrHius/DAT/project_DAT301m/anti_spoofing/config.py) : Định nghĩa kích thước ảnh ($224 \times 224$), kích thước Batch ($32$), số lớp phân loại và đường dẫn dữ liệu.
  * [models.py](file:///c:/Users/Admin/Desktop/TrHius/DAT/project_DAT301m/anti_spoofing/models.py) : Chứa các hàm xây dựng kiến trúc Model A (EffNet), Model B (CDCN+LBP), Model C (ResNet18-CBAM) và Model D (ResNet34-SE-CDC-MS).
  * [dataset.py](file:///c:/Users/Admin/Desktop/TrHius/DAT/project_DAT301m/anti_spoofing/dataset.py) : Tiền xử lý dữ liệu và tăng cường hình ảnh (Augmentations).
  * [face_detector.py](file:///c:/Users/Admin/Desktop/TrHius/DAT/project_DAT301m/anti_spoofing/face_detector.py) : Tự động crop và lưu trữ bộ nhớ đệm (cache) khuôn mặt để tăng tốc độ chạy.
  * [lbp.py](file:///c:/Users/Admin/Desktop/TrHius/DAT/project_DAT301m/anti_spoofing/lbp.py) : Xử lý trích xuất đặc trưng kết cấu LBP cục bộ.
* **Các file chạy chính ở thư mục gốc:**
  * [train_pipeline.py](file:///c:/Users/Admin/Desktop/TrHius/DAT/project_DAT301m/train_pipeline.py) : Script thống nhất để huấn luyện bất kỳ mô hình nào qua cờ `--model`.
  * [eval_pipeline.py](file:///c:/Users/Admin/Desktop/TrHius/DAT/project_DAT301m/eval_pipeline.py) : Script thống nhất để đánh giá và chạy Ensemble.
  * [realtime_demo.py](file:///c:/Users/Admin/Desktop/TrHius/DAT/project_DAT301m/realtime_demo.py) : Chạy thử nghiệm camera thời gian thực với đầy đủ 6 chế độ chạy thử.
  * [walkthrough.md](file:///c:/Users/Admin/Desktop/TrHius/DAT/project_DAT301m/walkthrough.md) : Hướng dẫn chi tiết từng pha phát triển.

---

## 2. Danh Sách Tệp Trọng Số (Pre-trained Weights) & Cách Chia Sẻ

Toàn bộ trọng số tốt nhất (Best checkpoints) được lưu tại các đường dẫn cục bộ sau:

1. **Model A (EffNet-style):** `checkpoints/model_a_effnet_style/best_weights.weights.h5`
2. **Model B (CDCN+LBP):** `checkpoints/model_b_cdcn_lbp/best_weights.weights.h5`
3. **Model C (ResNet18-CBAM):** `checkpoints/model_c_resnet_cbam/best_weights.weights.h5`
4. **Model D (ResNet34-SE-CDC-MS):** `checkpoints/model_d_resnet_se/best_weights.weights.h5`

> [!IMPORTANT]
> **Cách chuyển file mô hình (trọng số) cho người khác:**
> Do các tệp trọng số `.weights.h5` có dung lượng lớn (tổng cộng khoảng hơn 100MB) và đã được cấu hình tự động bỏ qua trong `.gitignore` để tránh làm nặng kho lưu trữ Git, người nhận code sẽ **không có** sẵn các file này nếu họ chỉ `git clone`. Bạn có thể gửi cho họ bằng 2 cách:
> 
> * **Cách 1 (Gửi qua File Nén Zip):** Nếu bạn bàn giao dự án bằng cách nén thư mục, hãy nén cả thư mục `project_DAT301m` và kiểm tra chắc chắn đã chứa thư mục `checkpoints/` ở bên trong.
> * **Cách 2 (Tải lên Google Drive / OneDrive):** Bạn upload nguyên thư mục `checkpoints/` lên Google Drive, tạo link chia sẻ công khai (Anyone with the link can view) và gửi kèm cho người nhận code. Người nhận chỉ cần tải về, tạo thư mục `checkpoints/` ở gốc dự án và đặt các thư mục con tương ứng vào đó.

---

## 3. Bảng Kết Quả Đánh Giá Tổng Hợp (Benchmark Table)

Kết quả đo đạc chính xác trên tập kiểm thử **LCC-FASD** (gồm 7.580 hình ảnh):

| Thứ Hạng | Tên Mô Hình | AUC (Cao = Tốt) | EER% (Thấp = Tốt) | ACER% (Thấp = Tốt) | APCER% (Lỗi lọt spoof) | BPCER% (Phạt nhầm) | Kích Thước (Params) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🏆 (AUC) | **Ensemble Mode A & C (ours)** | **0.9446** | 12.17% | 11.94% | 11.13% | 12.74% | ~4.74M |
| 🥈 (ACER) | **Model C - ResNet18-CBAM (ours)** | 0.9443 | **11.78%** | **11.80%** | 14.35% | **9.24%** | ~2.85M |
| 🥉 | **Model A - EffNet-style (ours)** | 0.9250 | 13.44% | 12.96% | **8.73%** | 17.20% | ~1.89M |
| 4 | **Model D - ResNet34-SE-CDC-MS (ours)** | 0.9214 | 14.00% | 15.36% | 17.34% | 13.38% | ~5.42M |
| 5 | *[Literature] MobileNetV3-Large* | 0.9210 | 16.10% | 16.30% | 17.30% | 15.40% | ~3.00M |
| 6 | **Model B - CDCN+LBP (ours)** | 0.8862 | 18.84% | 18.44% | 12.36% | 24.52% | **0.47M** |
| 7 | *[Literature] MobileNetV3-Small* | 0.8890 | 18.70% | 19.70% | 14.80% | 24.60% | ~1.00M |

---

## 4. Hướng Dẫn Vận Hành Hệ Thống

### A. Cài đặt Môi trường
Dự án sử dụng Python 3.10 và TensorFlow 2.11+. Để cài đặt các thư viện phụ thuộc:
```bash
pip install tensorflow opencv-python matplotlib numpy
```

### B. Huấn luyện Mô hình
Để huấn luyện mô hình mong muốn, chạy lệnh:
```bash
python train_pipeline.py --model [a/b/c/d/all]
```
*(Nếu chọn `all`, hệ thống sẽ huấn luyện tuần tự cả 4 mô hình).*

### C. Đánh giá Mô hình & Tạo Bảng So Sánh ROC/DET
Để chạy đánh giá tập test và tự động tính toán mô hình Ensemble:
```bash
python eval_pipeline.py --model all
```

### D. Khởi chạy Camera Kiểm thử Thời gian thực
```bash
python realtime_demo.py
```
Nhập phím từ `1` đến `6` để chọn mô hình thử nghiệm.

---

## 5. Hướng Dẫn Tích Hợp Vào Mã Nguồn Khác (Inference Snippet)

Dành cho nhà phát triển muốn import và dự đoán trực tiếp trên 1 bức ảnh mới bằng Python:

```python
import tensorflow as tf
import numpy as np
from anti_spoofing.config import Config
from anti_spoofing.models import build_resnet18_cbam

def run_face_anti_spoofing(image_path, weights_path):
    # 1. Khởi tạo cấu hình và kiến trúc mô hình
    cfg = Config()
    model = build_resnet18_cbam(cfg.img_size, cfg.num_classes)
    
    # 2. Tải trọng số đã huấn luyện
    model.load_weights(weights_path)
    
    # 3. Đọc và tiền xử lý ảnh
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, cfg.img_size)
    img = tf.cast(img, tf.float32) / 255.0  # Chuẩn hóa về [0, 1]
    img = tf.expand_dims(img, axis=0)       # Tạo chiều Batch (1, 224, 224, 3)
    
    # 4. Dự đoán xác suất
    logits = model({"rgb": img}, training=False)
    probs = tf.nn.softmax(logits, axis=-1)
    real_prob = probs[0, 1].numpy()  # Xác suất là mặt thật (Class 1)
    
    # 5. Phân loại kết quả
    if real_prob > 0.5:
        print(f"[KẾT QUẢ] Mặt THẬT (Real Face) | Độ tin cậy: {real_prob*100:.2f}%")
    else:
        print(f"[KẾT QUẢ] GIẢ MẠO (Spoof Attack) | Xác suất giả mạo: {(1-real_prob)*100:.2f}%")
    return real_prob

# Ví dụ thực thi:
# run_face_anti_spoofing("test_face.jpg", "checkpoints/model_c_resnet_cbam/best_weights.weights.h5")
```
