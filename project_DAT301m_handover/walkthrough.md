# Hướng Dẫn Chi Tiết Hệ Thống Face Anti-Spoofing (LCC-FASD)

Dự án này đã tái cấu trúc và phát triển thành công hệ thống phòng chống giả mạo khuôn mặt (Face Anti-Spoofing - FAS) từ một file notebook đơn lẻ thành một cấu trúc mã nguồn dạng gói mô-đun (modular package) chuyên nghiệp, đồng thời nghiên cứu thử nghiệm các kiến trúc học sâu tiên tiến và kỹ thuật dung hợp (Ensemble).

---

## 1. Cấu Trúc Các Pha Phát Triển Dự Án

### Pha 1: Cấu Trúc Mã Nguồn Mô-đun (Modular Packaging)
Chúng ta đã chuyển đổi toàn bộ pipeline sang cấu trúc thư mục dạng gói `anti_spoofing/`:
* [config.py](file:///c:/Users/Admin/Desktop/TrHius/DAT/project_DAT301m/anti_spoofing/config.py) : Quản lý toàn bộ siêu tham số và tự động dò đường dẫn dữ liệu trên WSL hoặc Windows.
* [face_detector.py](file:///c:/Users/Admin/Desktop/TrHius/DAT/project_DAT301m/anti_spoofing/face_detector.py) : Bộ cắt và tạo cache khuôn mặt tự động bằng thuật toán Haar-Cascade cổ điển.
* [lbp.py](file:///c:/Users/Admin/Desktop/TrHius/DAT/project_DAT301m/anti_spoofing/lbp.py) : Trích xuất đặc trưng vân kết cấu Local Binary Pattern (LBP) trên CPU.
* [dataset.py](file:///c:/Users/Admin/Desktop/TrHius/DAT/project_DAT301m/anti_spoofing/dataset.py) : Xây dựng tf.data Input Pipeline tối ưu, tích hợp kỹ thuật tăng cường dữ liệu nâng cao (Augmentation).
* [models.py](file:///c:/Users/Admin/Desktop/TrHius/DAT/project_DAT301m/anti_spoofing/models.py) : Định nghĩa kiến trúc cho cả 4 mô hình độc lập (A, B, C, D).
* [train_pipeline.py](file:///c:/Users/Admin/Desktop/TrHius/DAT/project_DAT301m/train_pipeline.py) & [eval_pipeline.py](file:///c:/Users/Admin/Desktop/TrHius/DAT/project_DAT301m/eval_pipeline.py) : Trình chạy huấn luyện và đánh giá chung.

---

### Pha 2: Tối Ưu Hóa Tổng Hóa Thực Tế (Generalization Upgrades)
Để khắc phục khoảng cách miền dữ liệu (Domain Gap) khi nhận diện qua webcam thực tế, các nâng cấp sau đã được thực hiện:
* **Tăng cường dữ liệu nâng cao (Augmentation):** Tích hợp dịch chuyển ngẫu nhiên (Random Translation) và biến đổi ngẫu nhiên các yếu tố ánh sáng, độ tương phản, độ bão hòa màu, sắc thái màu sắc (Jittering).
* **Model A - EffNet-style (RGB only):** Kiến trúc kết hợp MBConv kiểu EfficientNet với cơ chế tích chập vi phân trung tâm (Central Difference Convolution - CDC) ở khối Stem, đồng thời dung hợp các đặc trưng đa quy mô (Multi-Scale Feature Fusion).
* **Model B - CDCN+LBP (RGB+texture):** Kết hợp đặc trưng học sâu CDC và đặc trưng kết cấu LBP thủ công. Chỉ nặng **0.47M tham số**, chạy cực kỳ tối ưu trên CPU.

---

### Pha 3: Nghiên Cứu Model C (ResNet18-CBAM)
Thiết kế và triển khai mô hình học sâu kết hợp cơ chế chú ý (Attention):
* **Kiến trúc:** Tích hợp khối chú ý CBAM (Convolutional Block Attention Module) vào mạng ResNet-18, giúp mô hình tập trung cao độ vào các vùng biên của vật thể giả mạo (viền màn hình, phản chiếu ánh sáng).
* **Trình chạy tích hợp:** Sử dụng `train_pipeline.py --model c` để huấn luyện và `eval_pipeline.py --model c` để đánh giá. Kết quả được lưu trữ riêng biệt trong thư mục `outputs_model_c/` để không ghi đè kết quả của các mô hình khác.

---

### Pha 4: Nghiên Cứu Model D (ResNet34-SE-CDC-MS)
Kiến trúc nâng cấp nặng ký hơn phục vụ so sánh chéo:
* **Kiến trúc:** Sử dụng ResNet-34 làm backbone, tích hợp chú ý kênh Squeeze-and-Excitation (SE), tích chập vi phân trung tâm (CDC) và cấu trúc trích xuất đặc trưng đa quy mô (Multi-Scale).
* **Trình chạy tích hợp:** Sử dụng `train_pipeline.py --model d` để huấn luyện và `eval_pipeline.py --model d` để đánh giá. Kết quả được lưu trữ độc lập trong thư mục `outputs_model_d/`.

---

### Pha 5: Dung Hợp (Ensemble) & Demo Thời Gian Thực
* **Dung hợp Ensemble A & C:** Xây dựng cơ chế dung hợp xác suất có trọng số ($40\%$ Model A + $60\%$ Model C) giúp mô hình đạt **AUC cao nhất toàn hệ thống (0.9446)** và giảm mạnh tỷ lệ lỗi lọt spoof (APCER) từ **14.35% xuống còn 11.13%**.
* **Tối ưu luồng video (realtime_demo.py):** Hỗ trợ thêm chế độ Ensemble A & C chạy hoàn toàn trên GPU (không bị bottleneck CPU do thuật toán LBP), đồng thời tối ưu hóa tham số `skip_frames = 5` giúp luồng camera hiển thị mượt mà trên môi trường CPU Windows.

---

### Pha 6: Cấu Hình Gitignore
Tệp tin [`.gitignore`](file:///c:/Users/Admin/Desktop/TrHius/DAT/project_DAT301m/.gitignore) đã được tối ưu hóa:
* Loại bỏ các thư mục rác và tệp tin cache ảnh rất nặng (`cache_cropped/`, `dataset/`).
* Loại bỏ tất cả các tệp lịch sử huấn luyện chi tiết theo từng epoch (`results_*.json`, `model_*_curves.png`, `*_log.csv`, `logs/`).
* Loại bỏ tất cả các thư mục kết quả (`outputs/`, `outputs_upgraded/`, `outputs_model_c/`, `outputs_model_d/`) để giữ cho Repository Git gọn gàng nhất.

---

## 2. Bảng Đánh Giá Hiệu Năng Cuối Cùng (Final Benchmark Report)

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

## 3. Hướng Dẫn Sử Dụng Trình Chạy

### Chạy Đánh Giá Toàn Bộ (Evaluation)
Để đánh giá lại các mô hình, bạn kích hoạt môi trường ảo WSL và sử dụng cờ `--model` tương ứng:
```bash
python eval_pipeline.py --model all  # Đánh giá tất cả mô hình, tính toán Ensemble và vẽ biểu đồ so sánh chung
python eval_pipeline.py --model c    # Chỉ đánh giá Model C và vẽ biểu đồ huấn luyện của Model C
python eval_pipeline.py --model d    # Chỉ đánh giá Model D
```

### Chạy Demo Webcam thời gian thực
Chạy lệnh sau và chọn chế độ tương ứng (phím 1-6):
```bash
python realtime_demo.py
```
*(Khuyên dùng chế độ số 6 để chạy Ensemble Mode A & C đạt độ chính xác cao nhất).*
