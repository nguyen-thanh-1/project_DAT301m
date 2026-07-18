# Báo Cáo Tổng Hợp Các Thực Nghiệm Face Anti-Spoofing (LCC-FASD)

Báo cáo này tổng hợp kết quả của toàn bộ các lần thực nghiệm thiết kế, huấn luyện và đánh giá mô hình chống giả mạo khuôn mặt trên bộ dữ liệu **LCC-FASD**. Các thực nghiệm bao gồm phiên bản gốc, phiên bản nâng cấp, các kiến trúc cải tiến sâu (Model C, D) và các mô hình tham khảo từ tài liệu khoa học.

---

## 1. Bảng Tổng Hợp Kết Quả Thực Nghiệm (Master Table)

Dưới đây là bảng thống kê chi tiết các chỉ số đánh giá chính xác trên tập kiểm thử (Evaluation set) của tất cả các mô hình:

| Nhóm Mô Hình | Tên Mô Hình | AUC (Cao = Tốt) | EER% (Thấp = Tốt) | ACER% (Thấp = Tốt) | APCER% (Lỗi lọt spoof) | BPCER% (Lỗi phạt nhầm) | Số Tham Số (Params) | File Kết Quả |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Phát Triển (Nâng Cấp)** | **Ensemble Mode A & C (40% A + 60% C)** | **0.9446** | 12.17% | 11.94% | 11.13% | 12.74% | ~4.74M | [benchmark.csv](file:///c:/Users/Admin/Desktop/TrHius/DAT/project_DAT301m/outputs_upgraded/final_benchmark.csv) |
| | **Model C - ResNet18-CBAM (RGB+CBAM)** | 0.9443 | **11.78%** | **11.80%** | 14.35% | **9.24%** | ~2.85M | [results_c.json](file:///c:/Users/Admin/Desktop/TrHius/DAT/project_DAT301m/outputs_model_c/results_c.json) |
| | **Model A - EffNet-style (RGB only)** | 0.9250 | 13.44% | 12.96% | **8.73%** | 17.20% | ~1.89M | [results_a.json](file:///c:/Users/Admin/Desktop/TrHius/DAT/project_DAT301m/outputs_upgraded/results_a.json) |
| | **Model D - ResNet34-SE-CDC-MS** | 0.9214 | 14.00% | 15.36% | 17.34% | 13.38% | ~5.42M | [results_d.json](file:///c:/Users/Admin/Desktop/TrHius/DAT/project_DAT301m/outputs_model_d/results_d.json) |
| | **Model B - CDCN+LBP (RGB+texture)** | 0.8862 | 18.84% | 18.44% | 12.36% | 24.52% | **0.47M** | [results_b.json](file:///c:/Users/Admin/Desktop/TrHius/DAT/project_DAT301m/outputs_upgraded/results_b.json) |
| **Phát Triển (Gốc)** | **Model A - EffNet-style (RGB only)** | 0.9027 | 16.86% | 16.83% | 7.86% | 25.80% | ~1.89M | [results_a.json](file:///c:/Users/Admin/Desktop/TrHius/DAT/project_DAT301m/outputs/results_a.json) |
| | **Model B - CDCN+LBP (RGB+texture)** | 0.8640 | 20.11% | 20.62% | 9.40% | 31.85% | **0.47M** | [results_b.json](file:///c:/Users/Admin/Desktop/TrHius/DAT/project_DAT301m/outputs/results_b.json) |
| **Tham Chiếu (Tài Liệu)** | *MobileNetV3-Large* | 0.9210 | 16.10% | 16.30% | 17.30% | 15.40% | ~3.00M | [benchmark.csv](file:///c:/Users/Admin/Desktop/TrHius/DAT/project_DAT301m/outputs_upgraded/final_benchmark.csv) |
| | *MobileNetV3-Small* | 0.8890 | 18.70% | 19.70% | 14.80% | 24.60% | ~1.00M | [benchmark.csv](file:///c:/Users/Admin/Desktop/TrHius/DAT/project_DAT301m/outputs_upgraded/final_benchmark.csv) |

---

## 2. Phân Tích Và So Sánh Các Lần Thực Nghiệm

> [!NOTE]
> **Chỉ số ACER (Average Classification Error Rate):** Là trung bình cộng của APCER và BPCER. ACER càng thấp nghĩa là mô hình hoạt động càng chính xác và ổn định.

### A. Hiệu Quả Từ Việc Nâng Cấp Mô Hình (Gốc vs Nâng Cấp)
* **Model A:** Phiên bản nâng cấp đã cải thiện chỉ số **ACER từ 16.83% xuống 12.95%** (giảm gần 4% tỷ lệ lỗi). Độ phân biệt AUC tăng từ 0.9027 lên 0.9250.
* **Model B:** Chỉ số **ACER cải thiện từ 20.62% xuống 18.44%**. Lỗi phạt nhầm người thật (BPCER) giảm mạnh từ 31.85% xuống 24.52%, giúp mô hình thực tế hoạt động mượt mà hơn.
* **Kết luận:** Các thay đổi về cấu hình học sâu, tiền xử lý và tinh chỉnh siêu tham số ở phiên bản nâng cấp đều mang lại hiệu quả rõ rệt.

### B. Vai Trò Của Cơ Chế Chú Ý (Model C - ResNet18-CBAM)
* Model C đạt độ chính xác cao nhất trong tất cả các thực nghiệm (**ACER = 11.80%, AUC = 0.9443**).
* Việc tích hợp khối chú ý không gian và kênh (CBAM) giúp mô hình tập trung sâu vào các vùng chi tiết dễ phân biệt đặc trưng giả mạo (như viền màn hình điện thoại, nếp gấp giấy, phản chiếu ánh sáng trên mặt giả) thay vì các vùng nền không liên quan.
* Đặc biệt, Model C kiểm soát lỗi phạt nhầm người thật cực kỳ tốt (**BPCER = 9.24%**), lý tưởng cho các ứng dụng thực tế.

### C. Đánh Giá Mô Hình Phức Tạp (Model D - ResNet34-SE-CDC-MS)
* Mặc dù kết hợp nhiều cơ chế hiện đại (CDC, SE, Multi-scale) và có độ sâu lớn (ResNet34), Model D đạt chỉ số ACER ở mức trung bình khá (**15.37%**), tốt hơn MobileNetV3-Large nhưng kém hơn Model C và Model A.
* Điều này chỉ ra rằng đối với bộ dữ liệu LCC-FASD, một kiến trúc vừa phải như ResNet18 hoặc EfficientNet kết hợp attention hiệu quả hơn là một mô hình quá sâu (Over-parameterization) dễ dẫn đến hiện tượng bão hòa hoặc quá khớp nhẹ.

---

## 3. Bản Đồ Tương Quan Giữa Độ Chính Xác Và Kích Thước (Params)

Sự phân bổ của các mô hình phát triển được chia làm 3 phân khúc rõ rệt:
1. **Phân khúc Siêu Nhẹ (Lightweight):** **Model B (0.47M params)** - Phù hợp cho thiết bị nhúng di động, máy chủ CPU yếu. ACER ở mức chấp nhận được (18.44%).
2. **Phân khúc Cân Bằng (Best Trade-off):** **Model A (1.89M params)** - Đạt hiệu năng cực tốt (ACER 12.95%) với lượng tham số hợp lý, tối ưu cho cả CPU mạnh và GPU.
3. **Phân khúc Độ Chính Xác Cao (High-accuracy):** **Model C (2.85M params)** - Dành cho các hệ thống yêu cầu bảo mật cao nhất, chấp nhận kích thước mô hình trung bình (ACER 11.80%).

---

## 4. Tài Liệu Tham Khảo Và Trích Dẫn Nguồn (References)

Các chỉ số của mô hình thuộc nhóm **Tham Chiếu (Literature)** được thu thập chính xác từ nguồn mã nguồn mở sau đây:

1. **MobileNetV3-Large & MobileNetV3-Small (light-weight-FAS repo)**
   * **Nguồn:** Dự án benchmarking mã nguồn mở `light-weight-face-anti-spoofing` của Kirill Prokofiev.
   * **GitHub:** [kprokofi/light-weight-face-anti-spoofing](https://github.com/kprokofi/light-weight-face-anti-spoofing)
   * **Chi tiết:** Các chỉ số đánh giá trên bộ dữ liệu LCC-FASD (ACER, APCER, BPCER, EER) được trích dẫn trực tiếp từ báo cáo kết quả huấn luyện mô hình của repository này.
