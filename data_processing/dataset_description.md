# Mô tả thống kê Dataset

## Dataset Face Recognition - Project DAT301m

Dataset được sử dụng trong đồ án là một tập dữ liệu khuôn mặt đã qua tiền xử lý face alignment, bao gồm tổng cộng **318,822 ảnh** khuôn mặt thuộc **540 identity (người)** khác nhau. Toàn bộ ảnh có kích thước chuẩn **112×112 pixel**, phù hợp cho các kiến trúc face recognition hiện đại như ArcFace/iResNet.

Dataset được chia thành 3 tập con:

- **Tập huấn luyện (train):** Gồm **248,400 ảnh** thuộc **432 identity**. Tập này được cân bằng hoàn toàn (perfectly balanced) với mỗi identity có đúng **575 ảnh**, hệ số biến thiên CV = 0.0000. Việc cân bằng này giúp mô hình học đồng đều đặc trưng khuôn mặt của tất cả các identity mà không bị thiên vị (bias) về những lớp có nhiều mẫu hơn.

- **Tập kiểm định (validation):** Gồm **31,327 ảnh** thuộc **432 identity** — cùng tập identity với tập train. Số ảnh mỗi identity dao động từ **30 đến 143** (trung bình 72.5, độ lệch chuẩn 20.63), hệ số biến thiên CV = 0.2845, cho thấy mức độ mất cân bằng nhẹ. Tập val dùng để theo dõi hiệu suất mô hình trong quá trình huấn luyện và tránh overfitting.

- **Tập kiểm thử (test):** Gồm **39,095 ảnh** thuộc **108 identity** — hoàn toàn khác biệt so với tập train/val (0 identity trùng lặp). Số ảnh mỗi identity dao động từ **101 đến 642** (trung bình 362.0, độ lệch chuẩn 115.28), hệ số biến thiên CV = 0.3185. Việc sử dụng các identity chưa từng xuất hiện trong quá trình huấn luyện đảm bảo đánh giá khả năng tổng quát hóa (generalization) thực sự của mô hình — đây là yếu tố then chốt trong bài toán face verification/recognition.

**Tỷ lệ phân chia:** Train chiếm khoảng **77.9%**, Val chiếm **9.8%**, và Test chiếm **12.3%** tổng số ảnh trong dataset.

**Đặc điểm nổi bật:**
- Tập train được cân bằng hoàn hảo, phù hợp cho việc huấn luyện với ArcFace loss.
- Tập test sử dụng identity hoàn toàn mới (open-set evaluation), mô phỏng đúng kịch bản triển khai thực tế của hệ thống nhận diện khuôn mặt.
- Toàn bộ ảnh đã được align về kích thước 112×112, sẵn sàng đưa vào mô hình mà không cần thêm bước tiền xử lý.
