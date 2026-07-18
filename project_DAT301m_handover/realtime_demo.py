import os
import cv2
import numpy as np
import tensorflow as tf
from anti_spoofing.config import Config
from anti_spoofing.face_detector import detect_and_crop_face
from anti_spoofing.lbp import lbp_numpy_wrapper
from anti_spoofing.models import build_efficientnet_style, build_cdcn_lbp, build_resnet18_cbam, build_resnet34_se_cdc_multiscale

def main():
    # Tắt log cảnh báo từ absl
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    cfg = Config()
    
    print("="*50)
    print("CHỌN MÔ HÌNH ĐỂ CHẠY REALTIME INFERENCE:")
    print("1. Model A (EfficientNet-style, RGB Only)")
    print("2. Model B (CDCN + LBP, RGB + Texture)")
    print("3. Ensemble Mode A & B (Dung hợp Model A và B)")
    print("4. Model C (ResNet18-CBAM, Attention-based)")
    print("5. Model D (ResNet34-SE-CDC-MultiScale, Phiên bản Nâng Cấp)")
    print("6. Ensemble Mode A & C (Dung hợp Model A và C - ĐỘ CHÍNH XÁC CAO NHẤT)")
    print("="*50)
    choice = input("Lựa chọn (1, 2, 3, 4, 5 hoặc 6): ").strip()
    
    is_ensemble = False
    is_ensemble_a_c = False
    
    if choice == '3':
        is_ensemble = True
        model_name = "Ensemble_A_B"
        
        print("\nKhởi tạo cả hai mô hình A và B...")
        model_a = build_efficientnet_style(cfg.img_size, cfg.num_classes)
        model_b = build_cdcn_lbp(cfg.img_size, cfg.num_classes)
        
        ckpt_a = os.path.join(cfg.ckpt_dir, "model_a_effnet_style", "best_weights.weights.h5")
        ckpt_b = os.path.join(cfg.ckpt_dir, "model_b_cdcn_lbp", "best_weights.weights.h5")
        
        if not os.path.exists(ckpt_a) or not os.path.exists(ckpt_b):
            print("Lỗi: Không tìm thấy tệp trọng số checkpoints. Hãy chắc chắn đã chạy train_pipeline.py thành công!")
            return
            
        print("Tải trọng số Model A...")
        model_a.load_weights(ckpt_a)
        print("Tải trọng số Model B...")
        model_b.load_weights(ckpt_b)
        print("Khởi tạo và tải trọng số thành công cho chế độ Ensemble A & B!")
        
    elif choice == '6':
        is_ensemble_a_c = True
        model_name = "Ensemble_A_C"
        
        print("\nKhởi tạo cả hai mô hình A và C...")
        model_a = build_efficientnet_style(cfg.img_size, cfg.num_classes)
        model_c = build_resnet18_cbam(cfg.img_size, cfg.num_classes)
        
        ckpt_a = os.path.join(cfg.ckpt_dir, "model_a_effnet_style", "best_weights.weights.h5")
        ckpt_c = os.path.join(cfg.ckpt_dir, "model_c_resnet_cbam", "best_weights.weights.h5")
        
        if not os.path.exists(ckpt_a) or not os.path.exists(ckpt_c):
            print("Lỗi: Không tìm thấy tệp trọng số checkpoints. Hãy chắc chắn đã chạy train_pipeline.py thành công!")
            return
            
        print("Tải trọng số Model A...")
        model_a.load_weights(ckpt_a)
        print("Tải trọng số Model C...")
        model_c.load_weights(ckpt_c)
        print("Khởi tạo và tải trọng số thành công cho chế độ Ensemble A & C!")
        
    elif choice == '2':
        model_name = "model_b_cdcn_lbp"
        uses_lbp = True
        model = build_cdcn_lbp(cfg.img_size, cfg.num_classes)
        best_ckpt_path = os.path.join(cfg.ckpt_dir, model_name, "best_weights.weights.h5")
        
        print(f"\nĐang tải trọng số từ: {best_ckpt_path}...")
        if not os.path.exists(best_ckpt_path):
            print(f"Lỗi: Không tìm thấy tệp trọng số tại {best_ckpt_path}!")
            return
        model.load_weights(best_ckpt_path)
        print("Tải mô hình thành công!")
        
    elif choice == '4':
        model_name = "model_c_resnet_cbam"
        uses_lbp = False
        model = build_resnet18_cbam(cfg.img_size, cfg.num_classes)
        best_ckpt_path = os.path.join(cfg.ckpt_dir, model_name, "best_weights.weights.h5")
        
        print(f"\nĐang tải trọng số từ: {best_ckpt_path}...")
        if not os.path.exists(best_ckpt_path):
            print(f"Lỗi: Không tìm thấy tệp trọng số tại {best_ckpt_path}!")
            return
        model.load_weights(best_ckpt_path)
        print("Tải mô hình thành công!")
        
    elif choice == '5':
        model_name = "model_d_resnet_se"
        uses_lbp = False
        model = build_resnet34_se_cdc_multiscale(cfg.img_size, cfg.num_classes)
        best_ckpt_path = os.path.join(cfg.ckpt_dir, model_name, "best_weights.weights.h5")
        
        print(f"\nĐang tải trọng số từ: {best_ckpt_path}...")
        if not os.path.exists(best_ckpt_path):
            print(f"Lỗi: Không tìm thấy tệp trọng số tại {best_ckpt_path}!")
            return
        model.load_weights(best_ckpt_path)
        print("Tải mô hình thành công!")
        
    else:
        model_name = "model_a_effnet_style"
        uses_lbp = False
        model = build_efficientnet_style(cfg.img_size, cfg.num_classes)
        best_ckpt_path = os.path.join(cfg.ckpt_dir, model_name, "best_weights.weights.h5")
        
        print(f"\nĐang tải trọng số từ: {best_ckpt_path}...")
        if not os.path.exists(best_ckpt_path):
            print(f"Lỗi: Không tìm thấy tệp trọng số tại {best_ckpt_path}!")
            return
        model.load_weights(best_ckpt_path)
        print("Tải mô hình thành công!")

    # Mở camera (sử dụng camera mặc định số 0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Lỗi: Không thể mở camera. Hãy chắc chắn webcam của bạn đang hoạt động và không bị phần mềm khác chiếm dụng.")
        return

    print("\n" + "="*50)
    print("ĐÃ MỞ CAMERA THÀNH CÔNG")
    print(f"Chế độ chạy: {model_name}")
    print("- Đưa khuôn mặt vào khung hình để kiểm tra.")
    print("- Nhấn phím 'Q' trên cửa sổ camera để thoát demo.")
    print("="*50)
    
    # Load Haar cascade để khoanh vùng hiển thị trên camera
    haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(haar_path)

    window_name = f"Face Anti-Spoofing - Demo ({model_name})"
    cv2.namedWindow(window_name)

    # Sử dụng ngưỡng quyết định mặc định (0.50) theo cấu hình huấn luyện gốc của mô hình
    thresh_val = 0.50
    margin_val = 0.35  # Biên cắt khuôn mặt chuẩn lúc huấn luyện

    # Cấu hình bộ đệm làm mượt và bỏ qua khung hình (chống giật lag)
    prob_history = []
    max_history_len = 5      # Giảm xuống 5 để phản ứng nhanh hơn
    cached_predictions = []  # Lưu kết quả từ frame trước
    frame_idx = 0
    skip_frames = 5          # Giảm xuống 1 (chạy inference 1 trong 2 frame để giảm trễ)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Lỗi: Không đọc được luồng video từ camera.")
            break

        # Lật ảnh theo chiều ngang để có chế độ gương (mirror mode) tự nhiên
        frame = cv2.flip(frame, 1)

        # Nhân bản frame để hiển thị vẽ khung
        display_frame = frame.copy()
        h, w = frame.shape[:2]

        # Chỉ thực hiện detect khuôn mặt và chạy inference mỗi (skip_frames + 1) frame
        if frame_idx % (skip_frames + 1) == 0:
            cached_predictions = []
            
            # Phát hiện khuôn mặt
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

            if len(faces) == 0:
                prob_history.clear()  # Xóa lịch sử khi không thấy mặt để tránh trễ cho người tiếp theo

            for (x, y, fw, fh) in faces:
                # Cắt khuôn mặt theo margin chuẩn
                mx, my = int(fw * margin_val), int(fh * margin_val)
                x0, y0 = max(0, x - mx), max(0, y - my)
                x1, y1 = min(w, x + fw + mx), min(h, y + fh + my)
                face_crop = frame[y0:y1, x0:x1]

                if face_crop.size == 0:
                    continue

                # Resize & Chuẩn hóa ảnh
                face_resized = cv2.resize(face_crop, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_AREA)
                rgb_input = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

                # Dự đoán kết quả
                if is_ensemble:
                    # 1. Prediction từ Model A (RGB only)
                    inputs_a = {"rgb": np.expand_dims(rgb_input, axis=0)}
                    logits_a = model_a(inputs_a, training=False)
                    prob_a = tf.nn.softmax(logits_a, axis=-1).numpy()[0][1] # P(real) Model A
                    
                    # 2. Prediction từ Model B (RGB + LBP)
                    rgb_uint8 = (rgb_input * 255.0).astype(np.uint8)
                    lbp_input = lbp_numpy_wrapper(rgb_uint8, cfg.lbp_points, cfg.lbp_radius)
                    inputs_b = {
                        "rgb": np.expand_dims(rgb_input, axis=0),
                        "lbp": np.expand_dims(lbp_input, axis=0)
                    }
                    logits_b = model_b(inputs_b, training=False)
                    prob_b = tf.nn.softmax(logits_b, axis=-1).numpy()[0][1] # P(real) Model B
                    
                    # 3. Dung hợp xác suất có trọng số (60% Model A, 40% Model B)
                    current_prob = 0.60 * prob_a + 0.40 * prob_b
                elif is_ensemble_a_c:
                    # 1. Prediction từ Model A
                    inputs_a = {"rgb": np.expand_dims(rgb_input, axis=0)}
                    logits_a = model_a(inputs_a, training=False)
                    prob_a = tf.nn.softmax(logits_a, axis=-1).numpy()[0][1]
                    
                    # 2. Prediction từ Model C
                    inputs_c = {"rgb": np.expand_dims(rgb_input, axis=0)}
                    logits_c = model_c(inputs_c, training=False)
                    prob_c = tf.nn.softmax(logits_c, axis=-1).numpy()[0][1]
                    
                    # 3. Dung hợp xác suất có trọng số (40% Model A, 60% Model C)
                    current_prob = 0.40 * prob_a + 0.60 * prob_c
                else:
                    if uses_lbp:
                        rgb_uint8 = (rgb_input * 255.0).astype(np.uint8)
                        lbp_input = lbp_numpy_wrapper(rgb_uint8, cfg.lbp_points, cfg.lbp_radius)
                        inputs = {
                            "rgb": np.expand_dims(rgb_input, axis=0),
                            "lbp": np.expand_dims(lbp_input, axis=0)
                        }
                    else:
                        inputs = {
                            "rgb": np.expand_dims(rgb_input, axis=0)
                        }

                    logits = model(inputs, training=False)
                    probs = tf.nn.softmax(logits, axis=-1).numpy()[0]
                    current_prob = probs[1]  # Xác suất là ảnh thật (P(real))
                
                # Phản ứng tức thì nếu độ tin cậy cực kỳ cao (tránh trễ chuyển đổi trạng thái khi đưa ảnh spoof hoặc mặt thật vào)
                if current_prob < 0.20 or current_prob > 0.80:
                    prob_history = [current_prob]
                else:
                    prob_history.append(current_prob)
                    if len(prob_history) > max_history_len:
                        prob_history.pop(0)

                smoothed_real_prob = np.mean(prob_history)

                # Phân loại dựa trên xác suất đã làm mượt
                is_real = smoothed_real_prob >= thresh_val
                label = "REAL" if is_real else "SPOOF"
                color = (0, 255, 0) if is_real else (0, 0, 255)
                confidence = smoothed_real_prob if is_real else (1.0 - smoothed_real_prob)

                cached_predictions.append(((x, y, fw, fh), label, confidence, color))

        # Vẽ tất cả các khung hình và thông số (từ cache để đảm bảo FPS mượt)
        for (box, label, confidence, color) in cached_predictions:
            x, y, fw, fh = box
            cv2.rectangle(display_frame, (x, y), (x + fw, y + fh), color, 2)
            cv2.putText(display_frame, f"{label} ({confidence:.2%})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Hiển thị FPS và chế độ lên màn hình
        cv2.putText(display_frame, f"Mode: {model_name} | Latency Optimization: ON", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Hiển thị frame lên cửa sổ GUI
        cv2.imshow(window_name, display_frame)
        frame_idx += 1

        # Nhấn q để thoát loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
