import os
import cv2
import numpy as np
import argparse
from pathlib import Path

try:
    from mtcnn import MTCNN
except ImportError:
    print("[ERROR] Thư viện 'mtcnn' chưa được cài đặt. Vui lòng chạy: pip install mtcnn")
    exit(1)

def align_face(img, left_eye, right_eye):
    """
    Căn chỉnh khuôn mặt sao cho 2 mắt nằm trên một đường ngang.
    """
    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye
    
    dY = right_eye_y - left_eye_y
    dX = right_eye_x - left_eye_x
    angle = np.degrees(np.arctan2(dY, dX))
    
    eyes_center = (float((left_eye_x + right_eye_x) / 2.0), float((left_eye_y + right_eye_y) / 2.0))
    
    M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
    aligned_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)
    
    return aligned_img

def process_dataset(input_dir, output_dir, target_size=(112, 112)):
    detector = MTCNN()
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"[ERROR] Thư mục đầu vào không tồn tại: {input_dir}")
        return
        
    output_path.mkdir(parents=True, exist_ok=True)
    
    processed_count = 0
    failed_count = 0
    
    # Duyệt qua tất cả các class
    classes = sorted([d for d in input_path.iterdir() if d.is_dir()])
    for cls_dir in classes:
        out_cls_dir = output_path / cls_dir.name
        out_cls_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path in cls_dir.rglob("*"):
            if not img_path.is_file() or img_path.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
                continue
                
            out_img_path = out_cls_dir / img_path.name
            if out_img_path.exists():
                continue # Bỏ qua nếu đã xử lý
                
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = detector.detect_faces(rgb_img)
            
            if len(results) > 0:
                # Lấy khuôn mặt có confidence score cao nhất
                face = max(results, key=lambda b: b.get('confidence', 0))
                x, y, w, h = face['box']
                keypoints = face['keypoints']
                
                # Căn chỉnh mặt
                aligned_img = align_face(img, keypoints['left_eye'], keypoints['right_eye'])
                
                # Mở rộng Bounding Box để lấy thêm tóc và toàn bộ đầu
                margin_x = int(w * 0.3) # Mở rộng chiều rộng 30%
                margin_y_top = int(h * 0.5) # Mở rộng lên trên 50% để lấy trọn tóc
                margin_y_bottom = int(h * 0.2) # Mở rộng xuống dưới 20% cho cằm

                new_x = max(0, x - margin_x // 2)
                new_y = max(0, y - margin_y_top)
                new_w = min(aligned_img.shape[1] - new_x, w + margin_x)
                new_h = min(aligned_img.shape[0] - new_y, h + margin_y_top + margin_y_bottom)
                
                cropped = aligned_img[new_y:new_y+new_h, new_x:new_x+new_w]
                
                if cropped.size > 0:
                    resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(str(out_img_path), resized)
                    processed_count += 1
                else:
                    failed_count += 1
            else:
                failed_count += 1
                
    print(f"\n[DONE] Đã xử lý xong thư mục: {input_dir}")
    print(f"Thành công: {processed_count} ảnh")
    print(f"Thất bại (không nhận diện được mặt): {failed_count} ảnh")
    print(f"Lưu tại: {output_dir}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tiền xử lý ảnh: Detect, Align và Crop về 112x112")
    parser.add_argument("--input", type=str, default="dataset", help="Thư mục dataset gốc")
    parser.add_argument("--output", type=str, default="dataset_aligned", help="Thư mục đầu ra")
    args = parser.parse_args()
    
    train_in = os.path.join(args.input, "train")
    train_out = os.path.join(args.output, "train")
    if os.path.exists(train_in):
        process_dataset(train_in, train_out)
        
    val_in = os.path.join(args.input, "val")
    val_out = os.path.join(args.output, "val")
    if os.path.exists(val_in):
        process_dataset(val_in, val_out)

    test_in = os.path.join(args.input, "test")
    test_out = os.path.join(args.output, "test")
    if os.path.exists(test_in):
        process_dataset(test_in, test_out)
