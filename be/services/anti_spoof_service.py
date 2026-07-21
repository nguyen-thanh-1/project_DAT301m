import sys
import os
import cv2
import numpy as np
import tensorflow as tf
from typing import Tuple, Dict, Any

# Đảm bảo import được module từ project_DAT301m_handover
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HANDOVER_DIR = os.path.join(ROOT_DIR, "project_DAT301m_handover")
if HANDOVER_DIR not in sys.path:
    sys.path.append(HANDOVER_DIR)

from anti_spoofing.models import build_efficientnet_style, build_resnet18_cbam

_MODEL_A = None
_MODEL_C = None
_CKPT_DIR = os.path.join(HANDOVER_DIR, "checkpoints")

def get_anti_spoof_models():
    """Khởi tạo và tải trọng số cho Ensemble A & C nếu chưa có trong bộ nhớ."""
    global _MODEL_A, _MODEL_C
    if _MODEL_A is None or _MODEL_C is None:
        print("🛡️ Đang khởi tạo và tải mô hình Anti-Spoofing (Ensemble A & C)...")
        img_size = 224
        num_classes = 2
        
        _MODEL_A = build_efficientnet_style(img_size, num_classes)
        _MODEL_C = build_resnet18_cbam(img_size, num_classes)
        
        ckpt_a = os.path.join(_CKPT_DIR, "model_a_effnet_style", "best_weights.weights.h5")
        ckpt_c = os.path.join(_CKPT_DIR, "model_c_resnet_cbam", "best_weights.weights.h5")
        
        if not os.path.exists(ckpt_a) or not os.path.exists(ckpt_c):
            raise FileNotFoundError(f"Checkpoints không tồn tại tại:\n - {ckpt_a}\n - {ckpt_c}")
            
        _MODEL_A.load_weights(ckpt_a)
        _MODEL_C.load_weights(ckpt_c)
        print("✅ Tải mô hình Anti-Spoofing thành công!")
        
    return _MODEL_A, _MODEL_C

def warmup_anti_spoof_models():
    """Chạy dummy inference để dựng computational graph cho mô hình Anti-Spoofing."""
    model_a, model_c = get_anti_spoof_models()
    print("🔄 Đang warmup Anti-Spoofing models...")
    dummy_input = {"rgb": np.zeros((1, 224, 224, 3), dtype=np.float32)}
    model_a(dummy_input, training=False)
    model_c(dummy_input, training=False)
    print("✅ Warmup Anti-Spoofing hoàn tất!")

def check_face_spoof(img_rgb: np.ndarray, bbox: Dict[str, Any], thresh_val: float = 0.70) -> Tuple[bool, float, str]:
    """
    Kiểm tra tính xác thực (Thật / Giả mạo) cho một khuôn mặt trong ảnh RGB.
    
    Args:
        img_rgb: Ảnh gốc định dạng numpy array RGB (H, W, 3)
        bbox: Bounding box của mặt {"x": x, "y": y, "w": w, "h": h}
        thresh_val: Ngưỡng xác định mặt thật (mặc định 0.70)
        
    Returns:
        is_real (bool): True nếu là mặt thật, False nếu giả mạo.
        prob_real (float): Xác suất là mặt thật (0.0 -> 1.0).
        label (str): "REAL" hoặc "SPOOF".
    """
    model_a, model_c = get_anti_spoof_models()
    
    h_img, w_img = img_rgb.shape[:2]
    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    
    # Cắt khuôn mặt theo margin chuẩn (0.35) như khi huấn luyện
    margin_val = 0.35
    mx, my = int(w * margin_val), int(h * margin_val)
    x0, y0 = max(0, int(x) - mx), max(0, int(y) - my)
    x1, y1 = min(w_img, int(x + w) + mx), min(h_img, int(y + h) + my)
    
    face_crop = img_rgb[y0:y1, x0:x1]
    if face_crop.size == 0:
        return False, 0.0, "SPOOF"
        
    # Resize về 224x224 và chuẩn hóa [0, 1]
    face_resized = cv2.resize(face_crop, (224, 224), interpolation=cv2.INTER_AREA)
    rgb_input = face_resized.astype(np.float32) / 255.0
    
    # Dự đoán bằng Ensemble A (40%) + C (60%)
    inputs = {"rgb": np.expand_dims(rgb_input, axis=0)}
    
    logits_a = model_a(inputs, training=False)
    prob_a = tf.nn.softmax(logits_a, axis=-1).numpy()[0][1]
    
    logits_c = model_c(inputs, training=False)
    prob_c = tf.nn.softmax(logits_c, axis=-1).numpy()[0][1]
    
    prob_real = float(0.40 * prob_a + 0.60 * prob_c)
    
    is_real = prob_real >= thresh_val
    label = "REAL" if is_real else "SPOOF"
    
    return is_real, prob_real, label
