import cv2
import numpy as np
import tensorflow as tf
from typing import List, Tuple, Dict, Any
from mtcnn import MTCNN
import os
import io
from PIL import Image

_DETECTOR = None
_MODEL = None
_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models", "face_recognition", "iresnet", "results_iresnet50", "best_iresnet50_backbone.h5"
    # "models", "face_recognition", "iresnet", "results_iresnet50",'backbones' , "backbone_epoch12.h5"
)

def get_detector():
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = MTCNN()
    return _DETECTOR

from using_pretrained_model.verifier import _build_backbone

def get_model():
    global _MODEL
    if _MODEL is None:
        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(f"Backbone model not found at {_MODEL_PATH}")
        _MODEL = _build_backbone(input_shape=(112, 112, 3), embedding_dim=512, dropout_rate=0.0)
        _MODEL.load_weights(_MODEL_PATH)
    return _MODEL

def warmup_models():
    """Run a dummy image through the models to build the computational graph."""
    detector = get_detector()
    model = get_model()
    
    print("🔄 Đang warmup MTCNN...")
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    detector.detect_faces(dummy_img)
    
    print("🔄 Đang warmup iResNet50...")
    dummy_face = np.zeros((1, 112, 112, 3), dtype=np.float32)
    model.predict(dummy_face, verbose=0)
    print("✅ Warmup hoàn tất!")

def align_and_crop(img: np.ndarray, face: Dict[str, Any], img_size: int = 112) -> np.ndarray:
    """Aligns and crops a single face based on MTCNN results."""
    x, y, w, h = face["box"]
    kp = face["keypoints"]

    dx = kp["right_eye"][0] - kp["left_eye"][0]
    dy = kp["right_eye"][1] - kp["left_eye"][1]
    angle = np.degrees(np.arctan2(dy, dx))
    eye_center = (
        (kp["left_eye"][0] + kp["right_eye"][0]) / 2.0,
        (kp["left_eye"][1] + kp["right_eye"][1]) / 2.0,
    )
    M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
    aligned = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)

    mx = int(w * 0.3)
    my_top = int(h * 0.5)
    my_bot = int(h * 0.2)
    nx, ny = max(0, x - mx // 2), max(0, y - my_top)
    nw = min(aligned.shape[1] - nx, w + mx)
    nh = min(aligned.shape[0] - ny, h + my_top + my_bot)

    cropped = aligned[ny:ny + nh, nx:nx + nw]
    if cropped.size == 0:
        return None
    face_img = cv2.resize(cropped, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
    
    # Normalize: [-1, 1]
    face_img = (face_img.astype(np.float32) - 127.5) / 128.0
    return face_img

def process_attendance_frame(img_bytes: bytes) -> Tuple[List[Dict[str, int]], np.ndarray]:
    """
    Detects all faces in an image bytes array.
    Returns:
        bboxes: List of bounding boxes [{"x": x, "y": y, "w": w, "h": h}, ...]
        faces_tensor: numpy array of shape (N, 112, 112, 3) ready for model inference
    """
    image = Image.open(io.BytesIO(img_bytes))
    image = image.convert("RGB")
    img_np = np.array(image)
    # MTCNN expects RGB image
    
    detector = get_detector()
    results = detector.detect_faces(img_np)
    
    # Filter out weak detections
    results = [res for res in results if res.get("confidence", 0) > 0.8]
    
    bboxes = []
    face_crops = []
    
    # OpenCV image for warping/cropping (usually BGR in cv2, but MTCNN gave coords for the original image. 
    # The align_and_crop logic expects the image we pass in to be cropped correctly.
    # We will pass the RGB image since the training pipeline normalized the RGB image)
    for face in results:
        cropped = align_and_crop(img_np, face)
        if cropped is not None:
            box = face["box"]
            # MTCNN can sometimes return negative coordinates
            x, y, w, h = max(0, box[0]), max(0, box[1]), box[2], box[3]
            bboxes.append({"x": x, "y": y, "w": w, "h": h})
            face_crops.append(cropped)
            
    if len(face_crops) == 0:
        return [], np.array([])
        
    faces_tensor = np.stack(face_crops, axis=0)
    return bboxes, faces_tensor

def extract_embeddings(faces_tensor: np.ndarray) -> np.ndarray:
    """Extract embeddings for a batch of faces. Returns (N, 512)"""
    if faces_tensor.size == 0:
        return np.array([])
    model = get_model()
    embeddings = model.predict(faces_tensor, verbose=0)
    # L2 Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-10)
    return embeddings
