"""
Face preprocessing for inference: MTCNN detect → eye align → crop → resize 112.
Reuses the same pipeline as data_processing/preprocess.py.
"""

import cv2
import numpy as np

_DETECTOR = None


def _get_detector():
    global _DETECTOR
    if _DETECTOR is None:
        from mtcnn import MTCNN
        _DETECTOR = MTCNN()
    return _DETECTOR


def preprocess_face(img_path, img_size=112):
    """Detect, align, crop, resize a face image → normalized tensor [1,112,112,3].
    Falls back to center-crop if no face detected.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read: {img_path}")

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detector = _get_detector()
    results = detector.detect_faces(rgb)

    if results:
        face = max(results, key=lambda b: b.get("confidence", 0))
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
        face_img = cv2.resize(cropped, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
    else:
        print(f"[WARN] No face detected in {img_path}, using center-crop")
        h, w = img.shape[:2]
        s = min(h, w)
        cy, cx = h // 2, w // 2
        cropped = img[cy - s // 2:cy + s // 2, cx - s // 2:cx + s // 2]
        face_img = cv2.resize(cropped, (img_size, img_size), interpolation=cv2.INTER_CUBIC)

    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = (face_img.astype(np.float32) - 127.5) / 128.0
    return np.expand_dims(face_img, axis=0)
