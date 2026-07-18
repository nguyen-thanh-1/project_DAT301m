import cv2
from pathlib import Path

_haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_cascade = cv2.CascadeClassifier(_haar_path)

def detect_and_crop_face(img_bgr, margin=0.35, target_size=None):
    """Classical Haar-cascade crop. Falls back to a centered square crop if no face found."""
    if _face_cascade.empty():
        raise RuntimeError("Khong load duoc Haar cascade - kiem tra OpenCV install.")
    
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    h, w = img_bgr.shape[:2]
    if len(faces) > 0:
        # lay face lon nhat (gia dinh la subject chinh)
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        mx, my = int(fw * margin), int(fh * margin)
        x0, y0 = max(0, x - mx), max(0, y - my)
        x1, y1 = min(w, x + fw + mx), min(h, y + fh + my)
        crop = img_bgr[y0:y1, x0:x1]
    else:
        # fallback: center square crop (khong dung "hoc" gi ca, thuan hinh hoc)
        side = min(h, w)
        cx, cy = w // 2, h // 2
        x0, y0 = max(0, cx - side // 2), max(0, cy - side // 2)
        crop = img_bgr[y0:y0 + side, x0:x0 + side]

    if crop.size == 0:
        crop = img_bgr  # safety net
    if target_size is not None:
        crop = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return crop


def build_crop_cache(manifest, cache_dir, target_size, use_face_crop=True):
    """Precompute crops once to disk so training doesn't redo Haar detection every epoch."""
    cache_dir = Path(cache_dir)
    cached_manifest = {}
    n_no_face = 0
    n_total = 0
    for split, classes in manifest.items():
        cached_manifest[split] = {}
        for cls, files in classes.items():
            out_dir = cache_dir / split / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            cached_files = []
            for f in files:
                out_path = out_dir / f.name
                if not out_path.exists():
                    img = cv2.imread(str(f))
                    if img is None:
                        continue
                    n_total += 1
                    if use_face_crop:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        faces = _face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
                        if len(faces) == 0:
                            n_no_face += 1
                        crop = detect_and_crop_face(img, target_size=target_size)
                    else:
                        crop = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(str(out_path), crop)
                cached_files.append(out_path)
            cached_manifest[split][cls] = cached_files
    if n_total > 0:
        print(f"Haar cascade: no face detected in {n_no_face}/{n_total} newly processed images "
              f"({100*n_no_face/max(1,n_total):.1f}%) -> fell back to center-crop for those.")
    return cached_manifest
