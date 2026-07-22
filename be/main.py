from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List
import uvicorn
import numpy as np
import os
import uuid
import shutil
from contextlib import asynccontextmanager

from be.services.face_service import process_attendance_frame, extract_embeddings, get_model, get_detector, warmup_models
from be.services.anchor_store import register_anchor, get_all_anchors, identify_face, delete_anchor, load_store, add_image_to_anchor, remove_image_from_anchor, identify_face_dual
from be.services.anti_spoof_service import check_face_spoof

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Dang tai cac mo hinh (MTCNN va iResNet50) vao bo nho...")
    get_detector()
    get_model()
    warmup_models()
    print("Da tai mo hinh thanh cong!")
    yield
    print("Tat server...")

app = FastAPI(title="Face Recognition Attendance API", lifespan=lifespan)

# Allow CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For production, restrict to frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup static files directory for anchor images
ANCHOR_IMAGES_DIR = os.path.join(os.path.dirname(__file__), "anchor_images")
os.makedirs(ANCHOR_IMAGES_DIR, exist_ok=True)
app.mount("/static/anchors", StaticFiles(directory=ANCHOR_IMAGES_DIR), name="anchors")

@app.post("/api/anchors/upload")
async def upload_anchor(name: str = Form(...), images: List[UploadFile] = File(...)):
    if not images or len(images) > 3:
        raise HTTPException(status_code=400, detail="Must provide between 1 and 3 images.")
        
    # Create a safe folder name from the student's name
    safe_name = name.strip()
    for ch in ['<', '>', ':', '"', '/', '\\', '|', '?', '*']:
        safe_name = safe_name.replace(ch, '')
        
    folder_name = safe_name
    user_dir = os.path.join(ANCHOR_IMAGES_DIR, folder_name)
    
    # Handle duplicate names
    counter = 1
    while os.path.exists(user_dir):
        folder_name = f"{safe_name}_{counter}"
        user_dir = os.path.join(ANCHOR_IMAGES_DIR, folder_name)
        counter += 1
        
    os.makedirs(user_dir, exist_ok=True)
    user_id = str(uuid.uuid4())
        
    all_face_crops = []
    image_urls = []
    
    for i, image in enumerate(images):
        img_bytes = await image.read()
        
        # Save image to disk
        filename = f"{i+1}_{image.filename}"
        file_path = os.path.join(user_dir, filename)
        with open(file_path, "wb") as f:
            f.write(img_bytes)
            
        image_urls.append(f"/static/anchors/{folder_name}/{filename}")
        
        bboxes, face_tensor = process_attendance_frame(img_bytes)
        
        if len(bboxes) == 0:
            raise HTTPException(status_code=400, detail=f"No face detected in one of the images.")
        if len(bboxes) > 1:
            raise HTTPException(status_code=400, detail=f"Multiple faces detected in one of the anchor images. Please use an image with only {name}.")
            
        all_face_crops.append(face_tensor[0])
        
    # Stack all crops into a batch
    batch_tensor = np.stack(all_face_crops, axis=0)
    
    # Extract embeddings
    embeddings = extract_embeddings(batch_tensor)
    
    # Register anchor
    register_anchor(user_id, name, embeddings, image_urls)
    
    return {"status": "success", "user_id": user_id, "name": name, "num_images": len(images), "image_urls": image_urls}

@app.get("/api/anchors")
async def list_anchors():
    return {"anchors": get_all_anchors()}

@app.delete("/api/anchors/{user_id}")
async def remove_anchor(user_id: str):
    store = load_store()
    user_data = store.get(user_id)
    if not user_data:
        raise HTTPException(status_code=404, detail="Anchor not found")
        
    image_urls = user_data.get("image_urls", [])
    if image_urls:
        # Extract folder name from the first image url e.g., /static/anchors/nguyen_van_a/1.jpg
        parts = image_urls[0].split("/")
        if len(parts) >= 4:
            folder_name = parts[3]
            user_dir = os.path.join(ANCHOR_IMAGES_DIR, folder_name)
            if os.path.exists(user_dir):
                shutil.rmtree(user_dir)
                
    success = delete_anchor(user_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete anchor from store")
        
    return {"status": "success", "message": "Deleted"}

@app.post("/api/anchors/{user_id}/images")
async def add_anchor_image(user_id: str, image: UploadFile = File(...)):
    store = load_store()
    if user_id not in store:
        raise HTTPException(status_code=404, detail="Anchor not found")
        
    user_data = store[user_id]
    if len(user_data["individual_embeddings"]) >= 3:
        raise HTTPException(status_code=400, detail="Tối đa 3 ảnh cho mỗi người.")
        
    img_bytes = await image.read()
    bboxes, face_tensor = process_attendance_frame(img_bytes)
    
    if len(bboxes) == 0:
        raise HTTPException(status_code=400, detail="Không tìm thấy khuôn mặt trong ảnh.")
    if len(bboxes) > 1:
        raise HTTPException(status_code=400, detail="Phát hiện nhiều khuôn mặt. Vui lòng chọn ảnh chỉ có 1 người.")
        
    embeddings = extract_embeddings(face_tensor)
    
    # We need to find the correct folder
    folder_name = user_data["name"].strip()
    for ch in ['<', '>', ':', '"', '/', '\\', '|', '?', '*']:
        folder_name = folder_name.replace(ch, '')
        
    image_urls = user_data.get("image_urls", [])
    if image_urls:
        parts = image_urls[0].split("/")
        if len(parts) >= 4:
            folder_name = parts[3]
    
    user_dir = os.path.join(ANCHOR_IMAGES_DIR, folder_name)
    os.makedirs(user_dir, exist_ok=True)
    
    filename = f"{uuid.uuid4().hex[:8]}_{image.filename}"
    file_path = os.path.join(user_dir, filename)
    with open(file_path, "wb") as f:
        f.write(img_bytes)
        
    image_url = f"/static/anchors/{folder_name}/{filename}"
    
    success = add_image_to_anchor(user_id, embeddings[0], image_url)
    if not success:
        raise HTTPException(status_code=500, detail="Lỗi khi thêm ảnh vào dữ liệu.")
        
    return {"status": "success"}

@app.delete("/api/anchors/{user_id}/images/{index}")
async def delete_anchor_image(user_id: str, index: int):
    store = load_store()
    if user_id not in store:
        raise HTTPException(status_code=404, detail="Anchor not found")
        
    user_data = store[user_id]
    if len(user_data["individual_embeddings"]) <= 1:
        raise HTTPException(status_code=400, detail="Phải giữ lại ít nhất 1 ảnh. Nếu muốn xóa toàn bộ, hãy xóa người dùng này ở danh sách bên ngoài.")
        
    image_urls = user_data.get("image_urls", [])
    if 0 <= index < len(image_urls):
        url = image_urls[index]
        parts = url.split("/")
        if len(parts) >= 5:
            folder_name = parts[3]
            filename = parts[4]
            file_path = os.path.join(ANCHOR_IMAGES_DIR, folder_name, filename)
            if os.path.exists(file_path):
                os.remove(file_path)
                
    success = remove_image_from_anchor(user_id, index)
    if not success:
        raise HTTPException(status_code=500, detail="Lỗi khi xóa ảnh.")
        
    return {"status": "success"}

@app.post("/api/attendance")
async def process_attendance(
    image: UploadFile = File(...), 
    method: str = Form("mean"), 
    num_images: int = Form(3),
    check_spoof: bool = Form(False)
):
    if method not in ["one_shot", "mean", "average_cosine"]:
        raise HTTPException(status_code=400, detail="Invalid method. Use 'one_shot', 'mean' or 'average_cosine'.")
        
    img_bytes = await image.read()
    
    if check_spoof:
        bboxes, faces_tensor, img_np = process_attendance_frame(img_bytes, return_rgb_image=True)
    else:
        bboxes, faces_tensor = process_attendance_frame(img_bytes, return_rgb_image=False)
    
    if len(bboxes) == 0:
        return {"results": []}
        
    results = []
    valid_indices = []
    
    if check_spoof:
        for i, bbox in enumerate(bboxes):
            is_real, prob_real, _ = check_face_spoof(img_np, bbox)
            if not is_real:
                results.append({
                    "bbox": bbox,
                    "name": "Cảnh báo: Giả Mạo (Spoof)",
                    "score": 0.0,
                    "is_real": False,
                    "spoof_prob": prob_real
                })
            else:
                valid_indices.append(i)
                results.append({
                    "bbox": bbox,
                    "name": "Unknown",
                    "score": 0.0,
                    "is_real": True,
                    "spoof_prob": prob_real
                })
    else:
        valid_indices = list(range(len(bboxes)))
        for bbox in bboxes:
            results.append({
                "bbox": bbox,
                "name": "Unknown",
                "score": 0.0
            })
            
    if valid_indices:
        valid_faces_tensor = faces_tensor[valid_indices]
        embeddings = extract_embeddings(valid_faces_tensor)
        
        for idx_in_valid, orig_idx in enumerate(valid_indices):
            face_emb = embeddings[idx_in_valid]
            name, score = identify_face(face_emb, method=method, num_images=num_images)
            results[orig_idx]["name"] = name
            results[orig_idx]["score"] = score
        
    # Post-processing for duplicate identities
    best_instances = {}
    for i, res in enumerate(results):
        name = res["name"]
        if name not in ["Unknown", "Cảnh báo: Giả Mạo (Spoof)"] and not name.startswith("Cảnh báo:"):
            if name not in best_instances or res["score"] > results[best_instances[name]]["score"]:
                best_instances[name] = i
                
    for i, res in enumerate(results):
        name = res["name"]
        if name not in ["Unknown", "Cảnh báo: Giả Mạo (Spoof)"] and not name.startswith("Cảnh báo:") and i != best_instances.get(name):
            res["name"] = f"Unknown [trùng {name}]"
            
    return {"results": results}

@app.post("/api/attendance/dual")
async def process_attendance_dual(
    image1: UploadFile = File(...), 
    image2: UploadFile = File(...), 
    method: str = Form("mean"), 
    num_images: int = Form(3),
    threshold: float = Form(0.5),
    check_spoof: bool = Form(False)
):
    if method not in ["one_shot", "mean", "average_cosine"]:
        raise HTTPException(status_code=400, detail="Invalid method. Use 'one_shot', 'mean' or 'average_cosine'.")
        
    img1_bytes = await image1.read()
    img2_bytes = await image2.read()
    
    if check_spoof:
        bboxes1, faces_tensor1, img_np1 = process_attendance_frame(img1_bytes, return_rgb_image=True)
        bboxes2, faces_tensor2, img_np2 = process_attendance_frame(img2_bytes, return_rgb_image=True)
    else:
        bboxes1, faces_tensor1 = process_attendance_frame(img1_bytes, return_rgb_image=False)
        bboxes2, faces_tensor2 = process_attendance_frame(img2_bytes, return_rgb_image=False)
    
    results_cam1 = []
    results_cam2 = []
    combined_results = []
    
    spoof_info1 = []
    spoof_info2 = []
    
    if check_spoof:
        for bbox in bboxes1:
            is_real, prob_real, _ = check_face_spoof(img_np1, bbox)
            spoof_info1.append((is_real, prob_real))
        for bbox in bboxes2:
            is_real, prob_real, _ = check_face_spoof(img_np2, bbox)
            spoof_info2.append((is_real, prob_real))
    else:
        spoof_info1 = [(True, 1.0)] * len(bboxes1)
        spoof_info2 = [(True, 1.0)] * len(bboxes2)
        
    # Trường hợp cả 2 camera đều phát hiện được khuôn mặt
    if len(bboxes1) > 0 and len(bboxes2) > 0:
        is_real1, prob1 = spoof_info1[0]
        is_real2, prob2 = spoof_info2[0]
        
        # Nếu khuôn mặt chính (index 0) bị giả mạo ở 1 trong 2 camera -> Chặn nhận diện
        if not is_real1 or not is_real2:
            res1 = {
                "bbox": bboxes1[0],
                "name": "Cảnh báo: Giả Mạo (Spoof)" if not is_real1 else "Unknown",
                "score": 0.0,
                "is_real": is_real1,
                "spoof_prob": prob1,
                "cam": "Cam 1 (Trái)"
            }
            res2 = {
                "bbox": bboxes2[0],
                "name": "Cảnh báo: Giả Mạo (Spoof)" if not is_real2 else "Unknown",
                "score": 0.0,
                "is_real": is_real2,
                "spoof_prob": prob2,
                "cam": "Cam 2 (Phải)"
            }
            results_cam1.append(res1)
            results_cam2.append(res2)
            combined_results.append({
                "name": "Cảnh báo: Giả Mạo (Spoof)",
                "score": 0.0,
                "score_cam1": 0.0,
                "score_cam2": 0.0,
                "is_real": False,
                "cam": "TB 2 Cam (Bị Chặn)"
            })
        else:
            embeddings1 = extract_embeddings(faces_tensor1)
            embeddings2 = extract_embeddings(faces_tensor2)
            
            name, avg_score, s1, s2 = identify_face_dual(embeddings1[0], embeddings2[0], method=method, num_images=num_images, threshold=threshold)
            
            res1 = {
                "bbox": bboxes1[0],
                "name": name if name != "Unknown" else "Unknown",
                "score": s1,
                "avg_score": avg_score,
                "is_real": True,
                "spoof_prob": prob1,
                "cam": "Cam 1 (Trái)"
            }
            res2 = {
                "bbox": bboxes2[0],
                "name": name if name != "Unknown" else "Unknown",
                "score": s2,
                "avg_score": avg_score,
                "is_real": True,
                "spoof_prob": prob2,
                "cam": "Cam 2 (Phải)"
            }
            results_cam1.append(res1)
            results_cam2.append(res2)
            combined_results.append({
                "name": name,
                "score": avg_score,
                "score_cam1": s1,
                "score_cam2": s2,
                "is_real": True,
                "cam": "TB 2 Cam"
            })
            
        # Xử lý các khuôn mặt phụ khác (nếu có)
        embeddings1_extracted = embeddings1 if (is_real1 and is_real2) else extract_embeddings(faces_tensor1)
        for i in range(1, len(bboxes1)):
            is_r, p_r = spoof_info1[i]
            if not is_r:
                results_cam1.append({"bbox": bboxes1[i], "name": "Cảnh báo: Giả Mạo (Spoof)", "score": 0.0, "is_real": False, "spoof_prob": p_r, "cam": "Cam 1 (Trái)"})
            else:
                emb = embeddings1_extracted[i]
                n, s = identify_face(emb, method=method, num_images=num_images)
                results_cam1.append({"bbox": bboxes1[i], "name": n, "score": s, "is_real": True, "spoof_prob": p_r, "cam": "Cam 1 (Trái)"})
                
        embeddings2_extracted = embeddings2 if (is_real1 and is_real2) else extract_embeddings(faces_tensor2)
        for i in range(1, len(bboxes2)):
            is_r, p_r = spoof_info2[i]
            if not is_r:
                results_cam2.append({"bbox": bboxes2[i], "name": "Cảnh báo: Giả Mạo (Spoof)", "score": 0.0, "is_real": False, "spoof_prob": p_r, "cam": "Cam 2 (Phải)"})
            else:
                emb = embeddings2_extracted[i]
                n, s = identify_face(emb, method=method, num_images=num_images)
                results_cam2.append({"bbox": bboxes2[i], "name": n, "score": s, "is_real": True, "spoof_prob": p_r, "cam": "Cam 2 (Phải)"})
                
    elif len(bboxes1) > 0: # Chỉ Cam 1 thấy mặt
        embeddings1 = extract_embeddings(faces_tensor1)
        for i, bbox in enumerate(bboxes1):
            is_r, p_r = spoof_info1[i]
            if not is_r:
                res = {"bbox": bbox, "name": "Cảnh báo: Giả Mạo (Spoof)", "score": 0.0, "is_real": False, "spoof_prob": p_r, "cam": "Cam 1 (Trái)"}
                results_cam1.append(res)
                combined_results.append({"name": "Cảnh báo: Giả Mạo (Spoof)", "score": 0.0, "is_real": False, "cam": "Chỉ Cam 1 (Bị Chặn)"})
            else:
                n, s = identify_face(embeddings1[i], method=method, num_images=num_images)
                res = {"bbox": bbox, "name": n, "score": s, "is_real": True, "spoof_prob": p_r, "cam": "Cam 1 (Trái)"}
                results_cam1.append(res)
                combined_results.append({"name": n, "score": s, "is_real": True, "cam": "Chỉ Cam 1"})
            
    elif len(bboxes2) > 0: # Chỉ Cam 2 thấy mặt
        embeddings2 = extract_embeddings(faces_tensor2)
        for i, bbox in enumerate(bboxes2):
            is_r, p_r = spoof_info2[i]
            if not is_r:
                res = {"bbox": bbox, "name": "Cảnh báo: Giả Mạo (Spoof)", "score": 0.0, "is_real": False, "spoof_prob": p_r, "cam": "Cam 2 (Phải)"}
                results_cam2.append(res)
                combined_results.append({"name": "Cảnh báo: Giả Mạo (Spoof)", "score": 0.0, "is_real": False, "cam": "Chỉ Cam 2 (Bị Chặn)"})
            else:
                n, s = identify_face(embeddings2[i], method=method, num_images=num_images)
                res = {"bbox": bbox, "name": n, "score": s, "is_real": True, "spoof_prob": p_r, "cam": "Cam 2 (Phải)"}
                results_cam2.append(res)
                combined_results.append({"name": n, "score": s, "is_real": True, "cam": "Chỉ Cam 2"})
            
    return {
        "results_cam1": results_cam1,
        "results_cam2": results_cam2,
        "results": combined_results
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
