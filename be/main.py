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
from be.services.face_service import process_attendance_frame, extract_embeddings, get_model, get_detector, warmup_models
from be.services.anchor_store import register_anchor, get_all_anchors, identify_face, delete_anchor, load_store, add_image_to_anchor, remove_image_from_anchor

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
async def process_attendance(image: UploadFile = File(...), method: str = Form("mean"), num_images: int = Form(3)):
    if method not in ["one_shot", "mean", "average_cosine"]:
        raise HTTPException(status_code=400, detail="Invalid method. Use 'one_shot', 'mean' or 'average_cosine'.")
        
    img_bytes = await image.read()
    
    bboxes, faces_tensor = process_attendance_frame(img_bytes)
    
    if len(bboxes) == 0:
        return {"results": []}
        
    # Extract embeddings for all detected faces in a single batch
    embeddings = extract_embeddings(faces_tensor)
    
    results = []
    for i, bbox in enumerate(bboxes):
        face_emb = embeddings[i]
        name, score = identify_face(face_emb, method=method, num_images=num_images)
        
        results.append({
            "bbox": bbox,
            "name": name,
            "score": score
        })
        
    # Post-processing for duplicate identities
    # 1. Find the instance index with the maximum score for each recognized identity
    best_instances = {}
    for i, res in enumerate(results):
        name = res["name"]
        if name != "Unknown":
            if name not in best_instances or res["score"] > results[best_instances[name]]["score"]:
                best_instances[name] = i
                
    # 2. Mark duplicates as Unknown with specific tag
    for i, res in enumerate(results):
        name = res["name"]
        if name != "Unknown" and i != best_instances.get(name):
            res["name"] = f"Unknown [trùng {name}]"
            
    return {"results": results}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
