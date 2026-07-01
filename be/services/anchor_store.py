import json
import numpy as np
import os
from typing import List, Dict, Tuple, Optional
import uuid

STORE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "anchors.json")

def load_store() -> Dict:
    if os.path.exists(STORE_PATH):
        with open(STORE_PATH, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def save_store(data: Dict):
    with open(STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def register_anchor(user_id: str, name: str, embeddings: np.ndarray, image_urls: List[str] = None) -> str:
    """
    Register a user with their embeddings and image urls.
    embeddings: numpy array of shape (N, 512) where N <= 3.
    """
    store = load_store()
    
    # Calculate mean embedding
    mean_emb = np.mean(embeddings, axis=0)
    # L2 normalize the mean embedding
    mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-10)
    
    store[user_id] = {
        "name": name,
        "individual_embeddings": embeddings.tolist(),
        "mean_embedding": mean_emb.tolist(),
        "image_urls": image_urls or []
    }
    
    save_store(store)
    return user_id

def get_all_anchors() -> List[Dict]:
    store = load_store()
    return [{"id": k, "name": v["name"], "num_images": len(v["individual_embeddings"]), "image_urls": v.get("image_urls", [])} for k, v in store.items()]

def delete_anchor(user_id: str) -> bool:
    store = load_store()
    if user_id in store:
        del store[user_id]
        save_store(store)
        return True
    return False

def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    # Assuming embeddings are already L2 normalized
    return float(np.dot(emb1, emb2))

def identify_face(face_emb: np.ndarray, method: str = "mean", num_images: int = 3) -> Tuple[Optional[str], float]:
    """
    Match a single face embedding against the store.
    method: "one_shot", "mean" or "average_cosine"
    """
    store = load_store()
    if not store:
        return "Unknown", 0.0
        
    best_match_id = None
    best_score = -1.0
    highest_overall_score = -1.0
    
    for user_id, user_data in store.items():
        ind_embs = user_data["individual_embeddings"]
        
        # Limit to requested number of images
        ind_embs_to_use = ind_embs[:num_images] if num_images > 0 else ind_embs
        if method == "one_shot":
            ind_embs_to_use = ind_embs[:1]
            
        num_anchors = len(ind_embs_to_use)
        
        # Ngưỡng động (Dynamic threshold)
        if method == "one_shot" or num_anchors == 1:
            user_threshold = 0.57
        elif method == "mean":
            user_threshold = 0.50
        else:
            user_threshold = 0.50
        
        if method == "one_shot":
            anchor_emb = np.array(ind_embs_to_use[0])
            score = cosine_similarity(face_emb, anchor_emb)
        elif method == "mean":
            mean_emb = np.mean(np.array(ind_embs_to_use), axis=0)
            mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-10)
            score = cosine_similarity(face_emb, mean_emb)
        else: # average_cosine
            scores = [cosine_similarity(face_emb, np.array(a_emb)) for a_emb in ind_embs_to_use]
            score = sum(scores) / len(scores) if scores else 0.0
            
        if score > highest_overall_score:
            highest_overall_score = score
            
        # Chỉ lấy những ai vượt qua ngưỡng của riêng họ
        if score >= user_threshold and score > best_score:
            best_score = score
            best_match_id = user_id
            
    if best_match_id:
        return store[best_match_id]["name"], best_score
        
    return "Unknown", highest_overall_score

def add_image_to_anchor(user_id: str, embedding: np.ndarray, image_url: str) -> bool:
    store = load_store()
    if user_id not in store:
        return False
        
    user_data = store[user_id]
    
    # Append embedding
    ind_embs = user_data["individual_embeddings"]
    ind_embs.append(embedding.tolist())
    
    # Append URL
    urls = user_data.get("image_urls", [])
    urls.append(image_url)
    user_data["image_urls"] = urls
    
    # Recalculate mean
    mean_emb = np.mean(np.array(ind_embs), axis=0)
    mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-10)
    user_data["mean_embedding"] = mean_emb.tolist()
    
    save_store(store)
    return True

def remove_image_from_anchor(user_id: str, index: int) -> bool:
    store = load_store()
    if user_id not in store:
        return False
        
    user_data = store[user_id]
    ind_embs = user_data["individual_embeddings"]
    urls = user_data.get("image_urls", [])
    
    if index < 0 or index >= len(ind_embs):
        return False
        
    if len(ind_embs) <= 1:
        # Rule: Must keep at least 1 image
        return False
        
    ind_embs.pop(index)
    if index < len(urls):
        urls.pop(index)
        
    user_data["image_urls"] = urls
    
    # Recalculate mean
    mean_emb = np.mean(np.array(ind_embs), axis=0)
    mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-10)
    user_data["mean_embedding"] = mean_emb.tolist()
    
    save_store(store)
    return True
