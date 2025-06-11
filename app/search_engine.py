# app/search_engine.py
import numpy as np
from sqlalchemy.orm import Session
from app import models

def search_similar_images(db: Session, query_vec: np.ndarray, top_k: int = 5):
    vectors = []
    names = []
    for row in db.query(models.ImageVector).all():
        vec = np.frombuffer(row.vector, dtype=np.float32)
        vectors.append(vec)
        names.append((row.name, row.image_data))
    
    vectors = np.vstack(vectors)
    similarities = np.dot(vectors, query_vec)
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        name, data = names[idx]
        results.append({"name": name, "similarity": float(similarities[idx]), "image_data": data})
    return results
