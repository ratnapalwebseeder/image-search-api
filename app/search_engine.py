import numpy as np
from sqlalchemy.orm import Session
from app import models

def search_similar_images(db: Session, query_vec: np.ndarray, top_k: int = 5):
    vectors = []
    names = []
    
    for row in db.query(models.ImageVector).all():
        vec = np.frombuffer(row.vector, dtype=np.float32)
        vectors.append(vec)
        names.append((row.name, row.cad_name, row.image_data))  # Include cad_name
    
    vectors = np.vstack(vectors)
    similarities = np.dot(vectors, query_vec)
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        name, cad_name, data = names[idx]  # Unpack cad_name
        results.append({
            "name": name,
            "cad_name": cad_name,
            "similarity": float(similarities[idx]),
            "image_data": data
        })
    return results