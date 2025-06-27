import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import numpy as np
import faiss
import torch
import torchvision
from torchvision import transforms
import io
from main import app
from threading import Lock
from mongo_connect import get_cad_file
# Disable PIL's decompression bomb protection
Image.MAX_IMAGE_PIXELS = None

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
INDEX_PATH = "faiss_index.bin"
NAMES_PATH = "image_names.npy"
DATA_DIR = "./uploaded_images"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOP_K = 5  # Number of similar images to return
IMAGE_ROOT_PATH = "/uploads"

# Thread lock for FAISS index access
index_lock = Lock()

# Load ResNet18 model
model = torchvision.models.resnet18(weights="DEFAULT").to(DEVICE)
model.eval()

# Transform for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Hook to capture avgpool layer output
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model.avgpool.register_forward_hook(get_activation("avgpool"))

# Global variables for FAISS index and image names
index = faiss.read_index(INDEX_PATH)
image_names = np.load(NAMES_PATH).tolist()
logger.info(f"Loaded FAISS index with {len(image_names)} images")

def extract_embedding(image: Image.Image) -> np.ndarray:
    """Extract embedding from an input image."""
    try:
        image = image.convert("RGB")
        image = transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            _ = model(image)
        vec = activation["avgpool"].cpu().numpy().squeeze()
        vec = vec / np.linalg.norm(vec, keepdims=True)
        return vec.astype(np.float32)
    except Exception as e:
        logger.error(f"Error extracting embedding: {e}")
        raise HTTPException(status_code=500, detail="Error processing image")

@app.post("/search")
async def search_similar_images(file: UploadFile = File(...)):
    """Search for similar images given an input image."""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        query_vec = extract_embedding(image)
        with index_lock:
            distances, indices = index.search(query_vec.reshape(1, -1), TOP_K)
        similar_images = []

        for idx, dist in zip(indices[0], distances[0]):
            similarity_score = 1 - (dist ** 2) / 2
            image_name = image_names[idx]
            cad_file_path = get_cad_file(f"{IMAGE_ROOT_PATH}/{image_name}")
            image_info = {
                "name": image_name,
                "similarity": float(similarity_score),
                "img_url":f"{IMAGE_ROOT_PATH}/{image_name}",
                "cad_url": cad_file_path
            }
            similar_images.append(image_info)

        return {"similar_images": similar_images}
    except Exception as e:
        logger.error(f"Error in search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reload_index")
async def reload_index():
    """Reload the FAISS index and image names from disk."""
    try:
        with index_lock:
            global index, image_names
            index = faiss.read_index(INDEX_PATH)
            image_names = np.load(NAMES_PATH).tolist()
        logger.info(f"Reloaded FAISS index with {len(image_names)} images")
        return {"message": f"Index reloaded with {len(image_names)} images"}
    except Exception as e:
        logger.error(f"Error reloading index: {e}")
        raise HTTPException(status_code=500, detail=str(e))
