import sys
import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import logging
from fastapi import File, UploadFile, HTTPException, Query, Request
from fastapi.responses import JSONResponse, HTMLResponse
from PIL import Image
import numpy as np
import faiss
import torch
import torchvision
from torchvision import transforms
import io
from main import app
from threading import Lock
from dotenv import load_dotenv
from mongo_connect import get_cad_file, list_all

# loading env file
load_dotenv("prod.env")

# Ensure the 'logs' directory exists
os.makedirs("logs", exist_ok=True)

# Disable PIL's decompression bomb protection
Image.MAX_IMAGE_PIXELS = 200000000

# Set up logging
# Define a formatter
formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s'
)

# Create the main logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Stream handler (stdout)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)

# File handler for all logs
file_handler = logging.FileHandler("logs/image-search.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(stream_handler)
logger.addHandler(file_handler)

# Configuration
BASE_URL = os.getenv("BASE_URL", "")
INDEX_PATH = os.getenv("INDEX_PATH", "faiss_index.bin")
NAMES_PATH = os.getenv("NAMES_PATH", "image_names.npy")
DATA_DIR = os.getenv("DATA_DIR", "./uploaded_images")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOP_K = 5  # Number of similar images to return
IMAGE_ROOT_PATH = os.getenv("IMAGE_ROOT_PATH", "./uploads")

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
image_names = np.load(NAMES_PATH, allow_pickle=True).tolist()
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

@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_content = """
    <!DOCTYPE html>
    <html>
        <head>
            <title>Backend Status</title>
            <style>
                body { font-family: Arial, sans-serif; text-align: center; padding-top: 50px; background-color: #f4f4f4; }
                h1 { color: #2c3e50; }
                a { text-decoration: none; color: white; background-color: #007BFF; padding: 10px 20px; border-radius: 5px; }
                a:hover { background-color: #0056b3; }
            </style>
        </head>
        <body>
            <h1>🚀 Everything is working!</h1>
            <p>Welcome to the Backend API</p>
            <a href="/docs">Go to API Docs</a>
        </body>
    </html>
    """
    return html_content

@app.post("/search")
async def search_similar_images(file: UploadFile = File(...), request : Request = None):
    """Search for similar images given an input image."""
    effective_url = str(request.base_url) if request else BASE_URL
    # print(f"{effective_url=}")
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        # print(f"{file.filename=}")
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
                "img_url":f"{effective_url}{image_name}",
                "cad_url": f"{effective_url}{cad_file_path}"
            }
            similar_images.append(image_info)
        logger.info(f"Searched for image file {file.filename}")
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
            image_names = np.load(NAMES_PATH, allow_pickle=True).tolist()
        logger.info(f"Reloaded FAISS index with {len(image_names)} images")
        return {"message": f"Index reloaded with {len(image_names)} images"}
    except Exception as e:
        logger.error(f"Error reloading index: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/list", response_model=dict)
async def list_images(
    page: int = Query(1, ge=1, description="Page number, starting from 1"),
    limit: int = Query(10, ge=1, le=100, description="Number of items per page, max 100"),
    request: Request = None
):
    """List all images in the folder with pagination"""
    try:
        # Use request.base_url if available, otherwise fall back to environment variable
        effective_base_url = str(request.base_url) if request else BASE_URL
        data = list_all(page=page, limit=limit, base_url=effective_base_url)
        logger.info(f"Retrieved page {page} with limit {limit}, total items: {data['total']}")
        return JSONResponse(content=data)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error in list_images endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
