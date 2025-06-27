import os
import logging
import signal
import sys
import time
import numpy as np
import faiss
import torch
import torchvision
from torchvision import transforms
from PIL import Image
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from threading import Lock
from retry import retry
from pathlib import Path
import json
from dotenv import load_dotenv

load_dotenv("prod.env")

# Set up logging with structured format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s - [PID:%(process)d Thread:%(threadName)s]',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("faiss_indexer.log")
    ]
)
logger = logging.getLogger(__name__)

# Disable PIL's decompression bomb protection
Image.MAX_IMAGE_PIXELS = None

# Configuration
CONFIG = {
    "INDEX_PATH": "faiss_index.bin",
    "NAMES_PATH": "image_names.npy",
    "DATA_DIR": "../uploaded_images",
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "EMBEDDING_DIM": 512,  # ResNet18 avgpool output size
    "RETRY_ATTEMPTS": 3,
    "RETRY_DELAY": 1,
    "WATCHDOG_POLL_INTERVAL": 1.0,
    "VALID_EXTENSIONS": (".png", ".jpg", ".jpeg")
}

# Ensure data directory exists
Path(CONFIG["DATA_DIR"]).mkdir(parents=True, exist_ok=True)

# Thread lock for FAISS index updates
index_lock = Lock()

# Load ResNet18 model
model = torchvision.models.resnet18(weights="DEFAULT").to(CONFIG["DEVICE"])
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

# Initialize or load FAISS index and image names
def initialize_index():
    """Initialize or load FAISS index and image names, and index existing images."""
    global index, image_names
    index_path = Path(CONFIG["INDEX_PATH"])
    names_path = Path(CONFIG["NAMES_PATH"])
    
    if index_path.exists() and names_path.exists():
        try:
            index = faiss.read_index(str(index_path))
            image_names = np.load(str(names_path), allow_pickle=True).tolist()
            logger.info(f"Loaded FAISS index with {len(image_names)} images")
        except Exception as e:
            logger.error(f"Error loading index or names: {e}. Initializing new index.")
            index = faiss.IndexFlatL2(CONFIG["EMBEDDING_DIM"])
            image_names = []
    else:
        logger.info("Index or names file not found. Initializing new FAISS index.")
        index = faiss.IndexFlatL2(CONFIG["EMBEDDING_DIM"])
        image_names = []
    
    # Index existing images in DATA_DIR
    index_existing_images()
    save_index_and_names()

def index_existing_images():
    """Scan DATA_DIR and index existing images."""
    global index, image_names
    data_dir = Path(CONFIG["DATA_DIR"])
    logger.info(f"Scanning {data_dir} for existing images...")
    
    for file_path in data_dir.iterdir():
        if file_path.suffix.lower() not in CONFIG["VALID_EXTENSIONS"]:
            continue
        filename = file_path.name
        with index_lock:
            if filename in image_names:
                logger.info(f"Skipping already indexed file: {filename}")
                continue
        try:
            with Image.open(file_path) as image:
                image.verify()  # Verify image integrity
            with Image.open(file_path) as image:  # Reopen after verify
                vec = extract_embedding(image)
                if vec is None:
                    logger.warning(f"Skipping {filename} due to embedding extraction failure")
                    continue
                with index_lock:
                    index.add(vec.reshape(1, -1))
                    image_names.append(filename)
                logger.info(f"Indexed existing image: {filename}")
        except Exception as e:
            logger.error(f"Error processing existing image {filename}: {e}")

@retry(tries=CONFIG["RETRY_ATTEMPTS"], delay=CONFIG["RETRY_DELAY"], logger=logger)
def save_index_and_names():
    """Save FAISS index and image names with retries."""
    with index_lock:
        faiss.write_index(index, CONFIG["INDEX_PATH"])
        np.save(CONFIG["NAMES_PATH"], np.array(image_names, dtype=object))
    logger.info("Successfully saved FAISS index and image names")

def extract_embedding(image: Image.Image) -> np.ndarray:
    """Extract embedding from an input image."""
    try:
        image = image.convert("RGB")
        image = transform(image).unsqueeze(0).to(CONFIG["DEVICE"])
        with torch.no_grad():
            _ = model(image)
        vec = activation["avgpool"].cpu().numpy().squeeze()
        vec = vec / np.linalg.norm(vec, keepdims=True)
        return vec.astype(np.float32)
    except Exception as e:
        logger.error(f"Error extracting embedding: {e}")
        return None

class ImageFolderHandler(FileSystemEventHandler):
    def on_created(self, event):
        """Handle new file creation events."""
        global index, image_names
        if event.is_directory:
            return
        if not event.src_path.lower().endswith(CONFIG["VALID_EXTENSIONS"]):
            return
        
        logger.info(f"Detected new file: {event.src_path}")
        time.sleep(CONFIG["WATCHDOG_POLL_INTERVAL"])  # Ensure file is fully written
        filename = os.path.basename(event.src_path)
        
        with index_lock:
            if filename in image_names:
                logger.info(f"Skipping already indexed file: {filename}")
                return
        
        try:
            with Image.open(event.src_path) as image:
                image.verify()  # Verify image integrity
            with Image.open(event.src_path) as image:  # Reopen after verify
                vec = extract_embedding(image)
                if vec is None:
                    return
                with index_lock:
                    index.add(vec.reshape(1, -1))
                    image_names.append(filename)
                    save_index_and_names()
                logger.info(f"Added new image: {filename}")
        except Exception as e:
            logger.error(f"Error processing {event.src_path}: {e}")

def signal_handler(sig, frame):
 """Handle shutdown signals."""
 logger.info("Received shutdown signal. Stopping observer...")
 observer.stop()
 observer.join()
 logger.info("Observer stopped. Exiting.")
 sys.exit(0)

if __name__ == '__main__':
    # Initialize index and image names, including existing images
    initialize_index()
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start file system watcher
    observer = Observer()
    observer.schedule(ImageFolderHandler(), CONFIG["DATA_DIR"], recursive=False)
    observer.start()
    logger.info(f"Started watching folder: {CONFIG['DATA_DIR']}")
    
    try:
        while True:
            time.sleep(CONFIG["WATCHDOG_POLL_INTERVAL"])
    except Exception as e:
        logger.error(f"Unexpected error in main loop: {e}")
        observer.stop()
        observer.join()
        logger.info("Observer stopped due to error")
        sys.exit(1)