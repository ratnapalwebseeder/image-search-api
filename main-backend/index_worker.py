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
Image.MAX_IMAGE_PIXELS = 200000000
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from threading import Lock
from retry import retry
from pathlib import Path
from dotenv import load_dotenv

os.makedirs("worker_logs", exist_ok=True)
os.makedirs("vector_db", exist_ok=True)
# Define a formatter
formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s - [PID:%(process)d Thread:%(threadName)s]'
)

# Create the main logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Stream handler (stdout)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)

# File handler for all logs
file_handler = logging.FileHandler("worker_logs/faiss_indexer.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# File handler for error logs only
error_file_handler = logging.FileHandler("worker_logs/faiss_indexer_error.log")
error_file_handler.setLevel(logging.ERROR)
error_file_handler.setFormatter(formatter)

# Attach handlers
logger.addHandler(stream_handler)
logger.addHandler(file_handler)
logger.addHandler(error_file_handler)

# Optional: avoid duplicate logs if this file gets imported multiple times
logger.propagate = False

# Configuration
# Load environment variables from .env
load_dotenv("prod.env")

# Convert comma-separated extensions into tuple
def parse_extensions(ext_str):
    return tuple(e.strip() for e in ext_str.split(',') if e.strip())

# Configuration using environment variables
CONFIG = {
    "INDEX_PATH": os.getenv("INDEX_PATH", "faiss_index.bin"),
    "NAMES_PATH": os.getenv("NAMES_PATH", "image_names.npy"),
    "DATA_DIR": os.getenv("DATA_DIR", "./uploaded_images"),
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",  # remains dynamic
    "EMBEDDING_DIM": int(os.getenv("EMBEDDING_DIM", 512)),
    "RETRY_ATTEMPTS": int(os.getenv("RETRY_ATTEMPTS", 3)),
    "RETRY_DELAY": float(os.getenv("RETRY_DELAY", 1)),
    "WATCHDOG_POLL_INTERVAL": float(os.getenv("WATCHDOG_POLL_INTERVAL", 13.0)),
    "VALID_EXTENSIONS": parse_extensions(os.getenv("VALID_EXTENSIONS", ".png,.jpg,.jpeg")),
    "LOCK_TIMEOUT": float(os.getenv("LOCK_TIMEOUT", 15.0)),
    "PROCESSING_TIMEOUT": float(os.getenv("PROCESSING_TIMEOUT", 30.0)),
    "RECENTLY_PROCESSED_TTL": float(os.getenv("RECENTLY_PROCESSED_TTL", 300.0))
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

def initialize_index():
    """Initialize or load FAISS index and image names, and index existing images."""
    global index, image_names
    index_path = Path(CONFIG["INDEX_PATH"])
    names_path = Path(CONFIG["NAMES_PATH"])
    
    abs_data_dir = os.path.abspath(CONFIG["DATA_DIR"])
    logger.info(f"DATA_DIR absolute path: {abs_data_dir}")
    if not os.path.exists(abs_data_dir):
        logger.error(f"DATA_DIR {abs_data_dir} does not exist")
        sys.exit(1)
    if not os.access(abs_data_dir, os.R_OK | os.W_OK):
        logger.error(f"No read/write permissions for DATA_DIR {abs_data_dir}")
        sys.exit(1)
    
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
    
    logger.info("Starting initial indexing of existing images")
    index_existing_images()
    logger.info(f"Initial indexing completed, Index size: {len(image_names)} images")

def index_existing_images():
    """Scan DATA_DIR and index existing images sequentially."""
    global index, image_names
    data_dir = Path(CONFIG["DATA_DIR"])
    logger.info(f"Scanning {data_dir} for existing images...")
    
    image_files = [f for f in data_dir.iterdir() if f.suffix.lower() in CONFIG["VALID_EXTENSIONS"]]
    logger.info(f"Found {len(image_files)} images to process")
    
    images_added = False
    for file_path in image_files:
        filename = file_path.name
        try:
            # Check for duplicates first
            with index_lock:
                if filename in image_names:
                    logger.info(f"Skipping already indexed file: {filename}")
                    continue
            
            start_time = time.time()
            if not os.access(file_path, os.R_OK):
                logger.error(f"No read permission for {filename}")
                continue
            file_size = os.path.getsize(file_path)
            logger.info(f"Processing File {filename} size: {file_size} bytes")
            
            try:
                with Image.open(file_path) as image:
                    # logger.info(f"Verifying {filename}")  ## added for debuggng purpose
                    image.verify()
            except Exception as e:
                logger.error(f"Image verification failed for {filename}: {e}")
                continue
            
            # logger.info(f"Opening {filename} for embedding extraction")
            with Image.open(file_path) as image:
                vec = extract_embedding(image)
                if vec is None:
                    logger.warning(f"Embedding extraction failed for {filename}")
                    continue
                if time.time() - start_time > CONFIG["PROCESSING_TIMEOUT"]:
                    logger.error(f"Timeout processing {filename}")
                    continue
                
                with index_lock:
                    index.add(vec.reshape(1, -1))
                    image_names.append(filename)
                    images_added = True
                    logger.info(f"Indexed existing image: {filename}")
        except Exception as e:
            logger.error(f"Error processing existing image {filename}: {e}")
    
    if images_added:
        save_index_and_names()
    else:
        logger.info("No new images indexed, skipping save")

@retry(tries=CONFIG["RETRY_ATTEMPTS"], delay=CONFIG["RETRY_DELAY"], logger=logger)
def save_index_and_names():
    """Save FAISS index and image names with retries."""
    try:
        # logger.info("Attempting to save index and names")
        with index_lock:
            faiss.write_index(index, CONFIG["INDEX_PATH"])
            np.save(CONFIG["NAMES_PATH"], np.array(image_names, dtype=object))
            logger.info("Successfully saved FAISS index and image names")
    except Exception as e:
        logger.error(f"Error saving index and names: {e}")
        raise

def extract_embedding(image: Image.Image) -> np.ndarray:
    """Extract embedding from an input image."""
    try:
        image = image.convert("RGB")
        image = transform(image).unsqueeze(0).to(CONFIG["DEVICE"])
        # logger.info("Running model forward pass")
        with torch.no_grad():
            _ = model(image)
        vec = activation["avgpool"].cpu().numpy().squeeze()
        vec = vec / np.linalg.norm(vec, keepdims=True)
        # logger.info("Embedding extracted and normalized")
        return vec.astype(np.float32)
    except Exception as e:
        logger.error(f"Error extracting embedding: {e}")
        return None

class ImageFolderHandler(FileSystemEventHandler):
    def __init__(self):
        self.recently_processed = {}
        self.recently_processed_ttl = CONFIG["RECENTLY_PROCESSED_TTL"]
    
    def clean_recently_processed(self):
        """Remove entries from recently_processed older than TTL."""
        current_time = time.time()
        self.recently_processed = {
            fname: ts for fname, ts in self.recently_processed.items()
            if current_time - ts < self.recently_processed_ttl
        }
    
    # def on_any_event(self, event):
    #     """Log all watchdog events for debugging."""
    #     logger.info(f"Watchdog event: {event.event_type} at {event.src_path}")
    
    def on_created(self, event):
        """Handle new file creation events."""
        if event.is_directory or event.event_type != "created":
            return
        if not event.src_path.lower().endswith(CONFIG["VALID_EXTENSIONS"]):
            return
        
        filename = os.path.basename(event.src_path)
        logger.info(f"Detected new file: {event.src_path}")
        
        self.clean_recently_processed()
        if filename in self.recently_processed:
            logger.info(f"Ignoring duplicate event for {filename}")
            return
        
        time.sleep(CONFIG["WATCHDOG_POLL_INTERVAL"])
        self.recently_processed[filename] = time.time()
        
        try:
            start_time = time.time()
            if not os.access(event.src_path, os.R_OK):
                logger.error(f"No read permission for {filename}")
                return
            file_size = os.path.getsize(event.src_path)
            # logger.info(f"File {filename} size: {file_size} bytes")
            
            try:
                with Image.open(event.src_path) as image:
                    # logger.info(f"Verifying {filename}")
                    image.verify()
            except Exception as e:
                logger.error(f"Image verification failed for {filename}: {e}")
                return
            
            # logger.info(f"Opening {filename} for embedding extraction")
            with Image.open(event.src_path) as image:
                vec = extract_embedding(image)
                if vec is None:
                    logger.warning(f"Embedding extraction failed for {filename}")
                    return
                if time.time() - start_time > CONFIG["PROCESSING_TIMEOUT"]:
                    logger.error(f"Timeout processing {filename}")
                    return
                
                # Update index
                with index_lock:
                    if filename in image_names:
                        logger.info(f"Skipping already indexed file: {filename}")
                        return
                    index.add(vec.reshape(1, -1))
                    image_names.append(filename)
                    logger.info(f"Added new image: {filename}")
                
                # Save outside the lock
                save_index_and_names()
                
        except Exception as e:
            logger.error(f"Error processing {event.src_path}: {e}")
        finally:
            self.clean_recently_processed()

def signal_handler(sig, frame):
    """Handle shutdown signals."""
    logger.info("Received shutdown signal. Stopping observer...")
    observer.stop()
    observer.join()
    logger.info("Observer stopped. Exiting.")
    sys.exit(0)

if __name__ == '__main__':
    logger.info(f"Starting process with PID: {os.getpid()}")
    
    # Initialize index and image names, including existing images
    initialize_index()
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start file system watcher
    observer = Observer()
    observer.schedule(ImageFolderHandler(), CONFIG["DATA_DIR"], recursive=False)
    observer.start()
    logger.info(f"Started watching folder: {CONFIG['DATA_DIR']}, observer alive: {observer.is_alive()}")
    
    try:
        while True:
            time.sleep(CONFIG["WATCHDOG_POLL_INTERVAL"])
    except Exception as e:
        logger.error(f"Unexpected error in main loop: {e}")
        observer.stop()
        observer.join()
        logger.info("Observer stopped due to error")
        sys.exit(1)