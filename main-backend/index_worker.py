import os
import logging
from PIL import Image
import numpy as np
import faiss
import torch
import torchvision
from torchvision import transforms
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from threading import Lock
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
INDEX_PATH = "faiss_index.bin"
NAMES_PATH = "image_names.npy"
DATA_DIR = "../uploaded_images"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Thread lock for FAISS index updates
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

# Load FAISS index and image names
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
        return None

class ImageFolderHandler(FileSystemEventHandler):
    def on_created(self, event):
        """Handle new file creation events."""
        global index, image_names  # Declare global at the start
        if event.is_directory:
            return
        if event.src_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            logger.info(f"Detected new file: {event.src_path}")
            # Small delay to ensure file is fully written
            time.sleep(1)
            filename = os.path.basename(event.src_path)
            if filename in image_names:
                logger.info(f"Skipping already indexed file: {filename}")
                return
            try:
                image = Image.open(event.src_path)
                vec = extract_embedding(image)
                if vec is None:
                    return
                with index_lock:
                    index.add(vec.reshape(1, -1))
                    image_names.append(filename)
                    faiss.write_index(index, INDEX_PATH)
                    np.save(NAMES_PATH, np.array(image_names))
                logger.info(f"Added new image: {filename}")
            except Exception as e:
                logger.error(f"Error processing {event.src_path}: {e}")

if __name__ == '__main__':
    observer = Observer()
    observer.schedule(ImageFolderHandler(), DATA_DIR, recursive=False)
    observer.start()
    logger.info(f"Started watching folder: {DATA_DIR}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        logger.info("Stopped file system watcher")
    observer.join()