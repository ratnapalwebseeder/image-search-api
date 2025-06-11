# extractor.py
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np

# Load ResNet18
model = torchvision.models.resnet18(weights="DEFAULT")
model.eval()
activation = {}

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model.avgpool.register_forward_hook(get_activation("avgpool"))

# Transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def extract_vector(image: Image.Image) -> np.ndarray:
    img_tensor = transform(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        _ = model(img_tensor)
        vec = activation["avgpool"].numpy().squeeze()
        vec = vec / np.linalg.norm(vec)  # L2 normalize
        return vec
