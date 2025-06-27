from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv("prod.env")

# Get values from environment
mongodb_uri = os.getenv("MONGODB_URI")
db_name = os.getenv("MONGODB_DB")
collection_name = os.getenv("MONGODB_COLLECTION")

# Connect to MongoDB
client = MongoClient(mongodb_uri)
db = client[db_name]
gallery_collection = db[collection_name]

def get_cad_file(image_path):
    doc = gallery_collection.find_one({"image": image_path})
    if doc:
        return doc.get("cadFile")
    else:
        return None

def list_all():
    documents = gallery_collection.find({}, {"_id": 0, "image": 1, "cadFile": 1, "imageName":1})
    image_data = [
        {
            "name": doc["imageName"],
            "image_url": doc["image"],
            "cad_url": doc["cadFile"]
        }
        for doc in documents
    ]
    return image_data

print(list_all())

### testing locally
# image_path = "/uploads/A2.jpg"
# cad_file_path = get_cad_file(image_path)

# if cad_file_path:
#     print("Found CAD file:", cad_file_path)
# else:
#     print("Image not found in gallery.")

