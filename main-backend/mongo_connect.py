from pymongo import MongoClient
from dotenv import load_dotenv
from math import ceil
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

def list_all(page: int = 1, limit: int = 10):
    # Ensure page and limit are positive
    page = max(1, page)
    limit = max(1, limit)
    
    # Calculate skip value for pagination
    skip = (page - 1) * limit
    
    # Get total number of documents
    total = gallery_collection.count_documents({})
    
    # Fetch paginated documents
    documents = gallery_collection.find(
        {}, 
        {"_id": 0, "image": 1, "cadFile": 1, "imageName": 1}
    ).skip(skip).limit(limit)
    
    # Format the image data
    image_data = [
        {
            "name": doc["imageName"],
            "url": doc["image"],
            "cad_url": doc["cadFile"]
        }
        for doc in documents
    ]
    
    # Calculate total pages
    total_pages = ceil(total / limit) if total > 0 else 1
    
    # Return paginated response
    return {
        "data": image_data,
        "total": total,
        "page": page,
        "pages": total_pages
    }

# print(list_all())

### testing locally
# image_path = "/uploads/A2.jpg"
# cad_file_path = get_cad_file(image_path)

# if cad_file_path:
#     print("Found CAD file:", cad_file_path)
# else:
#     print("Image not found in gallery.")

