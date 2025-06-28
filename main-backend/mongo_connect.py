from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from dotenv import load_dotenv
from math import ceil
import os

# Load environment variables from .env file
load_dotenv("prod.env")

# Get values from environment
mongodb_uri = os.getenv("MONGODB_URI")
db_name = os.getenv("MONGODB_DB")
collection_name = os.getenv("MONGODB_COLLECTION")

# Validate environment variables
if not all([mongodb_uri, db_name, collection_name]):
    # logger.error("Missing required environment variables: MONGODB_URI, MONGODB_DB, or MONGODB_COLLECTION")
    raise ValueError("Missing required environment variables")

# Initialize MongoDB connection
try:
    client = MongoClient(
        mongodb_uri,
        serverSelectionTimeoutMS=5000,  # 5-second timeout for server selection
        connectTimeoutMS=10000,         # 10-second timeout for connection
        maxPoolSize=50                 # Connection pool size for scalability
    )
    # Test connection
    client.admin.command("ping")
    db = client[db_name]
    gallery_collection = db[collection_name]
    # logger.info("Successfully connected to MongoDB")
except ConnectionFailure as e:
    # logger.error(f"Failed to connect to MongoDB: {e}")
    raise Exception("MongoDB connection failed")

def get_cad_file(image_path):
    doc = gallery_collection.find_one({"image": image_path})
    if doc:
        return doc.get("cadFile")
    else:
        return None

def list_all(base_url: str, page: int = 1, limit: int = 10):
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
        
    # Construct base URL if not provided (e.g., from Request)
    effective_base_url = base_url.rstrip("/") if base_url else ""
    
    image_data = [
        {
            "name": doc["imageName"],
            "url": f"{effective_base_url}{doc['image']}" if effective_base_url else doc["image"],
            "cad_url": f"{effective_base_url}{doc['cadFile']}" if effective_base_url else doc["cadFile"]
        }
        for doc in documents
    ]
    total_pages = ceil(total / limit) if total > 0 else 1
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

