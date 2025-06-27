from pymongo import MongoClient

# Replace with your actual MongoDB connection URI
client = MongoClient("mongodb://localhost:27017")

# Select your database and collection
db = client["image_cad_data"]
gallery_collection = db["galleries"]


def get_cad_file(image_path):
    doc = gallery_collection.find_one({"image": image_path})
    
    if doc:
        return doc.get("cadFile")
    else:
        return None


### testing locally
# image_path = "/uploads/img2.png"
# cad_file_path = get_cad_file(image_path)

# if cad_file_path:
#     print("Found CAD file:", cad_file_path)
# else:
#     print("Image not found in gallery.")

