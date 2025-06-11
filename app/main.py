# app/main.py
from fastapi import FastAPI, UploadFile, File, Depends, Request
from sqlalchemy.orm import Session
from fastapi.responses import StreamingResponse
from app import database, models, schemas, extractor, search_engine
import numpy as np
from PIL import Image
import io

app = FastAPI()
models.Base.metadata.create_all(bind=database.engine)

def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/add")
async def add_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    vec = extractor.extract_vector(image).astype(np.float32)

    image_record = models.ImageVector(
        name=file.filename,
        vector=vec.tobytes(),
        image_data=contents
    )
    db.add(image_record)
    db.commit()
    return {"message": "Image added successfully"}

@app.post("/search", response_model=list[schemas.SearchResult])
async def search_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    request: Request = None  # 👈 Add this
):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    vec = extractor.extract_vector(image).astype(np.float32)
    results = search_engine.search_similar_images(db, vec)

    return [
        {
            "name": r["name"],
            "similarity": r["similarity"],
            "url": f"{request.base_url}image/{r['name']}"  # Add URL to each result
        }
        for r in results
    ]

@app.get("/image/{name}")
def get_image(name: str, db: Session = Depends(get_db)):
    image_row = db.query(models.ImageVector).filter(models.ImageVector.name == name).first()
    if image_row:
        return StreamingResponse(io.BytesIO(image_row.image_data), media_type="image/jpeg")
    return {"error": "Image not found"}

@app.get("/list", response_model=schemas.PaginatedImageList)
def list_images(
    page: int = 1,
    per_page: int = 10,
    db: Session = Depends(get_db),
    request: Request = None
):
    # Get total count
    total = db.query(models.ImageVector).count()
    
    # Calculate pagination
    pages = (total + per_page - 1) // per_page
    offset = (page - 1) * per_page
    
    # Get paginated items
    images = db.query(models.ImageVector)\
        .order_by(models.ImageVector.id.desc())\
        .offset(offset)\
        .limit(per_page)\
        .all()
    
    items = [
        schemas.ImageListItem(
            name=img.name,
            url=f"{request.base_url}image/{img.name}"
        ) for img in images
    ]
    
    return schemas.PaginatedImageList(
        items=items,
        total=total,
        page=page,
        pages=pages
    )
