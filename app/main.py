# app/main.py
from fastapi import FastAPI, UploadFile, File, Depends, Request
from sqlalchemy.orm import Session
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from app import database, models, schemas, extractor, search_engine
import numpy as np
from PIL import Image
import io
import os
import logging
import sys
from fastapi import HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
import magic


# Maximum file size (e.g., 10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB in bytes
CAD_FILE_TYPES = [
    "dwg",  # AutoCAD Drawing
    "dxf",  # AutoCAD Drawing Exchange Format
    "stp",  # STEP (ISO 10303)
    "step",  # STEP (ISO 10303)
    "igs",  # IGES
    "iges",  # IGES
    "ifc",  # Industry Foundation Classes (BIM)
    "sldprt",  # SolidWorks Part
    "sldasm",  # SolidWorks Assembly
    "slddrw",  # SolidWorks Drawing
    "catpart",  # CATIA Part
    "catproduct",  # CATIA Assembly
    "cgr",  # CATIA Graphical Representation
    "prt",  # PTC Creo Part / NX Part
    "asm",  # PTC Creo Assembly
    "drw",  # PTC Creo Drawing
    "ipt",  # Autodesk Inventor Part
    "iam",  # Autodesk Inventor Assembly
    "idw",  # Autodesk Inventor Drawing
    "f3d",  # Fusion 360 Design
    "f3z",  # Fusion 360 Archive
    "3dm",  # Rhino 3D Model
    "stl",  # Stereolithography (3D printing)
    "obj",  # Wavefront OBJ
    "3ds",  # 3D Studio
    "sat",  # ACIS
    "par",  # Solid Edge Part
    "psm",  # Solid Edge Sheet Metal
    "x_t",  # Parasolid (text)
    "x_b",  # Parasolid (binary)
    "jt",  # Jupiter Tessellation
    "dgn",  # MicroStation Design
    "rvt"   # Revit Project
]
# Supported MIME types for CAD files
CAD_MIME_TYPES = {
    "stp": "application/step",
    "step": "application/step",
    "dwg": "application/x-dwg",
    "dxf": "application/dxf",
    "ifc": "application/ifc",
}

app = FastAPI(title="Image Search API",
              description="API for searching similar images",
              version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("ALLOWED_ORIGINS", "*")],  # In production, specify exact domains
    allow_credentials=True,
    allow_methods=["*"],  # In production, specify exact methods (GET, POST, etc.)
    allow_headers=["*"],  # In production, specify exact headers
)

models.Base.metadata.create_all(bind=database.engine)

def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("image-search-api")

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

# Then add logging to your routes
@app.post("/add")
async def add_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
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
        logger.info(f"Added image: {file.filename}")
        return {"message": "Image added successfully"}
    except Exception as e:
        logger.error(f"Error adding image: {str(e)}")
        raise

@app.post("/v2/add")
async def add_image_v2(
    image: UploadFile = File(...),
    cad_file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Endpoint to upload and verify an image and CAD file, storing them in the database.
    """
    # Validate file sizes
    image_size = image.size
    cad_size = cad_file.size
    if image_size > MAX_FILE_SIZE or cad_size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File size exceeds 10MB limit")

    # Verify CAD file type using MIME
    try:

        # Verify file extension matches MIME type
        cad_extension = cad_file.filename.split(".")[-1].lower()
        if cad_extension not in CAD_FILE_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"CAD file must be one of: {', '.join(CAD_FILE_TYPES)}"
            )

        # Read full CAD file content
        cad_content = await cad_file.read()
        await cad_file.seek(0)  # Reset file pointer if needed later

        # Process image
        image_content = await image.read()
        try:
            with Image.open(io.BytesIO(image_content)) as img:
                img = img.convert("RGB")
                vec = extractor.extract_vector(img).astype(np.float32)
        except Exception as e:
            logger.error(f"Image processing error: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Store in database
        try:
            image_record = models.ImageVector(
                name=image.filename,
                vector=vec.tobytes(),
                image_data=image_content,
                cad_file=cad_content,
                cad_name=cad_file.filename
            )
            db.add(image_record)
            db.commit()
            logger.info(f"Added image: {image.filename}, CAD file: {cad_file.filename}")
            return {"message": "Image and CAD file added successfully"}
        except Exception as e:
            db.rollback()
            logger.error(f"Database error: {str(e)}")
            raise HTTPException(status_code=500, detail="Database error while saving files")
    
    except HTTPException:
        # Re-raise HTTPException to let FastAPI handle it
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    finally:
        # Ensure files are closed
        await image.close()
        await cad_file.close()


@app.post("/search", response_model=list[schemas.SearchResult])
async def search_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    request: Request = None
):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    vec = extractor.extract_vector(image).astype(np.float32)
    results = search_engine.search_similar_images(db, vec)

    return [
        {
            "name": r["name"],
            "similarity": r["similarity"],
            "img_url": f"{request.base_url}image/{r['name']}",  # Add URL to each result
            "cad_url": f"{request.base_url}cad/{r['cad_name']}"
        }
        for r in results
    ]

@app.get("/image/{name}")
def get_image(name: str, db: Session = Depends(get_db)):
    image_row = db.query(models.ImageVector).filter(models.ImageVector.name == name).first()
    if image_row:
        return StreamingResponse(io.BytesIO(image_row.image_data), media_type="image/jpeg")
    return {"error": "Image not found"}

@app.get("/cad/{name}")
async def get_cad_file(name: str, db: Session = Depends(get_db)):
    """
    Retrieve a CAD file by name from the database.
    """
    try:
        logger.info(f"Fetching CAD file: {name}")

        # Query database for record
        record = db.query(models.ImageVector).filter(models.ImageVector.cad_name == name).first()
        if not record or not record.cad_file:
            logger.error(f"CAD file not found for name: {name}")
            raise HTTPException(status_code=404, detail=f"CAD file '{name}' not found")

        # Determine MIME type based on extension
        extension = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        mime_type = CAD_MIME_TYPES.get(extension, "application/octet-stream")

        # Sanitize filename for Content-Disposition
        safe_filename = name.replace('"', '').replace('/', '_')

        # Return CAD file as a downloadable response
        return StreamingResponse(
            io.BytesIO(record.cad_file),
            media_type=mime_type,
            headers={
                "Content-Disposition": f"attachment; filename=\"{safe_filename}\"",
                "Content-Length": str(len(record.cad_file))
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving CAD file '{name}': {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

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
