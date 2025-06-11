# app/models.py
from sqlalchemy import Column, Integer, String, LargeBinary
from app.database import Base

class ImageVector(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True)
    vector = Column(LargeBinary)
    image_data = Column(LargeBinary)
