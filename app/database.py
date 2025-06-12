# app/database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://imagesdb_user:22hiRhiHyN4f5pailXiv1ieXFzxZkCwj@dpg-d14j6995pdvs73f83g6g-a.oregon-postgres.render.com/imagesdb")

# Add connection pooling configuration for production
engine = create_engine(
    DATABASE_URL,
    pool_size=5,  # Adjust based on your needs
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800,  # Recycle connections every 30 minutes
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
