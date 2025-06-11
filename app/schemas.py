# app/schemas.py
from pydantic import BaseModel

class SearchResult(BaseModel):
    name: str
    similarity: float
    url: str

class ImageListItem(BaseModel):
    name: str
    url: str

class PaginatedImageList(BaseModel):
    items: list[ImageListItem]
    total: int
    page: int
    pages: int