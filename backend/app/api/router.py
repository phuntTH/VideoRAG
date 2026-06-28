from fastapi import APIRouter
from app.api.endpoints import video, search

api_router = APIRouter()

api_router.include_router(video.router, prefix="/video", tags=["Video Processing"])
api_router.include_router(search.router, prefix="/search", tags=["Search"])
