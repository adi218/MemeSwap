from fastapi import APIRouter
from app.api.endpoints import face_swap, gif_search

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(gif_search.router, prefix="/gif", tags=["gif"])
api_router.include_router(face_swap.router, prefix="/face", tags=["face"]) 