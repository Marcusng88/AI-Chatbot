from fastapi import APIRouter

# Import routers here
from app.api.v1.endpoints import archives, ai_search

api_router = APIRouter()

# Include routers here
api_router.include_router(archives.router, prefix="", tags=["archives"])
api_router.include_router(ai_search.router, prefix="", tags=["ai-search"])

