from fastapi import APIRouter
from app.routers import admins

api_router = APIRouter()

api_router.include_router(admins.router)

__all__ = ["api_router"]
