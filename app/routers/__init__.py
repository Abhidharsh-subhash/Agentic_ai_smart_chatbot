from fastapi import APIRouter
from app.routers import admins, knowledge_base

api_router = APIRouter()

api_router.include_router(admins.router)
api_router.include_router(knowledge_base.router)

__all__ = ["api_router"]
