from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.dependencies import get_current_admin, get_db
from app.models.admins import Admins


router = APIRouter(prefix="/file_handling", tags=["KnowledgeBase"])


@router.post("/folder_creation", status_code=status.HTTP_201_CREATED)
async def create_folder(
    db: AsyncSession = Depends(get_db),
    current_admin: Admins = Depends(get_current_admin),
):
    pass


@router.put("rename_folder", status_code=status.HTTP_200_OK)
async def rename_folder(
    db: AsyncSession = Depends(get_db),
    current_admin: Admins = Depends(get_current_admin),
):
    pass


@router.delete("delete_folder", status_code=status.HTTP_200_OK)
async def delete_folder(
    db: AsyncSession = Depends(get_db),
    current_admin: Admins = Depends(get_current_admin),
):
    pass


@router.post("/{folder_id}/upload_file", status_code=status.HTTP_201_CREATED)
async def file_upload(
    db: AsyncSession = Depends(get_db),
    current_admin: Admins = Depends(get_current_admin),
):
    pass


@router.delete("/{folder_id}/delete_file", status_code=status.HTTP_204_NO_CONTENT)
async def delete_file(
    db: AsyncSession = Depends(get_db),
    current_admin: Admins = Depends(get_current_admin),
):
    pass
