from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.dependencies import get_current_admin, get_db
from app.models.admins import Admins
from app.models.knowledge_base import Folders, Files
from app.schemas.folders import FolderCreate, FolderResponse, FolderRename
from sqlalchemy.exc import IntegrityError


router = APIRouter(prefix="/file_handling", tags=["KnowledgeBase"])


@router.post(
    "/folder_creation",
    status_code=status.HTTP_201_CREATED,
    response_model=FolderResponse,
)
async def create_folder(
    payload: FolderCreate,
    db: AsyncSession = Depends(get_db),
    current_admin: Admins = Depends(get_current_admin),
):
    folder_existing = await db.execute(
        select(Folders).where(Folders.name == payload.name, Folders.deleted.is_(False))
    )
    if folder_existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="Folder name already exist"
        )
    folder = Folders(name=payload.name, admin_id=current_admin.id)
    db.add(folder)

    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Folder Creation Failed! Try again.",
        )
    await db.refresh(folder)
    return {
        "status_code": status.HTTP_201_CREATED,
        "message": "Folder created successfully!",
        "data": folder,
    }


@router.put(
    "rename_folder", status_code=status.HTTP_200_OK, response_model=FolderResponse
)
async def rename_folder(
    payload: FolderRename,
    db: AsyncSession = Depends(get_db),
    current_admin: Admins = Depends(get_current_admin),
):
    result = await db.execute(
        select(Folders).where(
            Folders.id == payload.folder_id,
            Folders.admin_id == current_admin.id,
            Folders.deleted.is_(False),
        )
    )
    folder = result.scalar_one_or_none()
    if not folder:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Folder not Found."
        )

    check_duplicate = await db.execute(
        select(Folders).where(
            Folders.name == payload.new_name, Folders.deleted.is_(False)
        )
    )
    if check_duplicate.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="Folder name already exist"
        )
    folder.name = payload.new_name
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to rename Folder, Try again!",
        )
    await db.refresh(folder)
    return {
        "status_code": status.HTTP_200_OK,
        "message": "Folder renamed successfully!",
        "data": folder,
    }


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
