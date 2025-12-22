from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.dependencies import get_current_admin, get_db
from app.models.admins import Admins
from app.models.knowledge_base import Folders, Files
from app.schemas.folders import (
    FolderCreate,
    FolderResponse,
    FolderRename,
    FolderDelete,
    DeleteResponse,
    FolderList,
)
from app.schemas.files import upload_file_response, delete_file_body, delete_response
from sqlalchemy.exc import IntegrityError
from uuid import UUID
from pathlib import Path
import uuid
from sqlalchemy.orm import selectinload


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
    print(folder)
    return {
        "status_code": status.HTTP_201_CREATED,
        "message": "Folder created successfully!",
        "data": folder,
    }


@router.put(
    "/rename_folder", status_code=status.HTTP_200_OK, response_model=FolderResponse
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


@router.delete(
    "/delete_folder", status_code=status.HTTP_200_OK, response_model=DeleteResponse
)
async def delete_folder(
    payload: FolderDelete,
    db: AsyncSession = Depends(get_db),
    current_admin: Admins = Depends(get_current_admin),
):
    result = await db.execute(
        select(Folders).where(
            Folders.id == payload.folder_id,
            Folders.deleted.is_(False),
        )
    )
    folder = result.scalar_one_or_none()
    if not folder:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Folder does not exist!"
        )
    if folder.admin_id != current_admin.id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="You are not authorized to delete the folder.",
        )
    folder.deleted = True
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to delete the file. Try again!",
        )
    return {
        "status_code": status.HTTP_204_NO_CONTENT,
        "message": "Folder Deleted Successfully!",
    }


@router.get("/folder_list", status_code=status.HTTP_200_OK, response_model=FolderList)
async def available_folders(
    db: AsyncSession = Depends(get_db),
    current_admin: Admins = Depends(get_current_admin),
):
    result = await db.execute(select(Folders).where(Folders.deleted.isnot(True)))
    folders = result.scalars().all()
    if len(folders) == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="No Folder Found"
        )
    return {"status_code": status.HTTP_200_OK, "data": folders}


@router.get(
    "/deleted_folder_list", status_code=status.HTTP_200_OK, response_model=FolderList
)
async def deleted_folders(
    db: AsyncSession = Depends(get_db),
    current_admin: Admins = Depends(get_current_admin),
):
    result = await db.execute(select(Folders).where(Folders.deleted.isnot(False)))
    folders = result.scalars().all()
    if len(folders) == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="No Folders Found."
        )
    return {"status_code": status.HTTP_200_OK, "data": folders}


UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@router.post(
    "/upload_file",
    status_code=status.HTTP_201_CREATED,
    response_model=upload_file_response,
)
async def file_upload(
    folder_id: UUID = Form(...),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_admin: Admins = Depends(get_current_admin),
):
    # validate folder
    result = await db.execute(
        select(Folders).where(
            Folders.id == folder_id,
            Folders.admin_id == current_admin.id,
            Folders.deleted.is_(False),
        )
    )
    folder = result.scalar_one_or_none()
    if not folder:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Folder not found"
        )

    # generate unique filename
    ext = Path(file.filename).suffix
    unique_name = f"{uuid.uuid4()}{ext}"

    # save file to disk
    file_path = UPLOAD_DIR / unique_name
    try:
        with file_path.open("wb") as buffer:
            buffer.write(await file.read())
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to save file"
        )

    # save data to database
    db_file = Files(
        original_filename=file.filename, unique_name=unique_name, folder_id=folder.id
    )
    db.add(db_file)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to upload the file, Try again!",
        )
    await db.refresh(db_file)
    return {
        "status_code": status.HTTP_201_CREATED,
        "message": "File Uploaded Successfully!",
        "data": db_file,
    }


@router.delete(
    "/delete_file",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_file(
    payload: delete_file_body,
    db: AsyncSession = Depends(get_db),
    current_admin: Admins = Depends(get_current_admin),
):
    if not payload.file_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Field id's cannot be emtpy"
        )

    # get first file only
    first_file_id = payload.file_ids[0]
    result = await db.execute(
        select(Files)
        .options(selectinload(Files.folder))
        .where(Files.id == first_file_id, Files.deleted.is_(False))
    )
    first_file = result.scalar_one_or_none()
    if not first_file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="File not Found"
        )

    # authorization check using folder
    if first_file.folder.admin_id != current_admin.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not authorized to delete files from the folder.",
        )

    # db operations
    result = await db.execute(
        select(Files).where(Files.id.in_(payload.file_ids), Files.deleted.is_(False))
    )
    files = result.scalars().all()
    if not files:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No valid files found to delete",
        )
    for file in files:
        file.deleted = True
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to delete the files"
        )
    if len(payload.file_ids) == 1:
        message = "File delete successfully"
    else:
        message = "Files delete successfully"
    # return {"status_code": status.HTTP_204_NO_CONTENT, "message": message}
    return
