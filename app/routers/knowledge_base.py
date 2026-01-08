from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    status,
    UploadFile,
    File,
    Form,
)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
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
from app.schemas.files import (
    upload_file_response,
    delete_file_body,
    delete_response,
    get_files,
    get_files_response,
)
from sqlalchemy.exc import IntegrityError
from uuid import UUID
from pathlib import Path
import uuid
from sqlalchemy.orm import aliased
from typing import List
from app.core.config import settings
from app.tasks.embedding_tasks import process_uploaded_files
from pytz import timezone
from fastapi.responses import FileResponse

IST = timezone("Asia/Kolkata")


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
    except IntegrityError as e:
        await db.rollback()
        print(e)
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
    # 1. Fetch folder
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

    # 2. Ownership check
    if folder.admin_id != current_admin.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not authorized to delete this folder.",
        )

    # 3. Check for files uploaded by other admin
    result = await db.execute(
        select(Files.id)
        .where(
            Files.folder_id == folder.id,
            Files.deleted.is_(False),
            Files.admin_id.is_not(None),
            Files.admin_id != current_admin.id,
        )
        .limit(1)
    )
    foreign_file = result.scalar_one_or_none()
    if foreign_file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Folder cannot be deleted because it contains files"
            "uploaded by other admins.",
        )

    # 4. Fetch all files in the folder to delete the embeddings
    result = await db.execute(
        select(Files).where(Files.folder_id == folder.id, Files.deleted.is_(False))
    )
    files = result.scalars().all()
    deleted_file_ids = []
    for file in files:
        file.deleted = True
        deleted_file_ids.append(file.id)

    # 5. Soft-delete folder
    folder.deleted = True
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to delete the file. Try again!",
        )
    if deleted_file_ids:
        from app.tasks.embedding_tasks import delete_files_embeddings_task

        delete_files_embeddings_task.delay(
            file_ids=deleted_file_ids,
        )
    return {
        "status_code": status.HTTP_204_NO_CONTENT,
        "message": "Folder Deleted Successfully!",
    }


@router.get(
    "/folder_list",
    status_code=status.HTTP_200_OK,
    response_model=FolderList,
)
async def available_folders(
    db: AsyncSession = Depends(get_db),
    current_admin: Admins = Depends(get_current_admin),
):
    stmt = (
        select(
            Folders.id,
            Folders.name,
            Folders.created_at,
            func.count(Files.id).label("file_count"),
        )
        .outerjoin(Files, Files.folder_id == Folders.id)
        .where(Folders.deleted.isnot(True))
        .group_by(Folders.id)
        .order_by(Folders.created_at.desc())
    )

    result = await db.execute(stmt)
    rows = result.all()

    data = []
    for row in rows:
        created_at_ist = (
            row.created_at.astimezone(IST)
            if row.created_at.tzinfo
            else IST.localize(row.created_at)
        )

        data.append(
            {
                "id": row.id,
                "name": row.name,
                "file_count": row.file_count,
                "created_at": created_at_ist,
            }
        )

    return {
        "status_code": status.HTTP_200_OK,
        "data": data,
    }


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


UPLOAD_DIR = Path(settings.upload_dir)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post(
    "/upload_files",
    status_code=status.HTTP_201_CREATED,
    response_model=upload_file_response,
)
async def upload_files(
    folder_id: UUID = Form(...),
    files: List[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db),
    current_admin: Admins = Depends(get_current_admin),
):
    """Upload files and trigger embedding generation."""
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided",
        )

    # Validate folder ownership
    result = await db.execute(
        select(Folders).where(
            Folders.id == folder_id,
            Folders.deleted.is_(False),
        )
    )
    folder = result.scalar_one_or_none()
    if not folder:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Folder not found",
        )

    uploaded_files: List[Files] = []
    skipped_files: List[str] = []

    for upload in files:
        ext = Path(upload.filename).suffix.lower()

        if ext not in settings.allowed_extensions:
            skipped_files.append(upload.filename)
            continue

        unique_name = f"{uuid.uuid4()}{ext}"
        file_path = UPLOAD_DIR / unique_name

        try:
            content = await upload.read()
            with file_path.open("wb") as buffer:
                buffer.write(content)
        except Exception:
            skipped_files.append(upload.filename)
            continue

        db_file = Files(
            original_filename=upload.filename,
            unique_name=unique_name,
            folder_id=folder.id,
            admin_id=current_admin.id,
        )
        db.add(db_file)
        uploaded_files.append(db_file)

    if not uploaded_files:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="No supported files uploaded",
        )

    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        for f in uploaded_files:
            fp = UPLOAD_DIR / f.unique_name
            if fp.exists():
                fp.unlink()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to upload files",
        )

    for f in uploaded_files:
        await db.refresh(f)

    # Trigger Celery task
    file_data = [
        {
            "file_id": str(f.id),
            "unique_name": f.unique_name,
            "original_filename": f.original_filename,
            "extension": Path(f.original_filename).suffix.lower(),
            "admin_id": str(current_admin.id),
            "folder_id": str(folder_id),
        }
        for f in uploaded_files
    ]

    task = process_uploaded_files.delay(file_data)

    return {
        "status_code": status.HTTP_201_CREATED,
        "message": (
            "Files uploaded partially"
            if skipped_files
            else "Files uploaded successfully"
        ),
        "uploaded_files": uploaded_files,
        "skipped_files": skipped_files,
        "task_id": task.id,
    }


@router.get(
    "/download_file",
    status_code=status.HTTP_200_OK,
)
async def download_file(
    file_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_admin: Admins = Depends(get_current_admin),
):
    # 1. Fetch file record
    result = await db.execute(
        select(Files).where(
            Files.id == file_id,
            Files.deleted.is_(False),
        )
    )
    file = result.scalar_one_or_none()

    if not file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found",
        )

    # 2. Authorization check
    # if file.admin_id != current_admin.id:
    #     raise HTTPException(
    #         status_code=status.HTTP_403_FORBIDDEN,
    #         detail="You are not authorized to download this file",
    #     )

    # 3. Resolve file path
    file_path = UPLOAD_DIR / file.unique_name

    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="File does not exist on disk",
        )

    # 4. Send file with original filename
    return FileResponse(
        path=file_path,
        filename=file.original_filename,
        media_type="application/octet-stream",
    )


@router.get("/task/{task_id}")
async def get_task_status(
    task_id: str,
    current_admin: Admins = Depends(get_current_admin),
):
    """Check the status of an embedding generation task."""
    from app.core.celery_app import celery_app

    result = celery_app.AsyncResult(task_id)

    response = {
        "task_id": task_id,
        "status": result.status,
    }

    if result.ready():
        if result.successful():
            response["result"] = result.get()
        else:
            response["error"] = str(result.result)

    return response


@router.delete(
    "/delete_file",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_file(
    payload: delete_file_body,
    db: AsyncSession = Depends(get_db),
    current_admin: Admins = Depends(get_current_admin),
):
    file_ids = payload.file_ids

    result = await db.execute(
        select(Files).where(
            Files.id.in_(file_ids),
            Files.admin_id == current_admin.id,
            Files.deleted.is_(False),
        )
    )
    files = result.scalars().all()

    if not files:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No file for you to delete",
        )

    deleted_file_ids: List[str] = []
    deleted_unique_names: List[str] = []

    for file in files:
        file.deleted = True
        deleted_file_ids.append(str(file.id))
        deleted_unique_names.append(file.unique_name)

    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to delete files",
        )

    # ✅ Use batch delete task (more efficient)
    from app.tasks.embedding_tasks import delete_files_embeddings_task

    delete_files_embeddings_task.delay(
        file_ids=deleted_file_ids,
    )

    return


@router.get(
    "/files",
    status_code=status.HTTP_200_OK,
    response_model=get_files_response,
)
async def get_folder_files(
    folder_id: UUID = Query(..., description="Folder ID"),
    db: AsyncSession = Depends(get_db),
    current_admin: Admins = Depends(get_current_admin),
):
    # 1️⃣ Validate folder
    result = await db.execute(
        select(Folders).where(
            Folders.id == folder_id,
            Folders.deleted.is_(False),
        )
    )
    folder = result.scalar_one_or_none()

    if not folder:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Folder not Found.",
        )

    # 2️⃣ Fetch files with creator info
    stmt = (
        select(
            Files.id,
            Files.original_filename,
            Files.unique_name,
            Files.admin_id,
            Files.created_at,
            Files.updated_at,
            Admins.username.label("created_by"),
        )
        .join(Admins, Files.admin_id == Admins.id)
        .where(
            Files.folder_id == folder.id,
            Files.deleted.is_(False),
        )
        .order_by(Files.created_at.desc())
    )

    result = await db.execute(stmt)
    rows = result.all()

    data = []
    for row in rows:
        created_at_ist = (
            row.created_at.astimezone(IST)
            if row.created_at.tzinfo
            else IST.localize(row.created_at)
        )
        updated_at_ist = (
            row.updated_at.astimezone(IST)
            if row.updated_at.tzinfo
            else IST.localize(row.updated_at)
        )

        data.append(
            {
                "id": row.id,
                "original_filename": row.original_filename,
                "unique_name": row.unique_name,
                "admin_id": row.admin_id,
                "created_by": row.created_by,
                "created_at": created_at_ist,
                "updated_at": updated_at_ist
            }
        )

    return {
        "status_code": status.HTTP_200_OK,
        "message": "Files fetched successfully",
        "data": data,
    }


@router.get(
    "/deleted_files",
    status_code=status.HTTP_200_OK,
    response_model=get_files_response,
)
async def get_deleted_files(
    folder_id: UUID = Query(..., description="Folder ID"),
    db: AsyncSession = Depends(get_db),
    current_admin: Admins = Depends(get_current_admin),
):
    # 1️⃣ Validate folder
    result = await db.execute(
        select(Folders).where(
            Folders.id == folder_id,
            Folders.deleted.is_(False),
        )
    )
    folder = result.scalar_one_or_none()

    if not folder:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Folder not Found.",
        )

    # 2️⃣ Fetch files with creator info
    stmt = (
        select(
            Files.id,
            Files.original_filename,
            Files.unique_name,
            Files.admin_id,
            Files.created_at,
            Files.updated_at,
            Admins.username.label("created_by"),
        )
        .join(Admins, Files.admin_id == Admins.id)
        .where(
            Files.folder_id == folder.id,
            Files.deleted.is_(True),
        )
        .order_by(Files.created_at.desc())
    )

    result = await db.execute(stmt)
    rows = result.all()

    data = []
    for row in rows:
        created_at_ist = (
            row.created_at.astimezone(IST)
            if row.created_at.tzinfo
            else IST.localize(row.created_at)
        )
        updated_at_ist = (
            row.updated_at.astimezone(IST)
            if row.updated_at.tzinfo
            else IST.localize(row.updated_at)
        )

        data.append(
            {
                "id": row.id,
                "original_filename": row.original_filename,
                "unique_name": row.unique_name,
                "admin_id": row.admin_id,
                "created_by": row.created_by,
                "created_at": created_at_ist,
                "updated_at": updated_at_ist,
            }
        )

    return {
        "status_code": status.HTTP_200_OK,
        "message": "Files fetched successfully",
        "data": data,
    }
