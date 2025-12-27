from pydantic import BaseModel, Field
from uuid import UUID
from typing import List, Optional
from datetime import datetime


class FileResponse(BaseModel):
    id: UUID
    original_filename: str
    unique_name: str
    folder_id: UUID
    created_at: datetime

    class Config:
        from_attributes = True


class upload_file_response(BaseModel):
    status_code: int
    message: str
    uploaded_files: List[FileResponse]
    skipped_files: List[str]
    task_id: Optional[str] = None  # Celery task ID for tracking


class delete_file_body(BaseModel):
    file_ids: List[UUID] = Field(..., min_length=1)


class delete_response(BaseModel):
    status: int
    message: str


class get_files(BaseModel):
    folder_id: UUID


class FileData(BaseModel):
    id: UUID
    original_filename: str
    unique_name: str
    created_by: str        # admin username
    created_at: datetime  # IST

    class Config:
        from_attributes = True


class get_files_response(BaseModel):
    status_code: int
    message: str
    data: List[FileData]
