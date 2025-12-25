from pydantic import BaseModel, Field
from uuid import UUID
from typing import List


class UploadFileData(BaseModel):
    id: UUID
    original_filename: str
    unique_name: str
    folder_id: UUID
    admin_id: UUID

    class Config:
        from_attributes = True  # VERY IMPORTANT for SQLAlchemy


class upload_file_response(BaseModel):
    status_code: int
    message: str
    uploaded_files: List[UploadFileData]
    skipped_files: List[str]


class delete_file_body(BaseModel):
    file_ids: List[UUID] = Field(..., min_length=1)


class delete_response(BaseModel):
    status: int
    message: str


class get_files(BaseModel):
    folder_id: UUID


class file(BaseModel):
    id: UUID
    original_filename: str
    unique_name: str


class get_files_response(BaseModel):
    status_code: int
    message: str
    data: List[file]
