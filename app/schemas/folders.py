from pydantic import BaseModel, Field
from uuid import UUID
from typing import List
from datetime import datetime


class FolderCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)


class FolderData(BaseModel):
    id: UUID
    name: str
    admin_id: UUID

    class Config:
        from_attributes = True


class FolderResponse(BaseModel):
    status_code: int
    message: str
    data: FolderData


class FolderRename(BaseModel):
    folder_id: UUID
    new_name: str


class FolderDelete(BaseModel):
    folder_id: UUID


class DeleteResponse(BaseModel):
    status_code: int
    message: str


class DataFolder(BaseModel):
    id: UUID
    name: str
    file_count: int
    created_at: datetime  # IST timestamp

    class Config:
        from_attributes = True


class FolderList(BaseModel):
    status_code: int
    data: List[DataFolder]
