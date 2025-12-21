from pydantic import BaseModel, Field
from uuid import UUID


class FolderCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)


class FolderData(BaseModel):
    id: UUID
    name: str
    admin: UUID

    class Config:
        from_attributes = True


class FolderResponse(BaseModel):
    status_code: int
    message: str
    data: FolderData


class FolderRename(BaseModel):
    folder_id: UUID
    new_name: str
