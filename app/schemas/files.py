from pydantic import BaseModel, Field
from uuid import UUID
from typing import List


class upload_file_data(BaseModel):
    id: UUID
    original_filename: str
    unique_name: str
    folder_id: UUID


class upload_file_response(BaseModel):
    status_code: int
    message: str
    data: upload_file_data


class delete_file_body(BaseModel):
    file_ids: List[UUID] = Field(..., min_length=1)


class delete_response(BaseModel):
    status: int
    message: str
