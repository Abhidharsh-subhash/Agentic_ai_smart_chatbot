from pydantic import BaseModel, EmailStr, Field
from typing import List


class CreateAdmin(BaseModel):
    user_name: str = Field(..., max_length=255, example="JohnDoe")
    email: EmailStr = Field(..., example="johndoe@example.com")
    password: str = Field(..., min_length=6, example="securePassword123")


class GetAdmin(BaseModel):
    username: str
    email: EmailStr

    model_config = {"from_attributes": True}


class GetAdmins(BaseModel):
    status_code: int
    admins: List[GetAdmin]


class AdminLogin(BaseModel):
    email: EmailStr = Field(..., example="johndoe@example.com")
    password: str = Field(..., min_length=6, example="securePassword123")
