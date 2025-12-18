from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.schemas.admin import CreateAdmin, GetAdmins, AdminLogin
from app.dependencies import get_db, get_current_admin
from app.models.admins import Admins
from sqlalchemy import select
from app.utils.password import hash_password, verify_password
from app.tasks.email import send_email_task
from app.utils.tokens import create_access_token, create_refresh_token

router = APIRouter(prefix="/admin", tags=["User"])


@router.post("/signup", status_code=status.HTTP_201_CREATED)
async def signup(admin: CreateAdmin, db: AsyncSession = Depends(get_db)):
    existance = await db.execute(select(Admins).where(Admins.email == admin.email))
    exist = existance.scalar_one_or_none()
    if exist:
        raise HTTPException(
            status_code=status.HTTP_226_IM_USED, detail="Email already registered"
        )
    password = hash_password(admin.password)

    new_user = Admins(username=admin.user_name, email=admin.email, password=password)
    db.add(new_user)
    await db.commit()

    subject = "Testing - Signup Successful"
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Welcome!</title>
    </head>
    <body style="font-family: Arial, sans-serif; background-color: #f4f4f4; padding: 20px;">
        <div style="max-width: 600px; margin: auto; background-color: white; padding: 30px; border-radius: 10px; text-align: center;">
            <h1 style="color: #4CAF50;">Welcome to the Team!</h1>
            <p>Hi there,</p>
            <p>We are thrilled to have you on board. Get ready for an exciting journey!</p>
            <p>Best regards,<br>Your Company Name</p>
        </div>
    </body>
    </html>
    """
    # send_email_task.delay(admin.email, subject, html_content)
    return {
        "status_code": status.HTTP_201_CREATED,
        "message": "User registered successfully",
    }


@router.post("/login", status_code=status.HTTP_200_OK)
async def login(credentials: AdminLogin, db: AsyncSession = Depends(get_db)):
    existense = await db.execute(
        select(Admins).where(Admins.email == credentials.email)
    )
    admin = existense.scalar_one_or_none()

    if not admin or not verify_password(credentials.password, admin.password):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Invalid email or password"
        )
    access_token = create_access_token({"sub": str(admin.id), "email": admin.email})
    refresh_token = create_refresh_token({"sub": str(admin.id)})
    return {
        "status_code": status.HTTP_200_OK,
        "access_token": access_token,
        "refresh_token": refresh_token,
        "message": "Admin loggedin successfully",
    }


@router.get("/getuser")
async def user_details(current_admin: Admins = Depends(get_current_admin)):
    return {
        "status_code": status.HTTP_200_OK,
        "id": current_admin.id,
        "user_name": current_admin.username,
        "email": current_admin.email,
    }


@router.get("/getadmins", response_model=GetAdmins)
async def user_details(
    current_admin: Admins = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Admins))
    admins = result.scalars().all()
    return {
        "status_code": status.HTTP_200_OK,
        "admins": admins,
    }
