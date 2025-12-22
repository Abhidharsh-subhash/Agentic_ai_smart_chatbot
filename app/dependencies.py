from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.postgres.database import async_session
from app.db.redis.redis import redis_client
from fastapi import Request, Depends, HTTPException, status
from jose import jwt, JWTError
from app.core.config import settings
from app.models.admins import Admins
from sqlalchemy import select


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        yield session


async def get_redis():
    try:
        yield redis_client
    finally:
        pass


async def get_current_admin(request: Request, db: AsyncSession = Depends(get_db)):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or Invalid Authorization Header",
        )
    token = auth_header.split(" ")[1]

    try:
        payload = jwt.decode(token, settings.secret_key, [settings.algorithm])
        admin_id: str = payload.get("sub")
        if admin_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Token"
            )
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or Expired Token"
        )
    result = await db.execute(select(Admins).where(Admins.id == admin_id))
    admin = result.scalar_one_or_none()

    if admin is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not Found"
        )
    return admin
