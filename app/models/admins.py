from app.db.postgres.database import Base
from sqlalchemy.orm import mapped_column, Mapped, relationship
from sqlalchemy import String, DateTime, func
from datetime import datetime
from sqlalchemy.dialects.postgresql import UUID
import uuid


class Admins(Base):
    __tablename__ = "admins"
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        unique=True,
        nullable=False,
    )
    email: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    username: Mapped[str] = mapped_column(String, nullable=False)
    password: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    created_folders = relationship(
        "Folders", back_populates="admin", cascade="all,delete-orphan"
    )

    def __repr__(self):
        return f"user with the mail {self.email}"
