from app.db.postgres.database import Base
from sqlalchemy.orm import mapped_column, Mapped, relationship
from datetime import datetime
from sqlalchemy import String, DateTime, func, ForeignKey, Boolean, text
import uuid
from sqlalchemy.dialects.postgresql import UUID


class Folders(Base):
    __tablename__ = "folders"
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        unique=True,
        default=uuid.uuid4,
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    admin_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("admins.id", ondelete="CASCADE"), nullable=False
    )
    deleted: Mapped[bool] = mapped_column(
        Boolean, server_default=text("false"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    admin = relationship("Admins", back_populates="created_folders")
    inserted_files = relationship(
        "Files", back_populates="folder", cascade="all,delete-orphan"
    )

    def __repr__(self):
        return f"folder {self.name} created by {self.admin_id}"


class Files(Base):
    __tablename__ = "files"
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        unique=True,
        default=uuid.uuid4,
        nullable=False,
    )
    original_filename: Mapped[str] = mapped_column(String, nullable=False)
    unique_name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    folder_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("folders.id", ondelete="CASCADE"), nullable=False
    )
    admin_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("admins.id", ondelete="CASCADE"), nullable=True
    )
    deleted: Mapped[bool] = mapped_column(
        Boolean, server_default=text("false"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    folder = relationship("Folders", back_populates="inserted_files")
    admin = relationship("Admins", back_populates="created_files")

    def __repr__(self):
        return f"folder {self.unique_name} created by {self.admin_id} of folder {self.folder_id}"
