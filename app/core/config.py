from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Set
from pathlib import Path


class Settings(BaseSettings):
    # GENERAL
    debug: bool = Field(False, env="DEBUG")

    # DATABASE
    database_name: str = Field(env="DATABASE_NAME")
    database_username: str = Field(env="DATABASE_USERNAME")
    database_password: str = Field(env="DATABASE_PASSWORD")
    database_host: str = Field(env="DATABASE_HOST")
    database_port: str = Field(env="DATABASE_PORT")

    # JWT
    secret_key: str = Field(env="SECRET_KEY")
    algorithm: str = Field(env="ALGORITHM")
    access_token_expiry_time: int = Field(150, env="ACCESS_TOKEN_EXPIRE_TIME")
    refresh_token_expiry_time: int = Field(300, env="REFRESH_TOKEN_EXPIRE_TIME")

    # Redis
    redis_url: str = Field(env="REDIS_URL")

    # Celery
    celery_broker_url: str = Field(env="CELERY_BROKER_URL")
    celery_result_backend: str = Field(env="CELERY_RESULT_BACKEND")

    # Email
    smtp_server: str = Field(env="SMTP_SERVER")
    smtp_port: int = Field(env="SMTP_PORT")
    sender_email: str = Field(env="SENDER_EMAIL")
    sender_password: str = Field(env="SENDER_PASSWORD")

    # OpenAI
    openai_api_key: str = Field(env="OPENAI_API_KEY")
    embedding_model: str = Field(env="EMBEDDING_MODEL")

    # File paths - stored as strings, converted to Path when needed
    upload_dir: str = Field(env="UPLOAD_DIR")
    vector_store_dir: str = Field(env="VECTOR_STORE_DIR")

    # Embeddings config
    chunk_size: int = Field(env="CHUNK_SIZE")
    chunk_overlap: int = Field(env="CHUNK_OVERLAP")

    # Allowed Extension
    allowed_extensions: Set[str] = Field(
        default_factory=set,
        env="ALLOWED_EXTENSIONS",
    )

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
