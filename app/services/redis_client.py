import redis
from redis import asyncio as aioredis
from typing import Optional
from app.core.config import settings


class RedisClient:
    """Singleton Redis client with connection pooling (sync + async)."""

    # Sync client
    _instance: Optional[redis.Redis] = None
    _pool: Optional[redis.ConnectionPool] = None

    # Async client
    _async_instance: Optional[aioredis.Redis] = None

    @classmethod
    def get_client(cls) -> redis.Redis:
        """Get or create sync Redis client instance."""
        if cls._instance is None:
            redis_url = settings.redis_url

            cls._pool = redis.ConnectionPool.from_url(
                redis_url,
                decode_responses=True,
                max_connections=20,
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True,
            )

            cls._instance = redis.Redis(connection_pool=cls._pool)

            try:
                cls._instance.ping()
                print(f"✅ Redis connected successfully: {redis_url}")
            except redis.ConnectionError as e:
                print(f"⚠️ Redis connection failed: {e}")
                cls._instance = None
                cls._pool = None
                raise

        return cls._instance

    @classmethod
    async def get_async_client(cls) -> aioredis.Redis:
        """Get or create async Redis client instance."""
        if cls._async_instance is None:
            redis_url = settings.redis_url

            cls._async_instance = await aioredis.from_url(
                redis_url,
                decode_responses=True,
                max_connections=20,
                socket_timeout=5,
                socket_connect_timeout=5,
            )

            try:
                await cls._async_instance.ping()
                print(f"✅ Async Redis connected: {redis_url}")
            except Exception as e:
                print(f"⚠️ Async Redis connection failed: {e}")
                cls._async_instance = None
                raise

        return cls._async_instance

    @classmethod
    def close(cls):
        """Close sync Redis connection."""
        if cls._pool:
            cls._pool.disconnect()
            cls._instance = None
            cls._pool = None

    @classmethod
    async def close_async(cls):
        """Close async Redis connection."""
        if cls._async_instance:
            await cls._async_instance.close()
            cls._async_instance = None

    @classmethod
    def is_connected(cls) -> bool:
        """Check if Redis is connected."""
        if cls._instance is None:
            return False
        try:
            cls._instance.ping()
            return True
        except (redis.ConnectionError, redis.TimeoutError):
            return False


def get_redis() -> redis.Redis:
    """Helper function to get sync Redis client."""
    return RedisClient.get_client()


async def get_async_redis() -> aioredis.Redis:
    """Helper function to get async Redis client."""
    return await RedisClient.get_async_client()
