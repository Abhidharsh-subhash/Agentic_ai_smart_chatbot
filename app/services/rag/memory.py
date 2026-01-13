# app/services/rag/memory.py
"""
Simple Redis-backed state storage for disambiguation.
The LangGraph MemorySaver handles conversation history.
"""
import json
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime

from app.services.redis_client import get_redis
from .config import Config


@dataclass
class DisambiguationState:
    """Disambiguation state."""

    is_active: bool = False
    original_query: str = ""
    current_options: List[str] = field(default_factory=list)
    search_results: str = ""
    question: str = ""


class SessionMemory:
    """Redis-backed session memory for disambiguation state only."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._key = f"{Config.REDIS_SESSION_PREFIX}:{session_id}:disambiguation"
        self._redis = None
        self._fallback: Optional[dict] = None

    @property
    def redis(self):
        if self._redis is None:
            try:
                self._redis = get_redis()
            except Exception as e:
                print(f"⚠️ Redis unavailable: {e}")
        return self._redis

    def _use_fallback(self) -> bool:
        return self.redis is None

    def set_disambiguation(
        self,
        original_query: str,
        options: List[str],
        search_results: str,
        question: str,
    ):
        """Set disambiguation state."""
        data = {
            "is_active": True,
            "original_query": original_query,
            "current_options": options,
            "search_results": search_results,
            "question": question,
        }

        if self._use_fallback():
            self._fallback = data
            return

        try:
            self.redis.set(self._key, json.dumps(data), ex=Config.REDIS_TTL_SECONDS)
        except Exception as e:
            print(f"Redis write error: {e}")
            self._fallback = data

    def get_disambiguation(self) -> Optional[DisambiguationState]:
        """Get disambiguation state."""
        data = None

        if self._use_fallback():
            data = self._fallback
        else:
            try:
                raw = self.redis.get(self._key)
                if raw:
                    data = json.loads(raw)
            except Exception as e:
                print(f"Redis read error: {e}")
                data = self._fallback

        if data and data.get("is_active"):
            return DisambiguationState(**data)
        return None

    def clear_disambiguation(self):
        """Clear disambiguation state."""
        if self._use_fallback():
            self._fallback = None
            return

        try:
            self.redis.delete(self._key)
        except Exception as e:
            print(f"Redis delete error: {e}")

    def clear(self):
        """Clear all session data."""
        self.clear_disambiguation()


class MemoryManager:
    """Manages session memories."""

    def __init__(self):
        self._instances: Dict[str, SessionMemory] = {}

    def get_or_create(self, session_id: str) -> SessionMemory:
        if session_id not in self._instances:
            self._instances[session_id] = SessionMemory(session_id)
        return self._instances[session_id]

    def remove(self, session_id: str):
        if session_id in self._instances:
            self._instances[session_id].clear()
            del self._instances[session_id]

    def clear_all(self):
        try:
            redis_client = get_redis()
            cursor = 0
            while True:
                cursor, keys = redis_client.scan(
                    cursor, match=f"{Config.REDIS_SESSION_PREFIX}:*", count=100
                )
                if keys:
                    redis_client.delete(*keys)
                if cursor == 0:
                    break
        except Exception as e:
            print(f"Error clearing sessions: {e}")
        self._instances.clear()


memory_manager = MemoryManager()
