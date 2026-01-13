# app/services/rag/memory.py
"""
Redis-backed session state for multi-user WebSocket support.
Stores disambiguation state and conversation history.
"""
import json
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime

from app.services.redis_client import get_redis
from .config import Config


@dataclass
class DisambiguationState:
    """Disambiguation state stored in Redis."""

    is_active: bool = False
    original_query: str = ""
    current_options: List[str] = field(default_factory=list)
    search_results: str = ""
    asked_question: str = ""


@dataclass
class SessionData:
    """Session data stored in Redis."""

    session_id: str
    conversation_history: List[dict] = field(default_factory=list)
    topic_history: List[str] = field(default_factory=list)
    disambiguation: Optional[dict] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> "SessionData":
        return cls(**json.loads(json_str))


class SessionMemory:
    """Redis-backed session memory."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._key = f"{Config.REDIS_SESSION_PREFIX}:{session_id}"
        self._redis = None
        self._fallback: Optional[SessionData] = None

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

    def _get_data(self) -> SessionData:
        if self._use_fallback():
            if self._fallback is None:
                self._fallback = SessionData(session_id=self.session_id)
            return self._fallback

        try:
            raw = self.redis.get(self._key)
            if raw:
                return SessionData.from_json(raw)
            return SessionData(session_id=self.session_id)
        except Exception as e:
            print(f"Redis read error: {e}")
            if self._fallback is None:
                self._fallback = SessionData(session_id=self.session_id)
            return self._fallback

    def _save_data(self, data: SessionData):
        data.updated_at = datetime.now().isoformat()
        if self._use_fallback():
            self._fallback = data
            return
        try:
            self.redis.set(self._key, data.to_json(), ex=Config.REDIS_TTL_SECONDS)
        except Exception as e:
            print(f"Redis write error: {e}")
            self._fallback = data

    # ===== Disambiguation State =====

    def set_disambiguation(
        self,
        original_query: str,
        options: List[str],
        search_results: str,
        question: str,
    ):
        """Set active disambiguation state."""
        data = self._get_data()
        data.disambiguation = {
            "is_active": True,
            "original_query": original_query,
            "current_options": options,
            "search_results": search_results,
            "asked_question": question,
        }
        self._save_data(data)
        print(f"[Memory] Set disambiguation: options={options}")

    def get_disambiguation(self) -> Optional[DisambiguationState]:
        """Get active disambiguation state."""
        data = self._get_data()
        if data.disambiguation and data.disambiguation.get("is_active"):
            return DisambiguationState(**data.disambiguation)
        return None

    def clear_disambiguation(self):
        """Clear disambiguation state."""
        data = self._get_data()
        if data.disambiguation:
            print(f"[Memory] Clearing disambiguation")
            data.disambiguation = None
            self._save_data(data)

    def is_awaiting_disambiguation(self) -> bool:
        state = self.get_disambiguation()
        return state is not None and state.is_active

    # ===== Conversation History =====

    def add_exchange(self, user_msg: str, assistant_msg: str):
        """Add a conversation exchange."""
        data = self._get_data()
        data.conversation_history.append(
            {
                "role": "user",
                "content": user_msg,
            }
        )
        data.conversation_history.append(
            {
                "role": "assistant",
                "content": assistant_msg,
            }
        )
        # Keep last 40 messages (20 exchanges)
        data.conversation_history = data.conversation_history[-40:]
        self._save_data(data)

    def get_conversation_history(self) -> List[dict]:
        return self._get_data().conversation_history

    # ===== Cleanup =====

    def clear(self):
        if self._use_fallback():
            self._fallback = None
            return
        try:
            self.redis.delete(self._key)
        except Exception as e:
            print(f"Redis delete error: {e}")


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
