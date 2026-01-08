import json
from typing import List, Dict, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from app.services.redis_client import get_redis
import redis

# ============================================
# Redis Key Configuration
# ============================================
KEY_PREFIX = "chat:session"
TTL_SECONDS = 86400 * 7  # 7 Days expiration


@dataclass
class ChatMessage:
    """Single Q&A pair in conversation."""

    id: int  # Message pair ID (1, 2, 3...)
    question: str
    answer: str
    question_timestamp: str
    answer_timestamp: str
    topics: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ChatMessage":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class SessionData:
    """Complete session data stored in Redis."""

    session_id: str
    chat_history: List[dict] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    summary: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    total_questions: int = 0
    status: str = "active"

    # Pending question
    pending_question: Optional[str] = None
    pending_question_timestamp: Optional[str] = None
    pending_topics: List[str] = field(default_factory=list)

    # NEW: Scenario clarification state
    awaiting_scenario_selection: bool = False
    pending_scenarios: List[dict] = field(default_factory=list)
    original_question_for_scenarios: str = ""
    full_response_for_scenarios: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> "SessionData":
        data = json.loads(json_str)
        return cls(**data)


class ConversationMemory:
    """Manages conversation memory for a session using Redis with Q&A pairs."""

    def __init__(self, session_id: str, max_pairs: int = 20):
        self.session_id = session_id
        self.max_pairs = max_pairs  # Max Q&A pairs to keep
        self._redis: Optional[redis.Redis] = None

        # Single Redis key for entire session
        self._key = f"{KEY_PREFIX}:{self.session_id}"

        # In-memory fallback
        self._fallback_data: Optional[SessionData] = None

    @property
    def redis(self) -> Optional[redis.Redis]:
        """Lazy Redis connection with fallback."""
        if self._redis is None:
            try:
                self._redis = get_redis()
            except Exception as e:
                print(f"⚠️ Redis unavailable for session {self.session_id}: {e}")
                self._redis = None
        return self._redis

    def _use_fallback(self) -> bool:
        """Check if we should use fallback in-memory storage."""
        return self.redis is None

    # ============================================
    # Core Data Access
    # ============================================

    def _get_session_data(self) -> SessionData:
        """Get session data from Redis or create new."""
        if self._use_fallback():
            if self._fallback_data is None:
                self._fallback_data = SessionData(session_id=self.session_id)
            return self._fallback_data

        try:
            raw_data = self.redis.get(self._key)
            if raw_data:
                return SessionData.from_json(raw_data)
            else:
                return SessionData(session_id=self.session_id)
        except Exception as e:
            print(f"Error reading session data from Redis: {e}")
            if self._fallback_data is None:
                self._fallback_data = SessionData(session_id=self.session_id)
            return self._fallback_data

    def _save_session_data(self, data: SessionData):
        """Save session data to Redis."""
        data.updated_at = datetime.now().isoformat()

        if self._use_fallback():
            self._fallback_data = data
            return

        try:
            self.redis.set(self._key, data.to_json(), ex=TTL_SECONDS)
        except Exception as e:
            print(f"Error saving session data to Redis: {e}")
            self._fallback_data = data

    # ============================================
    # Message Management (Q&A Pairs)
    # ============================================

    def add_user_message(self, content: str, topics: List[str] = None):
        """
        Add user question to memory.
        Stores as pending until answer is received.
        """
        data = self._get_session_data()

        # Store as pending question
        data.pending_question = content
        data.pending_question_timestamp = datetime.now().isoformat()
        data.pending_topics = topics or []

        # Update topics list
        self._update_topics_in_data(data, topics or [])

        self._save_session_data(data)

    def add_assistant_message(self, content: str, topics: List[str] = None):
        """
        Add assistant answer to memory.
        Pairs with pending question to create complete Q&A entry.
        """
        data = self._get_session_data()

        # Create Q&A pair if there's a pending question
        if data.pending_question:
            message_id = len(data.chat_history) + 1

            chat_message = ChatMessage(
                id=message_id,
                question=data.pending_question,
                answer=content,
                question_timestamp=data.pending_question_timestamp
                or datetime.now().isoformat(),
                answer_timestamp=datetime.now().isoformat(),
                topics=list(set(data.pending_topics + (topics or []))),
            )

            data.chat_history.append(chat_message.to_dict())
            data.total_questions = len(data.chat_history)

            # Clear pending question
            data.pending_question = None
            data.pending_question_timestamp = None
            data.pending_topics = []

            # Trim to max_pairs
            if len(data.chat_history) > self.max_pairs:
                data.chat_history = data.chat_history[-self.max_pairs :]
                # Re-number IDs
                for idx, msg in enumerate(data.chat_history, 1):
                    msg["id"] = idx

        self._save_session_data(data)

    def _update_topics_in_data(self, data: SessionData, new_topics: List[str]):
        """Update key topics in session data."""
        if not new_topics:
            return

        for topic in new_topics:
            if topic and topic not in data.topics:
                data.topics.append(topic)

        # Keep last 10 topics
        if len(data.topics) > 10:
            data.topics = data.topics[-10:]

    # ============================================
    # Chat History Retrieval
    # ============================================

    def get_chat_history(self) -> List[Dict]:
        """
        Get complete chat history as Q&A pairs.

        Returns:
            List of dicts with structure:
            [
                {
                    "id": 1,
                    "question": "What is visa?",
                    "answer": "A visa is...",
                    "question_timestamp": "2024-01-15T10:30:00",
                    "answer_timestamp": "2024-01-15T10:30:05",
                    "topics": ["visa"]
                },
                ...
            ]
        """
        data = self._get_session_data()
        return data.chat_history

    def get_last_n_exchanges(self, n: int = 5) -> List[Dict]:
        """Get last N Q&A exchanges."""
        data = self._get_session_data()
        return data.chat_history[-n:]

    def get_last_exchange(self) -> Optional[Dict]:
        """Get the most recent Q&A exchange."""
        data = self._get_session_data()
        if data.chat_history:
            return data.chat_history[-1]
        return None

    def get_last_user_query(self) -> Optional[str]:
        """Get the last user question."""
        last = self.get_last_exchange()
        if last:
            return last.get("question")
        return None

    def get_last_assistant_response(self) -> Optional[str]:
        """Get the last assistant answer."""
        last = self.get_last_exchange()
        if last:
            return last.get("answer")
        return None

    # ============================================
    # Context for LLM
    # ============================================

    def get_context_window(self, n_exchanges: int = 4) -> str:
        """Get formatted context window for LLM."""
        recent = self.get_last_n_exchanges(n_exchanges)
        if not recent:
            return "No previous conversation."

        context_parts = []
        for exchange in recent:
            context_parts.append(f"User: {exchange['question']}")
            context_parts.append(f"Assistant: {exchange['answer']}")

        return "\n".join(context_parts)

    def get_last_n_turns(self, n: int = 5) -> List[Dict]:
        """
        Get last N turns in alternating format (for compatibility).
        Converts Q&A pairs back to individual turns.
        """
        recent = self.get_last_n_exchanges((n + 1) // 2)  # Each exchange = 2 turns

        turns = []
        for exchange in recent:
            turns.append({"role": "user", "content": exchange["question"]})
            turns.append({"role": "assistant", "content": exchange["answer"]})

        return turns[-n:]  # Return exactly n turns

    # ============================================
    # Properties
    # ============================================

    @property
    def turns(self) -> List[Dict]:
        """Get all turns (compatibility property)."""
        return self.get_last_n_turns(self.max_pairs * 2)

    @property
    def key_topics(self) -> List[str]:
        """Get topics discussed."""
        return self._get_session_data().topics

    @key_topics.setter
    def key_topics(self, topics: List[str]):
        """Set topics."""
        data = self._get_session_data()
        data.topics = topics
        self._save_session_data(data)

    @property
    def summary(self) -> str:
        """Get conversation summary."""
        return self._get_session_data().summary

    @summary.setter
    def summary(self, value: str):
        """Set conversation summary."""
        data = self._get_session_data()
        data.summary = value
        self._save_session_data(data)

    def get_topics_discussed(self) -> str:
        """Get topics discussed in conversation."""
        topics = self.key_topics
        if not topics:
            return "No specific topics yet."
        return ", ".join(topics)

    def get_turn_count(self) -> int:
        """Get total number of Q&A exchanges."""
        return len(self._get_session_data().chat_history)

    # ============================================
    # Follow-up Detection
    # ============================================

    def is_likely_follow_up(self, current_query: str) -> bool:
        """Determine if current query is likely a follow-up."""
        if self.get_turn_count() < 1:
            return False

        current_lower = current_query.lower().strip()

        follow_up_indicators = [
            "it",
            "this",
            "that",
            "these",
            "those",
            "they",
            "them",
            "their",
            "more",
            "else",
            "another",
            "other",
            "also",
            "and",
            "but",
            "what about",
            "how about",
            "can you",
            "could you",
            "tell me more",
            "explain more",
            "why",
            "how",
            "when",
            "where",
            "the same",
            "similar",
            "previous",
            "earlier",
            "before",
            "you mentioned",
            "you said",
        ]

        if len(current_lower.split()) <= 4:
            for indicator in follow_up_indicators:
                if indicator in current_lower:
                    return True

        follow_up_starts = [
            "what about",
            "how about",
            "and ",
            "but ",
            "also ",
            "can you",
            "could you",
            "what if",
            "why ",
            "how ",
            "is it",
            "are they",
            "does it",
            "do they",
        ]

        for start in follow_up_starts:
            if current_lower.startswith(start):
                return True

        return False

    # ============================================
    # Metadata
    # ============================================

    def set_metadata(self, key: str, value: str):
        """Set session metadata."""
        data = self._get_session_data()
        if key == "questions_asked":
            data.total_questions = int(value)
        elif key == "status":
            data.status = value
        self._save_session_data(data)

    def get_metadata(self, key: str) -> Optional[str]:
        """Get session metadata."""
        data = self._get_session_data()
        if key == "questions_asked":
            return str(data.total_questions)
        elif key == "status":
            return data.status
        elif key == "created_at":
            return data.created_at
        return None

    # ============================================
    # Session Lifecycle
    # ============================================

    def initialize_session(self) -> bool:
        """Initialize session in Redis."""
        try:
            data = SessionData(session_id=self.session_id, status="connected")
            self._save_session_data(data)
            return True
        except Exception as e:
            print(f"Error initializing session: {e}")
            return False

    def mark_disconnected(self):
        """Mark session as disconnected."""
        data = self._get_session_data()
        data.status = "disconnected"
        self._save_session_data(data)

    def clear(self):
        """Clear conversation memory from Redis."""
        if self._use_fallback():
            self._fallback_data = None
            return

        try:
            self.redis.delete(self._key)
        except Exception as e:
            print(f"Error clearing memory from Redis: {e}")

    def refresh_ttl(self):
        """Refresh TTL on session key."""
        if self._use_fallback():
            return

        try:
            self.redis.expire(self._key, TTL_SECONDS)
        except Exception as e:
            print(f"Error refreshing TTL: {e}")

    def get_full_session(self) -> dict:
        """Get complete session data as dictionary."""
        data = self._get_session_data()
        return asdict(data)

    def set_pending_scenarios(
        self, scenarios: List[Dict], original_question: str, full_response: str
    ):
        """Store pending scenarios for clarification flow."""
        data = self._get_session_data()

        # Store in a new field (add to SessionData if needed)
        scenario_state = {
            "scenarios": scenarios,
            "original_question": original_question,
            "full_response": full_response,
            "awaiting_selection": True,
            "timestamp": datetime.now().isoformat(),
        }

        # Use metadata storage
        if self._use_fallback():
            if not hasattr(self, "_scenario_state"):
                self._scenario_state = {}
            self._scenario_state[self.session_id] = scenario_state
        else:
            try:
                scenario_key = f"{self._key}:scenarios"
                self.redis.set(scenario_key, json.dumps(scenario_state), ex=TTL_SECONDS)
            except Exception as e:
                print(f"Error storing scenario state: {e}")

    def get_pending_scenarios(self) -> Optional[Dict]:
        """Retrieve pending scenarios if any."""
        if self._use_fallback():
            return getattr(self, "_scenario_state", {}).get(self.session_id)

        try:
            scenario_key = f"{self._key}:scenarios"
            raw = self.redis.get(scenario_key)
            if raw:
                return json.loads(raw)
        except Exception as e:
            print(f"Error retrieving scenario state: {e}")

        return None

    def clear_pending_scenarios(self):
        """Clear pending scenarios after resolution."""
        if self._use_fallback():
            if hasattr(self, "_scenario_state"):
                self._scenario_state.pop(self.session_id, None)
        else:
            try:
                scenario_key = f"{self._key}:scenarios"
                self.redis.delete(scenario_key)
            except Exception as e:
                print(f"Error clearing scenario state: {e}")


class MemoryManager:
    """Manages multiple conversation memories."""

    def __init__(self):
        self._instances: Dict[str, ConversationMemory] = {}

    def get_or_create(self, session_id: str) -> ConversationMemory:
        """Get memory interface for a session."""
        if session_id not in self._instances:
            self._instances[session_id] = ConversationMemory(session_id=session_id)
        return self._instances[session_id]

    def remove(self, session_id: str):
        """Remove a session's memory."""
        memory = ConversationMemory(session_id=session_id)
        memory.clear()

        if session_id in self._instances:
            del self._instances[session_id]

    def clear_all(self):
        """Clear all memories."""
        try:
            redis_client = get_redis()
            cursor = 0
            while True:
                cursor, keys = redis_client.scan(
                    cursor, match=f"{KEY_PREFIX}:*", count=100
                )
                if keys:
                    redis_client.delete(*keys)
                if cursor == 0:
                    break
            print("✅ Cleared all conversation memories from Redis")
        except Exception as e:
            print(f"Error clearing all memories: {e}")

        self._instances.clear()

    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs."""
        try:
            redis_client = get_redis()
            sessions = []
            cursor = 0
            while True:
                cursor, keys = redis_client.scan(
                    cursor, match=f"{KEY_PREFIX}:*", count=100
                )
                for key in keys:
                    session_id = key.replace(f"{KEY_PREFIX}:", "")
                    sessions.append(session_id)
                if cursor == 0:
                    break
            return sessions
        except Exception as e:
            print(f"Error getting active sessions: {e}")
            return list(self._instances.keys())


# Global memory manager
memory_manager = MemoryManager()
