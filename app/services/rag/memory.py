from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class ConversationTurn:
    """Single turn in conversation."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    topics: List[str] = field(default_factory=list)


@dataclass
class ConversationMemory:
    """Manages conversation memory for a session."""

    session_id: str
    turns: List[ConversationTurn] = field(default_factory=list)
    summary: str = ""
    key_topics: List[str] = field(default_factory=list)
    max_turns: int = 20  # Keep last N turns

    def add_user_message(self, content: str, topics: List[str] = None):
        """Add user message to memory."""
        self.turns.append(
            ConversationTurn(role="user", content=content, topics=topics or [])
        )
        self._trim_history()
        self._update_topics(topics or [])

    def add_assistant_message(self, content: str, topics: List[str] = None):
        """Add assistant message to memory."""
        self.turns.append(
            ConversationTurn(role="assistant", content=content, topics=topics or [])
        )
        self._trim_history()

    def _trim_history(self):
        """Keep only the last max_turns."""
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns :]

    def _update_topics(self, new_topics: List[str]):
        """Update key topics discussed."""
        for topic in new_topics:
            if topic not in self.key_topics:
                self.key_topics.append(topic)
        # Keep last 10 topics
        self.key_topics = self.key_topics[-10:]

    def get_last_n_turns(self, n: int = 5) -> List[Dict]:
        """Get last N conversation turns."""
        return [
            {"role": turn.role, "content": turn.content} for turn in self.turns[-n:]
        ]

    def get_last_user_query(self) -> Optional[str]:
        """Get the last user query."""
        for turn in reversed(self.turns[:-1]):  # Exclude current
            if turn.role == "user":
                return turn.content
        return None

    def get_last_assistant_response(self) -> Optional[str]:
        """Get the last assistant response."""
        for turn in reversed(self.turns):
            if turn.role == "assistant":
                return turn.content
        return None

    def get_context_window(self, n_turns: int = 4) -> str:
        """Get formatted context window for LLM."""
        recent_turns = self.get_last_n_turns(n_turns)
        if not recent_turns:
            return "No previous conversation."

        context_parts = []
        for turn in recent_turns:
            role = "User" if turn["role"] == "user" else "Assistant"
            context_parts.append(f"{role}: {turn['content']}")

        return "\n".join(context_parts)

    def get_topics_discussed(self) -> str:
        """Get topics discussed in conversation."""
        if not self.key_topics:
            return "No specific topics yet."
        return ", ".join(self.key_topics)

    def is_likely_follow_up(self, current_query: str) -> bool:
        """Determine if current query is likely a follow-up."""
        if len(self.turns) < 2:
            return False

        current_lower = current_query.lower().strip()

        # Check for pronouns and references
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

        # Short queries are often follow-ups
        if len(current_lower.split()) <= 4:
            for indicator in follow_up_indicators:
                if indicator in current_lower:
                    return True

        # Check if query starts with follow-up patterns
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
        ]

        for start in follow_up_starts:
            if current_lower.startswith(start):
                return True

        return False

    def clear(self):
        """Clear conversation memory."""
        self.turns = []
        self.summary = ""
        self.key_topics = []


class MemoryManager:
    """Manages multiple conversation memories."""

    def __init__(self):
        self._memories: Dict[str, ConversationMemory] = {}

    def get_or_create(self, session_id: str) -> ConversationMemory:
        """Get existing memory or create new one."""
        if session_id not in self._memories:
            self._memories[session_id] = ConversationMemory(session_id=session_id)
        return self._memories[session_id]

    def remove(self, session_id: str):
        """Remove a session's memory."""
        if session_id in self._memories:
            del self._memories[session_id]

    def clear_all(self):
        """Clear all memories."""
        self._memories.clear()


# Global memory manager
memory_manager = MemoryManager()
