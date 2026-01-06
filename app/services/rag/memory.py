from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ConversationTurn:
    """Single turn in conversation."""

    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    topics: List[str] = field(default_factory=list)
    turn_type: str = "normal"  # normal, clarification, scenario_selection


@dataclass
class PendingClarification:
    """Tracks pending clarification requests."""

    original_query: str
    clarification_type: str  # "scenario", "ambiguous", "incomplete"
    scenarios: List[dict] = field(default_factory=list)
    attempts: int = 0
    raw_documents: List[dict] = field(default_factory=list)


@dataclass
class ConversationMemory:
    """Manages conversation memory for a session."""

    session_id: str
    turns: List[ConversationTurn] = field(default_factory=list)
    summary: str = ""
    key_topics: List[str] = field(default_factory=list)
    max_turns: int = 20

    # NEW: Pending clarification tracking
    pending_clarification: Optional[PendingClarification] = None

    # NEW: User preferences learned during conversation
    user_context: Dict = field(default_factory=dict)

    def add_user_message(
        self, content: str, topics: List[str] = None, turn_type: str = "normal"
    ):
        """Add user message to memory."""
        self.turns.append(
            ConversationTurn(
                role="user", content=content, topics=topics or [], turn_type=turn_type
            )
        )
        self._trim_history()
        self._update_topics(topics or [])

    def add_assistant_message(
        self, content: str, topics: List[str] = None, turn_type: str = "normal"
    ):
        """Add assistant message to memory."""
        self.turns.append(
            ConversationTurn(
                role="assistant",
                content=content,
                topics=topics or [],
                turn_type=turn_type,
            )
        )
        self._trim_history()

    def set_pending_clarification(
        self,
        original_query: str,
        clarification_type: str,
        scenarios: List[dict] = None,
        raw_documents: List[dict] = None,
    ):
        """Set a pending clarification request."""
        self.pending_clarification = PendingClarification(
            original_query=original_query,
            clarification_type=clarification_type,
            scenarios=scenarios or [],
            attempts=0,
            raw_documents=raw_documents or [],
        )

    def clear_pending_clarification(self):
        """Clear pending clarification."""
        self.pending_clarification = None

    def has_pending_clarification(self) -> bool:
        """Check if there's a pending clarification."""
        return self.pending_clarification is not None

    def increment_clarification_attempt(self):
        """Increment clarification attempt counter."""
        if self.pending_clarification:
            self.pending_clarification.attempts += 1

    def update_user_context(self, key: str, value: any):
        """Update user context with learned information."""
        self.user_context[key] = value

    def get_user_context(self, key: str, default=None):
        """Get user context value."""
        return self.user_context.get(key, default)

    def _trim_history(self):
        """Keep only the last max_turns."""
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns :]

    def _update_topics(self, new_topics: List[str]):
        """Update key topics discussed."""
        for topic in new_topics:
            if topic not in self.key_topics:
                self.key_topics.append(topic)
        self.key_topics = self.key_topics[-10:]

    def get_last_n_turns(self, n: int = 5) -> List[Dict]:
        """Get last N conversation turns."""
        return [
            {"role": turn.role, "content": turn.content, "type": turn.turn_type}
            for turn in self.turns[-n:]
        ]

    def get_last_user_query(self) -> Optional[str]:
        """Get the last user query."""
        for turn in reversed(self.turns[:-1]):
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

        follow_up_indicators = [
            "it",
            "this",
            "that",
            "these",
            "those",
            "they",
            "them",
            "more",
            "else",
            "another",
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
        self.pending_clarification = None
        self.user_context = {}


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


memory_manager = MemoryManager()
