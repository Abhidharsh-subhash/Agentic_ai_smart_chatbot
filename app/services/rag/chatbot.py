# app/services/rag/chatbot.py
"""
Chatbot service - matching standalone SmartChatbot EXACTLY.
Key: Let LangGraph's checkpointer handle message history persistence.
"""
import asyncio
import re
from datetime import datetime
from typing import Optional, List

from langchain_core.messages import HumanMessage, AIMessage

from .graph import create_agent
from .state import AgentState
from .config import InteractionMode, ScenarioStatus
from .memory import memory_manager


class ResponseSanitizer:
    """Sanitize responses to remove file references."""

    FILE_PATTERNS = [
        r"\b[\w\-]+\.(pdf|docx?|txt|xlsx?|pptx?|csv|json|xml|html?|md)\b",
        r"\(source:\s*[^)]+\)",
        r"source:\s*[\w\-\.]+",
        r"(?i)according to the [\w\s]+ document",
    ]

    @classmethod
    def sanitize(cls, response: str) -> str:
        if not response:
            return response
        sanitized = response
        for pattern in cls.FILE_PATTERNS:
            sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r"\s{2,}", " ", sanitized)
        sanitized = re.sub(r"\s+([.,!?])", r"\1", sanitized)
        return sanitized.strip()


class ChatbotService:
    """
    Chatbot that searches first, then disambiguates only when necessary.
    Matches standalone SmartChatbot behavior exactly.
    """

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or f"session_{datetime.now().timestamp()}"
        self.agent = create_agent()
        self.thread_id = self.session_id  # Use session_id as thread_id for checkpointer

        # Local context - matches standalone
        self.context = {}
        self.topic_history: List[str] = []
        self.conversation_history: List[dict] = []

        # Scenario tracking - matches standalone EXACTLY
        self.awaiting_scenario_selection = False
        self.current_scenario_options: List[str] = []
        self.original_query: Optional[str] = None
        self.search_results: Optional[str] = None

        # Redis memory for persistence across restarts
        self._memory = memory_manager.get_or_create(self.session_id)

        # Load state from Redis if exists
        self._load_state_from_redis()

        print(f"\n[ChatbotService] Initialized session: {self.session_id}")

    def _load_state_from_redis(self):
        """Load disambiguation state from Redis."""
        disambiguation = self._memory.get_disambiguation()
        if disambiguation and disambiguation.is_active:
            self.awaiting_scenario_selection = True
            self.current_scenario_options = disambiguation.current_options
            self.original_query = disambiguation.original_query
            self.search_results = disambiguation.search_results
            print(
                f"[ChatbotService] Loaded disambiguation state: options={self.current_scenario_options}"
            )

    def _save_state_to_redis(self):
        """Save disambiguation state to Redis."""
        if self.awaiting_scenario_selection and self.current_scenario_options:
            self._memory.set_disambiguation(
                original_query=self.original_query or "",
                options=self.current_scenario_options,
                search_results=self.search_results or "",
                question="",
            )
        else:
            self._memory.clear_disambiguation()

    def chat_sync(self, message: str) -> str:
        """
        Process a chat message - matches standalone chat() exactly.
        """
        print(f"\n{'='*60}")
        print(f"[ChatbotService] Processing: '{message}'")
        print(f"[ChatbotService] Session: {self.session_id}")
        print(
            f"[ChatbotService] Awaiting selection: {self.awaiting_scenario_selection}"
        )
        print(f"[ChatbotService] Options: {self.current_scenario_options}")
        print(f"{'='*60}")

        # Config for checkpointer - CRITICAL: same thread_id maintains conversation
        config = {"configurable": {"thread_id": self.thread_id}}

        # Build initial state - matches standalone EXACTLY
        initial_state: AgentState = {
            "messages": [HumanMessage(content=message)],
            "context": self.context,
            "clarification_needed": False,
            "clarification_reason": "",
            "follow_up_questions": [],
            "pending_clarification": False,
            "original_query": self.original_query or message,
            "clarification_attempts": 0,
            "user_intent": "",
            "detected_topics": [],
            "sentiment": "neutral",
            "interaction_mode": InteractionMode.QUERY.value,
            "conversation_history": self.conversation_history,
            "topic_history": self.topic_history,
            "search_confidence": 0.0,
            "search_quality": "",
            "has_searched": False,
            "search_results": self.search_results or "",
            "found_relevant_info": False,
            "best_match_score": float("inf"),
            "should_respond_not_found": False,
            "not_found_message": "",
            "has_multiple_scenarios": False,
            "detected_scenarios": [],
            "scenario_status": ScenarioStatus.SINGLE.value,
            "disambiguation_question": "",
            "selected_scenario": None,
            "disambiguation_depth": 0,
            "scenario_context": [],
            "awaiting_scenario_selection": self.awaiting_scenario_selection,
            "filtered_search_results": "",
            "current_scenario_options": self.current_scenario_options,
        }

        try:
            # Invoke agent
            result = self.agent.invoke(initial_state, config=config)

            # Update tracking state FROM result - matches standalone EXACTLY
            self.awaiting_scenario_selection = result.get(
                "awaiting_scenario_selection", False
            )
            self.current_scenario_options = result.get("current_scenario_options", [])

            if self.awaiting_scenario_selection:
                self.original_query = result.get("original_query", message)
                self.search_results = result.get("search_results", "")
            else:
                self.original_query = None
                self.search_results = None

            # Save state to Redis
            self._save_state_to_redis()

            # Get response - matches standalone
            for msg in reversed(result.get("messages", [])):
                if isinstance(msg, AIMessage) and msg.content:
                    response = ResponseSanitizer.sanitize(msg.content)

                    # Update conversation history
                    self.conversation_history.append(
                        {"role": "user", "content": message}
                    )
                    self.conversation_history.append(
                        {"role": "assistant", "content": response}
                    )

                    # Keep last 40 messages
                    self.conversation_history = self.conversation_history[-40:]

                    print(f"[ChatbotService] Response: {response[:100]}...")
                    return response

        except Exception as e:
            print(f"[ChatbotService] Error: {e}")
            import traceback

            traceback.print_exc()
            return "I encountered an issue. Please try again."

        return "I couldn't generate a response. Please try again."

    async def chat(self, message: str) -> str:
        """Async chat for WebSocket."""
        return await asyncio.to_thread(self.chat_sync, message)

    def reset(self):
        """Reset conversation state."""
        memory_manager.remove(self.session_id)
        self.context = {}
        self.topic_history = []
        self.conversation_history = []
        self.awaiting_scenario_selection = False
        self.current_scenario_options = []
        self.original_query = None
        self.search_results = None
        self.agent = create_agent()
        self._memory = memory_manager.get_or_create(self.session_id)


class SessionManager:
    """Manages chatbot sessions for WebSocket connections."""

    def __init__(self):
        self._sessions: dict[str, ChatbotService] = {}

    def get_or_create(self, session_id: str) -> ChatbotService:
        if session_id not in self._sessions:
            self._sessions[session_id] = ChatbotService(session_id)
        return self._sessions[session_id]

    def remove(self, session_id: str):
        if session_id in self._sessions:
            memory_manager.remove(session_id)
            del self._sessions[session_id]

    def clear_all(self):
        memory_manager.clear_all()
        self._sessions.clear()


session_manager = SessionManager()


def get_chatbot_service(session_id: str) -> ChatbotService:
    return session_manager.get_or_create(session_id)
