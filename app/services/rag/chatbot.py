# app/services/rag/chatbot.py
"""
Chatbot service - matching standalone SmartChatbot exactly.
"""
import asyncio
from datetime import datetime
from typing import Optional

from langchain_core.messages import HumanMessage, AIMessage

from .graph import create_agent
from .state import AgentState
from .config import InteractionMode, ScenarioStatus
from .memory import memory_manager
from .nodes import ResponseSanitizer


class ChatbotService:
    """Chatbot that searches first, then disambiguates only when necessary."""

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or f"session_{datetime.now().timestamp()}"
        self.agent = create_agent()
        self._memory = memory_manager.get_or_create(self.session_id)

        print(f"\n[ChatbotService] Initialized session: {self.session_id}")

    def _build_initial_state(self, message: str) -> AgentState:
        """Build initial state for each invocation."""
        # Get disambiguation state from Redis
        disambiguation = self._memory.get_disambiguation()

        awaiting = disambiguation is not None and disambiguation.is_active
        options = disambiguation.current_options if awaiting else []
        original = disambiguation.original_query if awaiting else ""
        search_results = disambiguation.search_results if awaiting else ""

        return {
            "messages": [HumanMessage(content=message)],
            "context": {"session_id": self.session_id},
            "clarification_needed": False,
            "clarification_reason": "",
            "follow_up_questions": [],
            "pending_clarification": False,
            "original_query": original or message,
            "clarification_attempts": 0,
            "user_intent": "",
            "detected_topics": [],
            "sentiment": "neutral",
            "interaction_mode": InteractionMode.QUERY.value,
            "conversation_history": self._memory.get_conversation_history(),
            "topic_history": [],
            "search_confidence": 0.0,
            "search_quality": "",
            "has_searched": False,
            "search_results": search_results,
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
            "awaiting_scenario_selection": awaiting,
            "filtered_search_results": "",
            "current_scenario_options": options,
        }

    def chat_sync(self, message: str) -> str:
        """Process a chat message synchronously."""
        print(f"\n{'='*60}")
        print(f"[ChatbotService] Message: '{message}'")
        print(f"[ChatbotService] Session: {self.session_id}")
        print(f"{'='*60}")

        config = {"configurable": {"thread_id": self.session_id}}
        initial_state = self._build_initial_state(message)

        try:
            result = self.agent.invoke(initial_state, config=config)

            # Extract response
            for msg in reversed(result.get("messages", [])):
                if isinstance(msg, AIMessage) and msg.content:
                    response = ResponseSanitizer.sanitize(msg.content)

                    # Save to conversation history
                    self._memory.add_exchange(message, response)

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
        """Reset session."""
        memory_manager.remove(self.session_id)
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
