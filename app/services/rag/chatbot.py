import asyncio
from datetime import datetime
from typing import List, Optional
from langchain_core.messages import HumanMessage, AIMessage

from .graph import create_rag_graph
from .state import AgentState
from .config import InteractionMode
from .analyzers import ResponseSanitizer
from .memory import memory_manager


class ChatbotService:
    """Chatbot service with CoT and Memory."""

    def __init__(self, session_id: Optional[str] = None):
        self.agent = create_rag_graph()
        self.session_id = session_id or f"session_{datetime.now().timestamp()}"
        self.context = {
            "session_id": self.session_id,  # Important for memory
            "last_topic": None,
            "topics_discussed": [],
            "questions_asked": 0,
            "session_start": datetime.now().isoformat(),
        }
        self.topic_history: List[str] = []
        self.conversation_history: List[dict] = []
        self.pending_clarification = False
        self.original_query: Optional[str] = None
        self.clarification_attempts = 0

    def _extract_topics(self, message: str) -> List[str]:
        """Extract topics from message."""
        topic_keywords = {
            "authentication": ["login", "password", "auth", "sign in", "logout"],
            "user_management": ["user", "account", "profile", "permission", "role"],
            "configuration": ["setting", "config", "setup", "configure"],
            "billing": ["invoice", "payment", "billing", "charge", "price"],
            "booking": ["booking", "reservation", "travel", "trip"],
            "visa": ["visa", "application", "passport", "travel document"],
            "admin": ["admin", "administrator", "management", "dashboard"],
        }

        message_lower = message.lower()
        detected = []

        for topic, keywords in topic_keywords.items():
            if any(kw in message_lower for kw in keywords):
                detected.append(topic)

        return detected

    def _build_initial_state(self, message: str) -> AgentState:
        """Build initial state for the agent."""
        topics = self._extract_topics(message)

        if topics:
            self.context["last_topic"] = topics[0]
            self.topic_history.extend(topics)

        self.context["questions_asked"] += 1

        return {
            "messages": [HumanMessage(content=message)],
            "context": self.context,
            "clarification_needed": False,
            "clarification_reason": "",
            "follow_up_questions": [],
            "pending_clarification": self.pending_clarification,
            "original_query": self.original_query or message,
            "clarification_attempts": self.clarification_attempts,
            "user_intent": "",
            "detected_topics": topics,
            "sentiment": "neutral",
            "interaction_mode": InteractionMode.QUERY.value,
            "conversation_history": self.conversation_history,
            "topic_history": self.topic_history,
            "search_confidence": 0.0,
            "search_quality": "",
            "has_searched": False,
            "search_results": "",
            "found_relevant_info": False,
            "best_match_score": float("inf"),
            "should_respond_not_found": False,
            "not_found_message": "",
            # New CoT fields
            "thinking": None,
            "planned_search_queries": [],
            # New Memory fields
            "conversation_summary": "",
            "last_assistant_response": "",
            "last_user_query": "",
            "is_follow_up_question": False,
            "follow_up_context": "",
        }

    def _process_result(self, result: dict, message: str) -> str:
        """Process agent result and extract response."""
        self.pending_clarification = result.get("pending_clarification", False)
        if self.pending_clarification:
            self.original_query = result.get("original_query", message)
            self.clarification_attempts = result.get("clarification_attempts", 0)
        else:
            self.original_query = None
            self.clarification_attempts = 0

        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                response = msg.content

                is_not_found = result.get("should_respond_not_found", False)

                if not is_not_found:
                    if ResponseSanitizer.contains_general_knowledge(response):
                        response = (
                            "I don't have specific information about that in my knowledge base. "
                            "Could you try asking about a different topic?"
                        )
                    else:
                        response = ResponseSanitizer.sanitize(response)

                self.conversation_history.append({"role": "user", "content": message})
                self.conversation_history.append(
                    {"role": "assistant", "content": response}
                )

                return response

        return "I couldn't generate a response. Please try again."

    def chat_sync(self, message: str) -> str:
        """Synchronous chat method."""
        config = {"configurable": {"thread_id": self.session_id}}
        initial_state = self._build_initial_state(message)

        try:
            result = self.agent.invoke(initial_state, config=config)
            return self._process_result(result, message)
        except Exception as e:
            print(f"Error in chat: {e}")
            import traceback

            traceback.print_exc()
            return "I encountered an issue processing your request. Please try again."

    async def chat(self, message: str) -> str:
        """Async chat method for WebSocket."""
        return await asyncio.to_thread(self.chat_sync, message)

    def reset(self):
        """Reset conversation state."""
        # Clear memory for this session
        memory_manager.remove(self.session_id)

        self.context = {
            "session_id": self.session_id,
            "last_topic": None,
            "topics_discussed": [],
            "questions_asked": 0,
            "session_start": datetime.now().isoformat(),
        }
        self.topic_history = []
        self.conversation_history = []
        self.pending_clarification = False
        self.original_query = None
        self.clarification_attempts = 0

    def get_conversation_summary(self) -> dict:
        """Get summary of current conversation."""
        memory = memory_manager.get_or_create(self.session_id)
        return {
            "session_id": self.session_id,
            "turns": len(memory.turns),
            "topics_discussed": memory.key_topics,
            "questions_asked": self.context.get("questions_asked", 0),
        }


class SessionManager:
    """Manages chatbot sessions for WebSocket connections."""

    def __init__(self):
        self._sessions: dict[str, ChatbotService] = {}

    def get_or_create(self, session_id: str) -> ChatbotService:
        """Get existing session or create new one."""
        if session_id not in self._sessions:
            self._sessions[session_id] = ChatbotService(session_id)
        return self._sessions[session_id]

    def remove(self, session_id: str):
        """Remove a session."""
        if session_id in self._sessions:
            # Also remove memory
            memory_manager.remove(session_id)
            del self._sessions[session_id]

    def clear_all(self):
        """Clear all sessions."""
        memory_manager.clear_all()
        self._sessions.clear()


session_manager = SessionManager()


def get_chatbot_service(session_id: str) -> ChatbotService:
    """Get chatbot service for a session."""
    return session_manager.get_or_create(session_id)
