import asyncio
import json
from datetime import datetime
from typing import List, Optional
from langchain_core.messages import HumanMessage, AIMessage

from .graph import create_rag_graph
from .state import AgentState
from .config import InteractionMode, SupportMode
from .analyzers import ResponseSanitizer
from .memory import memory_manager


class ChatbotService:
    """Enhanced Chatbot service with scenario handling."""

    def __init__(self, session_id: Optional[str] = None):
        self.agent = create_rag_graph()
        self.session_id = session_id or f"session_{datetime.now().timestamp()}"
        self.context = {
            "session_id": self.session_id,
            "last_topic": None,
            "topics_discussed": [],
            "questions_asked": 0,
            "session_start": datetime.now().isoformat(),
        }
        self.topic_history: List[str] = []
        self.conversation_history: List[dict] = []

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

        # Get memory to check for pending clarification
        memory = memory_manager.get_or_create(self.session_id)
        has_pending = memory.has_pending_clarification()
        pending_scenarios = []
        pending_docs = []
        original_query = message

        if has_pending and memory.pending_clarification:
            pending_scenarios = memory.pending_clarification.scenarios
            pending_docs = memory.pending_clarification.raw_documents
            original_query = memory.pending_clarification.original_query

        return {
            "messages": [HumanMessage(content=message)],
            "context": self.context,
            "clarification_needed": False,
            "clarification_reason": "",
            "follow_up_questions": [],
            "pending_clarification": has_pending,
            "original_query": original_query,
            "clarification_attempts": 0,
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
            # Chain of Thought
            "thinking": None,
            "planned_search_queries": [],
            # Memory
            "conversation_summary": "",
            "last_assistant_response": "",
            "last_user_query": "",
            "is_follow_up_question": False,
            "follow_up_context": "",
            # Scenario handling
            "has_multiple_scenarios": False,
            "detected_scenarios": pending_scenarios,
            "awaiting_scenario_selection": has_pending,
            "selected_scenario_id": None,
            "scenario_question": "",
            "raw_search_documents": pending_docs,
            "support_mode": SupportMode.DIRECT_ANSWER.value,
            "clarification_context": {},
        }

    def _process_result(self, result: dict, message: str) -> str:
        """Process agent result and extract response."""
        # Check if awaiting scenario selection
        if result.get("awaiting_scenario_selection", False):
            # The response should be the clarification question
            pass

        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                response = msg.content

                is_not_found = result.get("should_respond_not_found", False)
                is_scenario_question = result.get("awaiting_scenario_selection", False)

                if not is_not_found and not is_scenario_question:
                    if ResponseSanitizer.contains_general_knowledge(response):
                        search_results = result.get("search_results", "")
                        if search_results:
                            response = self._extract_document_answer(
                                search_results, message
                            )
                    else:
                        response = ResponseSanitizer.sanitize(response)

                self.conversation_history.append({"role": "user", "content": message})
                self.conversation_history.append(
                    {"role": "assistant", "content": response}
                )

                return response

        return "I couldn't generate a response. Please try again."

    def _extract_document_answer(self, search_results: str, question: str) -> str:
        """Extract answer from search results."""
        try:
            results = (
                json.loads(search_results)
                if isinstance(search_results, str)
                else search_results
            )
            documents = results.get("documents", [])

            if documents:
                best_doc = documents[0].get("content", "")
                if best_doc:
                    return f"Based on the documentation: {best_doc}"

            return "I found related information but couldn't extract a clear answer."
        except:
            return "I found related information but couldn't extract a clear answer."

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

    def get_conversation_summary(self) -> dict:
        """Get summary of current conversation."""
        memory = memory_manager.get_or_create(self.session_id)
        return {
            "session_id": self.session_id,
            "turns": len(memory.turns),
            "topics_discussed": memory.key_topics,
            "questions_asked": self.context.get("questions_asked", 0),
            "has_pending_clarification": memory.has_pending_clarification(),
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
