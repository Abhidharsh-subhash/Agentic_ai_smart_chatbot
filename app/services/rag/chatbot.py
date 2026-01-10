# app/services/rag/chatbot.py

import asyncio
import json
from datetime import datetime
from typing import List, Optional, Dict
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

from .graph import create_rag_graph
from .state import AgentState
from .config import InteractionMode
from .analyzers import ResponseSanitizer
from .memory import memory_manager
from app.core.config import settings


class NewQuestionDetector:
    """Detects if user input is a new unrelated question vs a clarification response."""

    DETECTION_PROMPT = """Determine if the user's message is a NEW UNRELATED QUESTION or a RESPONSE to the ongoing clarification.

CONVERSATION CONTEXT:
- Original Topic: {original_topic}
- We were asking clarification questions about: {clarification_context}
- Last question we asked: "{last_question}"

USER'S NEW MESSAGE: "{user_message}"

RULES:
1. NEW QUESTION indicators:
   - Completely different topic
   - General inquiry questions: "how can I...", "what is...", "how do I..."
   - Questions that make complete sense without any previous context

2. CLARIFICATION RESPONSE indicators:
   - Direct answers: "yes", "no", "correct", "that's right"
   - Describing the specific error/situation being discussed
   - Short responses that answer the question asked

Respond with ONLY valid JSON (no markdown):
{{"is_new_question": true/false, "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""

    def __init__(self):
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=getattr(settings, "openai_model", "gpt-4o"),
                temperature=0.0,
                openai_api_key=settings.openai_api_key,
            )
        return self._llm

    def is_new_question(
        self,
        user_message: str,
        original_topic: str,
        last_question: str,
        clarification_context: str = "",
    ) -> Dict:
        from langchain_core.messages import SystemMessage

        user_lower = user_message.lower().strip()

        # ===== FAST PATH: Obvious clarification responses =====
        obvious_responses = {
            "yes",
            "no",
            "nope",
            "yep",
            "yeah",
            "yup",
            "correct",
            "that's right",
            "not that",
            "neither",
            "none",
            "n",
            "y",
            "ok",
            "okay",
            "right",
            "exactly",
            "affirmative",
            "negative",
        }

        if user_lower in obvious_responses:
            print(f"[NewQuestionDetector] FAST: Obvious clarification response")
            return {
                "is_new_question": False,
                "confidence": 0.95,
                "reasoning": "Obvious clarification response",
            }

        # ===== FAST PATH: Obvious new question patterns =====
        new_question_patterns = [
            ("how can i ", 0.9),
            ("how do i ", 0.9),
            ("how to ", 0.85),
            ("what is ", 0.85),
            ("what are ", 0.85),
            ("can you tell me ", 0.9),
            ("i want to know ", 0.85),
            ("tell me about ", 0.85),
        ]

        for pattern, base_confidence in new_question_patterns:
            if user_lower.startswith(pattern):
                print(
                    f"[NewQuestionDetector] Pattern match: '{pattern}' → NEW QUESTION"
                )
                return {
                    "is_new_question": True,
                    "confidence": base_confidence,
                    "reasoning": f"New question pattern '{pattern}'",
                }

        # Default: treat as clarification response
        return {
            "is_new_question": False,
            "confidence": 0.6,
            "reasoning": "Default to clarification",
        }


new_question_detector = NewQuestionDetector()


class ChatbotService:
    """Chatbot service with CoT and Redis-backed Memory."""

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

        # ===== EXPLICIT CLARIFICATION STATE TRACKING =====
        # This is the source of truth for clarification flow
        self._clarification_state: Optional[Dict] = None
        self._awaiting_scenario_selection: bool = False
        self._original_query: str = ""

        self._memory = memory_manager.get_or_create(self.session_id)

        stored_count = self._memory.get_metadata("questions_asked")
        if stored_count:
            try:
                self.context["questions_asked"] = int(stored_count)
            except ValueError:
                pass

    def _extract_topics(self, message: str) -> List[str]:
        """Extract topics from message."""
        topic_keywords = {
            "visa": ["visa", "application", "passport", "travel document"],
            "booking": ["booking", "reservation", "travel", "trip"],
            "billing": ["invoice", "payment", "billing", "charge", "price"],
            "admin": ["admin", "administrator", "management", "dashboard"],
        }

        message_lower = message.lower()
        detected = []

        for topic, keywords in topic_keywords.items():
            if any(kw in message_lower for kw in keywords):
                detected.append(topic)

        return detected

    def _is_new_question(self, message: str) -> bool:
        """Check if this is a new question while in clarification flow."""
        if not self._awaiting_scenario_selection:
            return False

        if not self._clarification_state:
            return False

        print(f"\n{'='*60}")
        print(f"[ChatbotService] Checking if new question: '{message}'")
        print(f"[ChatbotService] Original query: '{self._original_query}'")
        print(f"{'='*60}")

        result = new_question_detector.is_new_question(
            user_message=message,
            original_topic=self._original_query,
            last_question=self._clarification_state.get("current_question", ""),
            clarification_context=self._clarification_state.get("gathered_context", ""),
        )

        is_new = result.get("is_new_question", False)
        confidence = result.get("confidence", 0.5)

        print(f"[ChatbotService] Result: is_new={is_new}, confidence={confidence}")

        return is_new and confidence >= 0.7

    def _reset_clarification_state(self):
        """Reset clarification state."""
        print(f"[ChatbotService] ⚠️ RESETTING CLARIFICATION STATE")
        self._clarification_state = None
        self._awaiting_scenario_selection = False
        self._original_query = ""

    def _build_initial_state(
        self, message: str, include_clarification: bool = True
    ) -> AgentState:
        """Build state for graph invocation."""
        topics = self._extract_topics(message)

        if topics:
            self.context["last_topic"] = topics[0]
            for t in topics:
                if t not in self.context["topics_discussed"]:
                    self.context["topics_discussed"].append(t)

        self.context["questions_asked"] += 1
        self._memory.set_metadata(
            "questions_asked", str(self.context["questions_asked"])
        )
        self._memory.refresh_ttl()

        # Build base state
        state = {
            "messages": [HumanMessage(content=message)],
            "context": self.context,
            "clarification_needed": False,
            "clarification_reason": "",
            "follow_up_questions": [],
            "pending_clarification": False,
            "original_query": self._original_query or message,
            "clarification_attempts": 0,
            "user_intent": "",
            "detected_topics": topics,
            "sentiment": "neutral",
            "interaction_mode": InteractionMode.QUERY.value,
            "conversation_history": [],
            "topic_history": [],
            "search_confidence": 0.0,
            "search_quality": "",
            "has_searched": False,
            "search_results": "",
            "found_relevant_info": False,
            "best_match_score": float("inf"),
            "should_respond_not_found": False,
            "not_found_message": "",
            "thinking": None,
            "planned_search_queries": [],
            "conversation_summary": "",
            "last_assistant_response": "",
            "last_user_query": "",
            "is_follow_up_question": False,
            "follow_up_context": "",
            "response_analysis": None,
            "needs_scenario_clarification": False,
            "scenario_clarification_question": "",
            "scenario_options": [],
            "original_full_response": "",
            "parsed_scenarios": [],
            "selected_scenario_id": None,
            "user_scenario_context": "",
        }

        # ===== INCLUDE CLARIFICATION STATE IF ACTIVE =====
        if (
            include_clarification
            and self._awaiting_scenario_selection
            and self._clarification_state
        ):
            print(f"[ChatbotService] Including clarification state in initial state")
            state["awaiting_scenario_selection"] = True
            state["scenario_clarification_pending"] = True
            state["clarification_state"] = self._clarification_state
            state["user_scenario_context"] = message
        else:
            state["awaiting_scenario_selection"] = False
            state["scenario_clarification_pending"] = False
            state["clarification_state"] = None

        return state

    def _update_from_result(self, result: dict, message: str):
        """Update local state from graph result."""
        # Check for new clarification flow
        if result.get("awaiting_scenario_selection", False):
            new_clarification_state = result.get("clarification_state")
            if new_clarification_state and new_clarification_state.get(
                "is_active", False
            ):
                print(f"[ChatbotService] ✅ Clarification flow started/continuing")
                self._awaiting_scenario_selection = True
                self._clarification_state = new_clarification_state
                self._original_query = new_clarification_state.get(
                    "original_question", message
                )
                return

        # Check if clarification ended
        clarification_state = result.get("clarification_state")
        if clarification_state is None or not clarification_state.get(
            "is_active", False
        ):
            if self._awaiting_scenario_selection:
                print(f"[ChatbotService] Clarification flow ended")
            self._reset_clarification_state()

    def _extract_response(self, result: dict) -> str:
        """Extract response from result."""
        for msg in reversed(result.get("messages", [])):
            if isinstance(msg, AIMessage) and msg.content:
                response = msg.content
                is_not_found = result.get("should_respond_not_found", False)

                if not is_not_found:
                    response = ResponseSanitizer.sanitize(response)

                return response

        return "I couldn't generate a response. Please try again."

    def chat_sync(self, message: str) -> str:
        """Synchronous chat method."""

        # ===== CHECK FOR NEW QUESTION WHILE IN CLARIFICATION =====
        if self._is_new_question(message):
            print(f"[ChatbotService] 🔄 NEW QUESTION DETECTED - Resetting")
            self._reset_clarification_state()
            # Create new graph to clear any stale state
            self.agent = create_rag_graph()

        # Use consistent thread_id for this session
        config = {"configurable": {"thread_id": self.session_id}}

        # Build state - include clarification if active
        initial_state = self._build_initial_state(
            message, include_clarification=self._awaiting_scenario_selection
        )

        print(f"\n[ChatbotService] Invoking graph with:")
        print(
            f"  - awaiting_scenario_selection: {initial_state.get('awaiting_scenario_selection')}"
        )
        print(
            f"  - clarification_state active: {initial_state.get('clarification_state') is not None}"
        )

        try:
            result = self.agent.invoke(initial_state, config=config)

            # Update local state from result
            self._update_from_result(result, message)

            return self._extract_response(result)

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
        self._reset_clarification_state()
        self.context = {
            "session_id": self.session_id,
            "last_topic": None,
            "topics_discussed": [],
            "questions_asked": 0,
            "session_start": datetime.now().isoformat(),
        }
        self.agent = create_rag_graph()
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
