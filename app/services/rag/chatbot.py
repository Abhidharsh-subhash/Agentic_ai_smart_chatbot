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
    """
    Detects if user input is a new unrelated question vs a clarification response.
    Used BEFORE the graph to reset state if needed.
    """

    DETECTION_PROMPT = """Determine if the user's message is a NEW UNRELATED QUESTION or a RESPONSE to the ongoing clarification.

CONVERSATION CONTEXT:
- Original Topic: {original_topic}
- We were asking clarification questions about: {clarification_context}
- Last question we asked: "{last_question}"

USER'S NEW MESSAGE: "{user_message}"

RULES:
1. NEW QUESTION indicators:
   - Completely different topic (e.g., "how can I apply for visa" when discussing application errors)
   - General inquiry questions: "how can I...", "what is...", "how do I...", "can you tell me about..."
   - Questions that make complete sense without any previous context
   - Topic change from troubleshooting/errors to general procedures

2. CLARIFICATION RESPONSE indicators:
   - Direct answers: "yes", "no", "correct", "that's right"
   - Describing the specific error/situation being discussed
   - Short responses that answer the question asked
   - References to the ongoing issue: "it shows...", "the error is..."

IMPORTANT: "{user_message}" compared to original topic "{original_topic}"
- If topics are DIFFERENT → is_new_question = true
- If user is answering/elaborating on the SAME topic → is_new_question = false

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
        """
        Determine if the user is asking a new question.

        Returns:
            Dict with is_new_question, confidence, reasoning
        """
        from langchain_core.messages import SystemMessage

        user_lower = user_message.lower().strip()

        print(f"\n{'='*60}")
        print(f"[NewQuestionDetector] Analyzing: '{user_message}'")
        print(f"[NewQuestionDetector] Original topic: '{original_topic}'")
        print(f"[NewQuestionDetector] Last question: '{last_question}'")
        print(f"{'='*60}")

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
            ("can you explain ", 0.9),
            ("i want to know ", 0.85),
            ("tell me about ", 0.85),
            ("i need to ", 0.8),
            ("i want to ", 0.8),
            ("where can i ", 0.85),
            ("when can i ", 0.85),
            ("can i ", 0.7),
        ]

        for pattern, base_confidence in new_question_patterns:
            if user_lower.startswith(pattern):
                # Check topic similarity
                topic_similarity = self._check_topic_similarity(
                    user_message, original_topic
                )

                print(f"[NewQuestionDetector] Pattern match: '{pattern}'")
                print(f"[NewQuestionDetector] Topic similarity: {topic_similarity}")

                # If topics are different, it's definitely a new question
                if topic_similarity < 0.3:
                    print(
                        f"[NewQuestionDetector] ✅ NEW QUESTION (low topic similarity)"
                    )
                    return {
                        "is_new_question": True,
                        "confidence": base_confidence,
                        "reasoning": f"New question pattern '{pattern}' with different topic",
                    }
                elif topic_similarity > 0.6:
                    # High similarity - might be follow-up about same topic
                    print(f"[NewQuestionDetector] Same topic follow-up")
                    return {
                        "is_new_question": False,
                        "confidence": 0.7,
                        "reasoning": "Follow-up question on same topic",
                    }

        # ===== LLM PATH: Ambiguous cases =====
        print(f"[NewQuestionDetector] Using LLM for ambiguous case...")

        try:
            prompt = self.DETECTION_PROMPT.format(
                original_topic=original_topic,
                clarification_context=clarification_context or original_topic,
                last_question=last_question,
                user_message=user_message,
            )

            result = self.llm.invoke([SystemMessage(content=prompt)])
            response_text = result.content.strip()

            print(f"[NewQuestionDetector] LLM response: {response_text[:200]}")

            analysis = self._parse_json(response_text)

            is_new = analysis.get("is_new_question", False)
            confidence = analysis.get("confidence", 0.5)

            print(
                f"[NewQuestionDetector] LLM result: is_new={is_new}, confidence={confidence}"
            )

            return {
                "is_new_question": is_new,
                "confidence": confidence,
                "reasoning": analysis.get("reasoning", "LLM analysis"),
            }

        except Exception as e:
            print(f"[NewQuestionDetector] LLM Error: {e}")
            # Fallback: treat long questions with ? as new questions
            if "?" in user_message and len(user_message.split()) > 5:
                return {
                    "is_new_question": True,
                    "confidence": 0.6,
                    "reasoning": "Fallback - long question",
                }
            return {
                "is_new_question": False,
                "confidence": 0.5,
                "reasoning": "Fallback",
            }

    def _check_topic_similarity(self, message1: str, message2: str) -> float:
        """Check keyword overlap between two messages."""
        # Common words to ignore
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "i",
            "my",
            "for",
            "to",
            "can",
            "how",
            "what",
            "do",
            "you",
            "me",
            "it",
            "this",
            "that",
            "have",
            "has",
            "be",
            "been",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "about",
            "with",
        }

        def extract_keywords(text):
            words = set(text.lower().split())
            # Remove punctuation
            words = {w.strip(".,?!") for w in words}
            # Remove stop words and short words
            return {w for w in words if w not in stop_words and len(w) > 2}

        kw1 = extract_keywords(message1)
        kw2 = extract_keywords(message2)

        if not kw1 or not kw2:
            return 0.0

        overlap = kw1 & kw2
        union = kw1 | kw2

        similarity = len(overlap) / len(union) if union else 0.0

        print(f"[TopicSimilarity] KW1: {kw1}")
        print(f"[TopicSimilarity] KW2: {kw2}")
        print(f"[TopicSimilarity] Overlap: {overlap}, Similarity: {similarity:.2f}")

        return similarity

    def _parse_json(self, text: str) -> Dict:
        """Parse JSON from LLM response."""
        import re

        try:
            return json.loads(text)
        except:
            pass

        # Try to find JSON in markdown blocks
        for pattern in [r"```json\s*([\s\S]*?)\s*```", r"```\s*([\s\S]*?)\s*```"]:
            match = re.search(pattern, text)
            if match:
                try:
                    return json.loads(match.group(1).strip())
                except:
                    continue

        # Find JSON object
        brace_start = text.find("{")
        if brace_start != -1:
            brace_count = 0
            for i, char in enumerate(text[brace_start:], brace_start):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        try:
                            return json.loads(text[brace_start : i + 1])
                        except:
                            break

        return {"is_new_question": False, "confidence": 0.5}


# Global detector instance
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

        # Clarification state tracking
        self.pending_clarification = False
        self.original_query: Optional[str] = None
        self.clarification_attempts = 0

        # ===== NEW: Track clarification flow state locally =====
        self.in_clarification_flow = False
        self.clarification_original_question = ""
        self.clarification_last_question = ""
        self.clarification_context = ""

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

    def _check_if_new_question(self, message: str) -> bool:
        """
        Check if the incoming message is a new unrelated question.
        Called BEFORE graph invocation to reset state if needed.
        """
        if not self.in_clarification_flow:
            return False

        print(f"\n{'#'*60}")
        print(f"[ChatbotService] Checking if new question while in clarification flow")
        print(f"[ChatbotService] Message: '{message}'")
        print(
            f"[ChatbotService] Original question: '{self.clarification_original_question}'"
        )
        print(f"{'#'*60}")

        result = new_question_detector.is_new_question(
            user_message=message,
            original_topic=self.clarification_original_question,
            last_question=self.clarification_last_question,
            clarification_context=self.clarification_context,
        )

        is_new = result.get("is_new_question", False)
        confidence = result.get("confidence", 0.5)

        print(
            f"[ChatbotService] Detection result: is_new={is_new}, confidence={confidence}"
        )

        # Require decent confidence
        return is_new and confidence >= 0.6

    def _reset_clarification_state(self):
        """Reset all clarification-related state."""
        print(f"[ChatbotService] ⚠️ RESETTING CLARIFICATION STATE")
        self.in_clarification_flow = False
        self.clarification_original_question = ""
        self.clarification_last_question = ""
        self.clarification_context = ""
        self.pending_clarification = False
        self.original_query = None
        self.clarification_attempts = 0

    def _update_clarification_state_from_result(
        self, result: dict, original_message: str
    ):
        """Update local clarification state based on graph result."""
        # Check if we entered a clarification flow
        if result.get("awaiting_scenario_selection", False) or result.get(
            "needs_scenario_clarification", False
        ):
            self.in_clarification_flow = True

            # Store the original question if not already set
            if not self.clarification_original_question:
                self.clarification_original_question = result.get(
                    "original_query", original_message
                )

            # Update the last clarification question from the response
            clarification_state = result.get("clarification_state", {})
            if clarification_state:
                self.clarification_last_question = clarification_state.get(
                    "current_question", ""
                )
                self.clarification_context = clarification_state.get(
                    "gathered_context", ""
                )

            # Also try to get from the last AI message
            for msg in reversed(result.get("messages", [])):
                if isinstance(msg, AIMessage) and msg.content:
                    # If it's a question, update
                    if "?" in msg.content:
                        self.clarification_last_question = msg.content
                    break

            print(f"[ChatbotService] Updated clarification state:")
            print(f"  - in_flow: {self.in_clarification_flow}")
            print(f"  - original_q: '{self.clarification_original_question}'")
            print(f"  - last_q: '{self.clarification_last_question[:50]}...'")

        # Check if clarification flow ended
        if not result.get("awaiting_scenario_selection", False) and not result.get(
            "needs_scenario_clarification", False
        ):
            clarification_state = result.get("clarification_state")
            if clarification_state is None or not clarification_state.get(
                "is_active", False
            ):
                # Flow ended naturally
                if self.in_clarification_flow:
                    print(f"[ChatbotService] Clarification flow ended naturally")
                self._reset_clarification_state()

    def _build_initial_state(
        self, message: str, force_fresh: bool = False
    ) -> AgentState:
        """Build initial state for the agent."""
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

        # Base state
        state = {
            "messages": [HumanMessage(content=message)],
            "context": self.context,
            "clarification_needed": False,
            "clarification_reason": "",
            "follow_up_questions": [],
            "pending_clarification": (
                False if force_fresh else self.pending_clarification
            ),
            "original_query": (
                message if force_fresh else (self.original_query or message)
            ),
            "clarification_attempts": 0 if force_fresh else self.clarification_attempts,
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
            "is_follow_up_question": False if force_fresh else False,
            "follow_up_context": "",
            # Scenario clarification fields
            "response_analysis": None,
            "needs_scenario_clarification": False,
            "scenario_clarification_question": "",
            "scenario_options": [],
            "original_full_response": "",
            "parsed_scenarios": [],
            "awaiting_scenario_selection": False,  # Force false if fresh
            "selected_scenario_id": None,
            "scenario_clarification_pending": False,  # Force false if fresh
            "user_scenario_context": "",
            "clarification_state": None,  # Force none if fresh
        }

        return state

    def _process_result(self, result: dict, message: str) -> str:
        """Process agent result and extract response."""
        # Update clarification tracking
        self._update_clarification_state_from_result(result, message)

        self.pending_clarification = result.get("pending_clarification", False)
        if self.pending_clarification:
            self.original_query = result.get("original_query", message)
            self.clarification_attempts = result.get("clarification_attempts", 0)
        else:
            if not self.in_clarification_flow:
                self.original_query = None
                self.clarification_attempts = 0

        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                response = msg.content
                is_not_found = result.get("should_respond_not_found", False)

                if not is_not_found:
                    if ResponseSanitizer.contains_general_knowledge(response):
                        search_results = result.get("search_results", "")
                        if search_results:
                            response = self._extract_document_answer(
                                search_results, message
                            )
                        else:
                            response = (
                                "I found some information but I'm not certain it directly answers your question. "
                                "Could you try rephrasing?"
                            )
                    else:
                        response = ResponseSanitizer.sanitize(response)

                return response

        return "I couldn't generate a response. Please try again."

    def _extract_document_answer(self, search_results: str, question: str) -> str:
        """Extract a simple answer from search results without hallucination."""
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

        # ===== PRE-CHECK: Is this a new question while in clarification flow? =====
        force_fresh = False
        if self._check_if_new_question(message):
            print(
                f"[ChatbotService] 🔄 NEW QUESTION DETECTED - Resetting state and starting fresh"
            )
            self._reset_clarification_state()
            force_fresh = True

            # Create a new graph instance with fresh checkpointer
            # This ensures no stale state from previous flow
            self.agent = create_rag_graph()

        config = {
            "configurable": {
                "thread_id": (
                    self.session_id
                    if not force_fresh
                    else f"{self.session_id}_{datetime.now().timestamp()}"
                )
            }
        }
        initial_state = self._build_initial_state(message, force_fresh=force_fresh)

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
        self._reset_clarification_state()

        # Create fresh graph
        self.agent = create_rag_graph()
        self._memory = memory_manager.get_or_create(self.session_id)

    def get_conversation_summary(self) -> dict:
        """Get summary of current conversation."""
        memory = memory_manager.get_or_create(self.session_id)
        return {
            "session_id": self.session_id,
            "turns": memory.get_turn_count(),
            "topics_discussed": memory.key_topics,
            "questions_asked": self.context.get("questions_asked", 0),
            "recent_context": memory.get_context_window(n_turns=3),
            "in_clarification_flow": self.in_clarification_flow,
        }

    def get_conversation_history(self) -> List[dict]:
        """Get full conversation history from Redis."""
        memory = memory_manager.get_or_create(self.session_id)
        return [turn.to_dict() for turn in memory.turns]


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

    def get_session_count(self) -> int:
        """Get number of active sessions."""
        return len(self._sessions)

    def get_all_session_ids(self) -> List[str]:
        """Get all active session IDs."""
        return list(self._sessions.keys())


# Global session manager
session_manager = SessionManager()


def get_chatbot_service(session_id: str) -> ChatbotService:
    """Get chatbot service for a session."""
    return session_manager.get_or_create(session_id)
