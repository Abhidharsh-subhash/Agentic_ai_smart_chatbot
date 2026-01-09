# app/services/rag/nodes.py
import json
import re
import random
from typing import Literal, Dict
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode

from app.core.config import settings
from app.services.embeddings import embedding_service
from .state import AgentState
from .config import InteractionMode, SearchQuality
from .analyzers import (
    QueryAnalyzer,
    NotFoundResponseGenerator,
    ResponseScenarioAnalyzer,
)
from .tools import tools
from .memory import memory_manager


# Initialize LLMs
llm = ChatOpenAI(
    model=settings.openai_model if hasattr(settings, "openai_model") else "gpt-4o",
    temperature=0.0,
    openai_api_key=settings.openai_api_key,
)

thinking_llm = ChatOpenAI(
    model=settings.openai_model if hasattr(settings, "openai_model") else "gpt-4o",
    temperature=0.0,
    openai_api_key=settings.openai_api_key,
)

# Response analyzer LLM (can use same or different model)
analyzer_llm = ChatOpenAI(
    model=settings.openai_model if hasattr(settings, "openai_model") else "gpt-4o",
    temperature=0.0,
    openai_api_key=settings.openai_api_key,
)

llm_with_tools = llm.bind_tools(tools)

# ============================================
# INLINE CONTEXT ANALYZER (to avoid import issues)
# ============================================

CONTEXT_ANALYSIS_PROMPT = """Determine if the user is responding to a clarification question OR asking a completely new unrelated question.

CURRENT CLARIFICATION CONTEXT:
- Original Question: {original_question}
- Last Clarification Question Asked: {last_question}

USER'S NEW INPUT: "{user_input}"

ANALYSIS RULES:

CLARIFICATION RESPONSE (is_new_question=false):
- "yes", "no", "nope", "yep", "correct", "that's right", etc.
- Describing their specific error/situation related to original topic
- References like "it shows...", "the error says...", "I'm seeing..."
- Short responses that answer the clarification question

NEW UNRELATED QUESTION (is_new_question=true):
- Different topic entirely (original was about errors → new is about "how to apply")
- Questions like "how can I...", "what is...", "how do I...", "can you tell me..."
- Makes complete sense WITHOUT the previous context
- Topic shift (troubleshooting → general inquiry)

CRITICAL: "how can i apply for visa" when original question was about "application posting failed" = NEW QUESTION (different topic)

Respond with ONLY valid JSON:
{{"is_new_question": true/false, "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""


def analyze_if_new_question(
    user_input: str,
    original_question: str,
    last_clarification_question: str,
) -> Dict:
    """
    Determine if user input is a new question or clarification response.
    Returns dict with is_new_question, confidence, reasoning.
    """
    user_lower = user_input.lower().strip()

    print(f"\n{'='*60}")
    print(f"[ContextAnalyzer] Analyzing input: '{user_input}'")
    print(f"[ContextAnalyzer] Original question: '{original_question}'")
    print(f"[ContextAnalyzer] Last clarification: '{last_clarification_question}'")
    print(f"{'='*60}")

    # ===== FAST PATH: Obvious clarification responses =====
    obvious_clarification = {
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
        "both",
        "all of them",
        "the first one",
        "the second one",
        "option 1",
        "option 2",
        "n",
        "y",
        "ok",
        "okay",
        "right",
        "exactly",
        "affirmative",
        "negative",
        "not really",
        "sort of",
    }

    # Check for exact matches first
    if user_lower in obvious_clarification:
        print(f"[ContextAnalyzer] FAST PATH: Obvious clarification '{user_lower}'")
        return {
            "is_new_question": False,
            "confidence": 0.95,
            "reasoning": f"Obvious clarification response: '{user_lower}'",
        }

    # ===== FAST PATH: Obvious new question patterns =====
    new_question_starters = [
        "how can i",
        "how do i",
        "how to",
        "what is",
        "what are",
        "can you tell me",
        "can you explain",
        "i want to know",
        "tell me about",
        "explain",
        "help me with",
        "what about",
        "i need to",
        "i want to",
        "can i",
        "could you",
        "would you",
        "where can i",
        "when can i",
        "who can",
        "why is",
        "why do",
    ]

    for starter in new_question_starters:
        if user_lower.startswith(starter):
            # Additional check: is this about a completely different topic?
            original_keywords = set(original_question.lower().split())
            input_keywords = set(user_lower.split())

            # Remove common words
            common_words = {
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
            }
            original_keywords -= common_words
            input_keywords -= common_words

            overlap = original_keywords & input_keywords
            overlap_ratio = len(overlap) / max(len(input_keywords), 1)

            print(f"[ContextAnalyzer] FAST PATH: New question pattern '{starter}'")
            print(
                f"[ContextAnalyzer] Keyword overlap: {overlap}, ratio: {overlap_ratio:.2f}"
            )

            # Low overlap = definitely new question
            if overlap_ratio < 0.3:
                print(f"[ContextAnalyzer] ✅ DETECTED AS NEW QUESTION (low overlap)")
                return {
                    "is_new_question": True,
                    "confidence": 0.9,
                    "reasoning": f"New question pattern '{starter}' with low topic overlap ({overlap_ratio:.2f})",
                }

    # ===== LLM PATH: Ambiguous cases =====
    print(f"[ContextAnalyzer] Using LLM for ambiguous case...")

    try:
        prompt = CONTEXT_ANALYSIS_PROMPT.format(
            original_question=original_question,
            last_question=last_clarification_question,
            user_input=user_input,
        )

        result = analyzer_llm.invoke([SystemMessage(content=prompt)])
        response_text = result.content.strip()

        print(f"[ContextAnalyzer] LLM response: {response_text[:200]}")

        # Parse JSON
        analysis = _extract_json_from_text(response_text)

        is_new = analysis.get("is_new_question", False)
        confidence = analysis.get("confidence", 0.5)
        reasoning = analysis.get("reasoning", "LLM analysis")

        print(f"[ContextAnalyzer] LLM result: is_new={is_new}, confidence={confidence}")

        return {
            "is_new_question": is_new,
            "confidence": confidence,
            "reasoning": reasoning,
        }

    except Exception as e:
        print(f"[ContextAnalyzer] LLM ERROR: {e}")
        import traceback

        traceback.print_exc()

        # Fallback: if it looks like a question and is long, probably new
        if "?" in user_input or len(user_input.split()) > 5:
            print(f"[ContextAnalyzer] FALLBACK: Treating as new question")
            return {
                "is_new_question": True,
                "confidence": 0.6,
                "reasoning": "Fallback: looks like a question",
            }

        return {
            "is_new_question": False,
            "confidence": 0.5,
            "reasoning": "Fallback: treating as clarification",
        }


def _extract_json_from_text(text: str) -> Dict:
    """Extract JSON from LLM response."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try markdown blocks
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


# ============================================
# SYSTEM PROMPTS
# ============================================

THINKING_PROMPT = """Analyze the user's question and plan how to search for the answer.

Conversation Context:
{conversation_context}

Topics Discussed:
{topics_discussed}

Current Question:
{user_question}

Is User Providing Clarification for Previous Scenarios: {is_scenario_clarification}
Previous Scenarios (if any): {previous_scenarios}

Respond with ONLY a JSON object (no markdown, no explanation, no code blocks):

{{"understanding": "what the user is asking", "is_follow_up": false, "is_scenario_response": false, "referenced_context": "", "key_topics": [], "search_queries": ["query1"], "reasoning": "why these queries"}}"""


AGENT_SYSTEM_PROMPT = """You are a STRICT document-lookup assistant. You ONLY report what documents say.

## CONVERSATION CONTEXT:
{conversation_context}

## YOUR ANALYSIS:
{thinking_output}

## CRITICAL RULES - FOLLOW EXACTLY:

### RULE 1: DOCUMENT-ONLY RESPONSES
- ONLY provide information that is EXPLICITLY written in the search results
- Do NOT add ANY information from your general knowledge
- Do NOT explain, elaborate, or provide context beyond what documents say

### RULE 2: EXACT EXTRACTION
When documents contain a solution or answer:
- Extract and report ONLY that solution/answer
- Do NOT add reasons why it's important

### RULE 3: SEARCH FIRST
- Call `search_documents` with the question
- Check the `should_respond` field
- If `should_respond: false` → say you don't have that information

### RULE 4: RESPONSE FORMAT
- Be direct and comprehensive
- Include all relevant scenarios from documents
- List all conditions/cases if multiple exist in documents

You are a lookup tool. Report what documents say accurately and completely."""


SCENARIO_CLARIFICATION_PROMPT = """The user asked a question that has multiple possible scenarios in the documentation.

## ORIGINAL QUESTION:
{original_question}

## DETECTED SCENARIOS:
{scenarios}

## CLARIFICATION QUESTION TO ASK:
{clarification_question}

Generate a friendly, conversational message that:
1. Acknowledges their question
2. Explains that there are multiple possible situations
3. Asks the clarifying question naturally
4. Optionally lists the main options briefly

Keep it concise and helpful. Do NOT answer the question yet - just ask for clarification."""


FOCUSED_RESPONSE_PROMPT = """Based on the user's clarification, provide a focused answer.

## ORIGINAL QUESTION:
{original_question}

## USER'S CLARIFICATION:
{user_clarification}

## MATCHED SCENARIO:
{matched_scenario}

## FULL CONTEXT FROM DOCUMENTS:
{full_response}

Generate a response that:
1. Directly addresses their specific situation
2. Provides the relevant solution clearly
3. Is concise and actionable
4. Does NOT mention other scenarios that don't apply

Be helpful and direct."""


# Tool node
tool_node = ToolNode(tools)

MAX_CLARIFICATION_ATTEMPTS = 10  # Maximum follow-up questions before giving up


# ============================================
# HELPER FUNCTIONS
# ============================================


def get_response_analyzer():
    """Get or create the response scenario analyzer."""
    from .analyzers import ResponseScenarioAnalyzer

    return ResponseScenarioAnalyzer(llm=analyzer_llm)


def extract_json_from_response(response_text: str) -> dict:
    """Extract JSON from LLM response, handling various formats."""
    if not response_text:
        raise ValueError("Empty response")

    text = response_text.strip()

    # Try direct JSON parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract from markdown code blocks
    code_block_patterns = [
        r"```json\s*([\s\S]*?)\s*```",
        r"```\s*([\s\S]*?)\s*```",
    ]

    for pattern in code_block_patterns:
        match = re.search(pattern, text)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                continue

    # Try to find JSON object in the text
    json_match = re.search(r"\{[^{}]*\}", text)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    # Try to find nested JSON
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
                    except json.JSONDecodeError:
                        break

    raise ValueError(f"Could not extract JSON from: {text[:200]}")


def get_response_analyzer():
    """Get or create the response scenario analyzer."""
    return ResponseScenarioAnalyzer(llm=analyzer_llm)


# ============================================
# NODE FUNCTIONS
# ============================================


# app/services/rag/nodes.py


def analyze_input(state: AgentState) -> dict:
    """Analyze user input for intent and clarity."""
    messages = state["messages"]
    context = state.get("context", {})
    session_id = context.get("session_id", "default")

    user_message = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break

    if not user_message:
        return {
            "clarification_needed": False,
            "interaction_mode": InteractionMode.QUERY.value,
        }

    memory = memory_manager.get_or_create(session_id)

    print(f"\n[AnalyzeInput] Processing: '{user_message}'")

    # Check if in clarification flow
    clarification_state = state.get("clarification_state")
    awaiting_scenario = state.get("awaiting_scenario_selection", False)

    # If clarification_state exists and is active, continue clarification flow
    # (New question detection happens BEFORE this in ChatbotService)
    if clarification_state and clarification_state.get("is_active", False):
        print(f"[AnalyzeInput] Continuing clarification flow")
        return {
            "interaction_mode": InteractionMode.QUERY.value,
            "clarification_needed": False,
            "is_follow_up_question": True,
            "user_scenario_context": user_message,
            "scenario_clarification_pending": True,
            "last_user_query": clarification_state.get("original_question", ""),
        }

    # Legacy check for awaiting_scenario_selection
    if awaiting_scenario:
        print(f"[AnalyzeInput] Continuing clarification flow (legacy)")
        return {
            "interaction_mode": InteractionMode.QUERY.value,
            "clarification_needed": False,
            "is_follow_up_question": True,
            "user_scenario_context": user_message,
            "scenario_clarification_pending": True,
            "last_user_query": state.get("original_query", ""),
        }

    # === Normal flow ===
    is_follow_up = memory.is_likely_follow_up(user_message)
    last_user = memory.get_last_user_query()
    last_assistant = memory.get_last_assistant_response()

    # Check for greetings
    if QueryAnalyzer.is_greeting(user_message):
        return {
            "interaction_mode": InteractionMode.GREETING.value,
            "clarification_needed": False,
            "is_follow_up_question": False,
            "awaiting_scenario_selection": False,
            "clarification_state": None,
        }

    # Check for closings
    if QueryAnalyzer.is_closing(user_message):
        return {
            "interaction_mode": InteractionMode.CLOSING.value,
            "clarification_needed": False,
            "is_follow_up_question": False,
            "awaiting_scenario_selection": False,
            "clarification_state": None,
        }

    # Handle pending vague query clarification
    if state.get("pending_clarification", False):
        return {
            "clarification_needed": False,
            "pending_clarification": False,
            "interaction_mode": InteractionMode.QUERY.value,
            "is_follow_up_question": True,
            "last_user_query": last_user or "",
            "last_assistant_response": last_assistant or "",
            "awaiting_scenario_selection": False,
        }

    # Analyze query clarity
    analysis = QueryAnalyzer.analyze(user_message, context)
    needs_clarification = not analysis["is_clear"] and analysis["confidence"] < 0.3

    return {
        "clarification_needed": needs_clarification,
        "clarification_reason": analysis.get("clarification_type", ""),
        "follow_up_questions": analysis.get("follow_up_questions", []),
        "pending_clarification": needs_clarification,
        "original_query": user_message,
        "search_confidence": analysis["confidence"],
        "interaction_mode": (
            InteractionMode.CLARIFICATION.value
            if needs_clarification
            else InteractionMode.QUERY.value
        ),
        "is_follow_up_question": is_follow_up,
        "last_user_query": last_user or "",
        "last_assistant_response": last_assistant or "",
        "awaiting_scenario_selection": False,
        "scenario_clarification_pending": False,
        "clarification_state": None,
    }


def think_and_plan(state: AgentState) -> dict:
    """Chain of Thought: Analyze question and plan search strategy."""
    messages = state["messages"]
    context = state.get("context", {})
    session_id = context.get("session_id", "default")

    # Get user question
    user_question = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_question = msg.content
            break

    if not user_question:
        return {
            "thinking": None,
            "planned_search_queries": [user_question] if user_question else [],
        }

    # Get conversation memory
    memory = memory_manager.get_or_create(session_id)
    conversation_context = memory.get_context_window(n_exchanges=4)
    topics_discussed = memory.get_topics_discussed()

    # Check if this is a scenario clarification
    is_scenario_clarification = state.get("scenario_clarification_pending", False)
    previous_scenarios = state.get("parsed_scenarios", [])

    # Build thinking prompt
    prompt = THINKING_PROMPT.format(
        conversation_context=conversation_context,
        topics_discussed=topics_discussed,
        user_question=user_question,
        is_scenario_clarification=is_scenario_clarification,
        previous_scenarios=(
            json.dumps(previous_scenarios[:3]) if previous_scenarios else "None"
        ),
    )

    try:
        response = thinking_llm.invoke([SystemMessage(content=prompt)])
        response_text = response.content.strip() if response.content else ""

        if not response_text:
            raise ValueError("Empty response from thinking LLM")

        thinking_output = extract_json_from_response(response_text)

        thinking = {
            "understanding": thinking_output.get("understanding", user_question),
            "key_topics": thinking_output.get("key_topics", []),
            "search_queries": thinking_output.get("search_queries", [user_question]),
            "reasoning": thinking_output.get("reasoning", ""),
            "is_follow_up": thinking_output.get("is_follow_up", False),
            "is_scenario_response": thinking_output.get(
                "is_scenario_response", is_scenario_clarification
            ),
            "referenced_context": thinking_output.get("referenced_context", ""),
        }

        if not thinking["search_queries"]:
            thinking["search_queries"] = [user_question]

        return {
            "thinking": thinking,
            "planned_search_queries": thinking["search_queries"],
            "detected_topics": thinking["key_topics"],
            "is_follow_up_question": thinking["is_follow_up"],
            "follow_up_context": thinking["referenced_context"] or "",
        }

    except Exception as e:
        print(f"[CoT] Error in thinking: {e}")
        return {
            "thinking": {
                "understanding": user_question,
                "key_topics": [],
                "search_queries": [user_question],
                "reasoning": "Direct search fallback",
                "is_follow_up": state.get("is_follow_up_question", False),
                "is_scenario_response": is_scenario_clarification,
                "referenced_context": "",
            },
            "planned_search_queries": [user_question],
        }


def handle_greeting(state: AgentState) -> dict:
    """Handle greeting messages."""
    context = state.get("context", {})
    session_id = context.get("session_id", "default")
    memory = memory_manager.get_or_create(session_id)

    if memory.get_turn_count() > 0:
        greetings = [
            "Welcome back! How can I help you continue our conversation?",
            "Hello again! What would you like to know?",
            "Hi! Ready to help. What's on your mind?",
        ]
    else:
        greetings = [
            "Hello! I'm your support assistant. I can answer questions based on the available documentation. How can I help you today?",
            "Hi there! I'm here to help you find information from our knowledge base. What would you like to know?",
        ]

    response = random.choice(greetings)
    memory.add_user_message("greeting")
    memory.add_assistant_message(response)

    return {"messages": [AIMessage(content=response)]}


def handle_closing(state: AgentState) -> dict:
    """Handle closing messages."""
    context = state.get("context", {})
    session_id = context.get("session_id", "default")
    memory = memory_manager.get_or_create(session_id)

    if memory.get_turn_count() > 4:
        topics = memory.get_topics_discussed()
        closings = [
            f"You're welcome! We covered {topics}. Feel free to come back anytime! 👋",
            "Happy to help! Don't hesitate to ask if anything else comes up. Take care!",
        ]
    else:
        closings = [
            "You're welcome! Feel free to come back if you have more questions. Have a great day! 👋",
            "Happy to help! Don't hesitate to ask if anything else comes up. Take care!",
        ]

    response = random.choice(closings)
    memory.add_user_message("closing")
    memory.add_assistant_message(response)

    return {"messages": [AIMessage(content=response)]}


def handle_document_listing(state: AgentState) -> dict:
    """Handle requests for available documents listing."""
    context = state.get("context", {})
    session_id = context.get("session_id", "default")
    memory = memory_manager.get_or_create(session_id)

    try:
        result = embedding_service.get_available_documents()

        if not result.get("exists") or result.get("total_files", 0) == 0:
            message = (
                "📂 **No documents available**\n\n"
                "There are currently no documents in the knowledge base."
            )
        else:
            files = result.get("files", [])
            total_files = result.get("total_files", 0)
            total_chunks = result.get("total_chunks", 0)

            message = f"📂 **Available Documents**\n\n"
            message += f"**Total Files:** {total_files}\n"
            message += f"**Total Chunks:** {total_chunks}\n"

        memory.add_user_message("document listing request")
        memory.add_assistant_message(message, topics=["documents"])

        return {"messages": [AIMessage(content=message)]}

    except Exception as e:
        print(f"Error listing documents: {e}")
        return {
            "messages": [
                AIMessage(
                    content="I encountered an error retrieving the document list."
                )
            ]
        }


def ask_clarification(state: AgentState) -> dict:
    """Ask for clarification when needed (for vague queries, not scenarios)."""
    follow_up_questions = state.get("follow_up_questions", [])
    attempts = state.get("clarification_attempts", 0)

    if attempts >= 2:
        return {
            "messages": [AIMessage(content="Let me search with what I have...")],
            "clarification_needed": False,
            "pending_clarification": False,
        }

    message = (
        follow_up_questions[0]
        if follow_up_questions
        else "Could you provide more details?"
    )

    return {
        "messages": [AIMessage(content=message)],
        "pending_clarification": True,
        "clarification_attempts": attempts + 1,
    }


def agent(state: AgentState) -> dict:
    """Main agent node with strict document-only responses."""
    messages = state["messages"]
    context = state.get("context", {})
    session_id = context.get("session_id", "default")
    thinking = state.get("thinking", {})

    memory = memory_manager.get_or_create(session_id)
    conversation_context = memory.get_context_window(n_exchanges=4)

    if thinking:
        thinking_str = f"""
Understanding: {thinking.get('understanding', 'N/A')}
Is Follow-up: {thinking.get('is_follow_up', False)}
Key Topics: {', '.join(thinking.get('key_topics', []))}
Search Queries: {', '.join(thinking.get('search_queries', []))}
"""
    else:
        thinking_str = "No prior analysis available."

    system_prompt = AGENT_SYSTEM_PROMPT.format(
        conversation_context=conversation_context,
        thinking_output=thinking_str,
    )

    system = SystemMessage(content=system_prompt)
    response = llm_with_tools.invoke([system] + list(messages))

    return {"messages": [response], "has_searched": True}


def validate_search_results(state: AgentState) -> dict:
    """Validate search results and determine if we should respond."""
    messages = state["messages"]

    last_tool_result = None
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            try:
                last_tool_result = json.loads(msg.content)
                break
            except (json.JSONDecodeError, AttributeError):
                continue

    if last_tool_result is None:
        return {"should_respond_not_found": False}

    should_respond = last_tool_result.get("should_respond", False)
    quality = last_tool_result.get("quality", SearchQuality.NOT_FOUND.value)
    confidence = last_tool_result.get("confidence", 0)

    if not should_respond or quality == SearchQuality.NOT_FOUND.value:
        not_found_msg = NotFoundResponseGenerator.generate(
            query=state.get("original_query", ""),
            search_analysis=last_tool_result,
        )

        return {
            "should_respond_not_found": True,
            "not_found_message": not_found_msg,
            "search_quality": quality,
            "search_confidence": confidence,
            "found_relevant_info": False,
        }

    return {
        "should_respond_not_found": False,
        "search_quality": quality,
        "search_confidence": confidence,
        "found_relevant_info": True,
    }


def handle_not_found(state: AgentState) -> dict:
    """Handle case when no relevant information was found."""
    context = state.get("context", {})
    session_id = context.get("session_id", "default")
    memory = memory_manager.get_or_create(session_id)

    not_found_message = state.get("not_found_message")

    if not not_found_message:
        if state.get("is_follow_up_question"):
            not_found_message = (
                "I don't have additional information about that in my knowledge base. "
                "Would you like to ask about something else?"
            )
        else:
            not_found_message = (
                "I don't have information about that topic in my knowledge base. "
                "Would you like to ask about something else?"
            )

    # Get user message for memory
    user_msg = None
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            user_msg = msg.content
            break

    if user_msg:
        memory.add_user_message(user_msg)
    memory.add_assistant_message(not_found_message)

    return {"messages": [AIMessage(content=not_found_message)]}


# ============================================
# NEW: SCENARIO ANALYSIS NODES
# ============================================


def analyze_response_scenarios(state: AgentState) -> dict:
    """
    Analyze response for multiple scenarios.
    Initialize multi-turn clarification if needed.
    """
    messages = state["messages"]

    # Skip if already in clarification flow
    if state.get("scenario_clarification_pending", False):
        return {"needs_scenario_clarification": False}

    clarification_state = state.get("clarification_state")
    if clarification_state and clarification_state.get("is_active", False):
        return {"needs_scenario_clarification": False}

    # Get last AI response
    last_ai_response = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            last_ai_response = msg.content
            break

    if not last_ai_response:
        return {"needs_scenario_clarification": False}

    # Get user question
    user_question = None
    for msg in messages:
        if isinstance(msg, HumanMessage):
            user_question = msg.content
            break

    if not user_question:
        return {"needs_scenario_clarification": False}

    try:
        analyzer = get_response_analyzer()
        analysis = analyzer.analyze_response(last_ai_response, user_question)

        if (
            analysis.get("has_multiple_scenarios", False)
            and analysis.get("scenario_count", 0) >= 2  # Changed from 3 to 2
        ):
            scenarios = analysis.get("scenarios", [])
            first_question = analysis.get("first_question", "")

            # If no first question, use the identifying question from first scenario
            if not first_question and scenarios:
                first_question = scenarios[0].get(
                    "identifying_question",
                    "Could you describe what's happening when you try this?",
                )

            # Get the first scenario's ID for tracking
            first_scenario_id = scenarios[0].get("id") if scenarios else None

            print(
                f"[ScenarioAnalysis] Detected {len(scenarios)} scenarios, starting clarification flow"
            )

            # ============ DYNAMIC MAX ATTEMPTS ============
            # Set max attempts based on number of scenarios
            # Each scenario gets ONE chance to be asked about
            # We cap it at MAX_CLARIFICATION_ATTEMPTS for safety
            dynamic_max_attempts = min(len(scenarios), MAX_CLARIFICATION_ATTEMPTS)

            print(
                f"[ScenarioAnalysis] Dynamic max attempts set to: {dynamic_max_attempts}"
            )

            # Initialize clarification state
            new_clarification_state = {
                "is_active": True,
                "attempt_count": 0,
                "max_attempts": dynamic_max_attempts,  # DYNAMIC instead of hardcoded
                "original_question": user_question,
                "all_scenarios": scenarios,
                "remaining_scenarios": scenarios.copy(),
                "eliminated_scenarios": [],
                "user_responses": [],
                "asked_questions": [],
                "current_question": first_question,
                "gathered_context": "",
                # NEW: Track which scenario the current question is about
                "current_scenario_id": first_scenario_id,
                "scenario_question_map": {
                    s.get("id"): s.get("identifying_question", "") for s in scenarios
                },
            }

            return {
                "needs_scenario_clarification": True,
                "response_analysis": analysis,
                "scenario_clarification_question": first_question,
                "original_full_response": last_ai_response,
                "parsed_scenarios": scenarios,
                "awaiting_scenario_selection": True,
                "original_query": user_question,
                "clarification_state": new_clarification_state,
            }
        else:
            return {
                "needs_scenario_clarification": False,
                "clarification_state": None,
            }

    except Exception as e:
        print(f"[ScenarioAnalysis] Error: {e}")
        import traceback

        traceback.print_exc()
        return {"needs_scenario_clarification": False}


def ask_scenario_clarification(state: AgentState) -> dict:
    """
    Ask the current clarification question.
    Track which scenario this question is about.
    """
    context = state.get("context", {})
    session_id = context.get("session_id", "default")
    memory = memory_manager.get_or_create(session_id)

    clarification_state = state.get("clarification_state", {})
    current_question = clarification_state.get("current_question", "")

    if not current_question:
        current_question = state.get(
            "scenario_clarification_question",
            "Could you provide more details about what you're experiencing?",
        )

    # Save to memory
    user_msg = None
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            user_msg = msg.content
            break

    if user_msg:
        memory.add_user_message(user_msg, state.get("detected_topics", []))
    memory.add_assistant_message(current_question)

    # Update clarification state with question tracking
    if clarification_state:
        clarification_state["asked_questions"].append(current_question)

        # Find and track which scenario this question is for
        remaining = clarification_state.get("remaining_scenarios", [])
        for scenario in remaining:
            if scenario.get("identifying_question") == current_question:
                clarification_state["current_scenario_id"] = scenario.get("id")
                print(
                    f"[AskClarification] Tracking scenario ID: {scenario.get('id')} for question"
                )
                break

    return {
        "messages": [AIMessage(content=current_question)],
        "awaiting_scenario_selection": True,
        "pending_clarification": False,
        "clarification_state": clarification_state,
    }


def process_scenario_selection(state: AgentState) -> dict:
    """
    Process user's response in multi-turn clarification flow.
    Either narrow down, match, or ask another question.
    """
    context = state.get("context", {})
    session_id = context.get("session_id", "default")
    memory = memory_manager.get_or_create(session_id)

    user_response = state.get("user_scenario_context", "")
    clarification_state = state.get("clarification_state", {})

    # Get data from state
    remaining_scenarios = clarification_state.get(
        "remaining_scenarios", state.get("parsed_scenarios", [])
    )
    original_question = clarification_state.get(
        "original_question", state.get("original_query", "")
    )
    gathered_context = clarification_state.get("gathered_context", "")
    asked_questions = clarification_state.get("asked_questions", [])
    attempt_count = clarification_state.get("attempt_count", 0)
    max_attempts = clarification_state.get("max_attempts", len(remaining_scenarios))
    user_responses = clarification_state.get("user_responses", [])
    last_question = asked_questions[-1] if asked_questions else ""
    current_scenario_id = clarification_state.get("current_scenario_id")
    all_scenarios = clarification_state.get("all_scenarios", [])

    # Update tracking
    user_responses.append(user_response)
    attempt_count += 1

    # Update gathered context
    gathered_context = f"{gathered_context}\nUser said: {user_response}".strip()

    # ============ HELPER: Generate options list ============
    def get_available_options_message():
        """Generate a message listing all available options."""
        if not all_scenarios:
            return "Could you please rephrase your question or provide more details?"

        # Extract titles or conditions from all scenarios
        options = []
        for s in all_scenarios:
            title = s.get("title", "")
            condition = s.get("condition", "")
            if title:
                options.append(title)
            elif condition:
                options.append(condition)

        if options:
            options_list = ", ".join(options[:-1])
            if len(options) > 1:
                options_list += f" or {options[-1]}"
            else:
                options_list = options[0]
            return (
                f"I couldn't find an exact match for your situation. "
                f"I have information about: {options_list}.\n\n"
                f"Could you please specify which one applies to you, or describe your situation in more detail?"
            )
        return "Could you please rephrase your question or provide more details about your specific situation?"

    # No match response - now more helpful
    no_match_response = get_available_options_message()

    if not user_response or not remaining_scenarios:
        memory.add_user_message(user_response or "")
        memory.add_assistant_message(no_match_response)
        return {
            "messages": [AIMessage(content=no_match_response)],
            "awaiting_scenario_selection": False,
            "scenario_clarification_pending": False,
            "clarification_state": None,
        }

    # ============ HANDLE EXPLICIT "NO" RESPONSES ============
    user_lower = user_response.lower().strip()
    is_explicit_no = user_lower in [
        "no",
        "nope",
        "n",
        "not that",
        "not this",
        "no, not that",
        "nah",
        "negative",
        "none of these",
        "neither",
        "not really",
        "no thanks",
        "not applicable",
        "na",
        "nop",
    ]

    # Check for explicit "yes" responses
    is_explicit_yes = user_lower in [
        "yes",
        "yeah",
        "yep",
        "y",
        "correct",
        "that's right",
        "right",
        "exactly",
        "yes please",
        "yup",
        "affirmative",
        "that one",
        "yes, that's it",
        "that's the one",
        "bingo",
        "yes!",
    ]

    # If user explicitly said "yes" and we know which scenario was asked about
    if is_explicit_yes and current_scenario_id is not None:
        print(
            f"[ScenarioSelection] User said 'yes' - matching scenario {current_scenario_id}"
        )

        # Find the matched scenario
        matched_scenario = None
        for s in all_scenarios:
            if s.get("id") == current_scenario_id:
                matched_scenario = s
                break

        if matched_scenario:
            analyzer = get_response_analyzer()
            final_response = analyzer.generate_final_response(
                matched_scenario, gathered_context
            )

            memory.add_user_message(user_response, state.get("detected_topics", []))
            memory.add_assistant_message(final_response)

            return {
                "messages": [AIMessage(content=final_response)],
                "awaiting_scenario_selection": False,
                "scenario_clarification_pending": False,
                "clarification_state": None,
                "selected_scenario_id": current_scenario_id,
            }

    # If user explicitly said "no" and we know which scenario was asked about
    if is_explicit_no and current_scenario_id is not None:
        print(
            f"[ScenarioSelection] User said 'no' - eliminating scenario {current_scenario_id}"
        )

        # Immediately eliminate the current scenario
        remaining_scenarios = [
            s for s in remaining_scenarios if s.get("id") != current_scenario_id
        ]
        eliminated_scenarios = clarification_state.get("eliminated_scenarios", [])
        eliminated_scenarios.append(current_scenario_id)

        print(
            f"[ScenarioSelection] Remaining scenarios after elimination: {len(remaining_scenarios)}"
        )

        # Check if we've exhausted all scenarios
        if len(remaining_scenarios) == 0:
            print(
                f"[ScenarioSelection] All scenarios eliminated - asking for clarification"
            )

            # Generate helpful message with all options
            clarification_message = get_available_options_message()

            memory.add_user_message(user_response)
            memory.add_assistant_message(clarification_message)
            return {
                "messages": [AIMessage(content=clarification_message)],
                "awaiting_scenario_selection": False,
                "scenario_clarification_pending": False,
                "clarification_state": None,
            }

        # If only one scenario remains, ASK about it instead of assuming
        if len(remaining_scenarios) == 1:
            last_scenario = remaining_scenarios[0]
            last_scenario_title = last_scenario.get("title", "this option")

            # Ask specifically about the last scenario
            confirmation_question = f"Is your question about {last_scenario_title}?"

            # Update clarification state to track we're asking about the last one
            updated_state = {
                "is_active": True,
                "attempt_count": attempt_count,
                "max_attempts": max_attempts,
                "original_question": original_question,
                "all_scenarios": all_scenarios,
                "remaining_scenarios": remaining_scenarios,
                "eliminated_scenarios": eliminated_scenarios,
                "user_responses": user_responses,
                "asked_questions": asked_questions + [confirmation_question],
                "current_question": confirmation_question,
                "gathered_context": gathered_context,
                "current_scenario_id": last_scenario.get("id"),
                "scenario_question_map": clarification_state.get(
                    "scenario_question_map", {}
                ),
                "is_last_scenario_confirmation": True,  # NEW: Flag to track this is the last
            }

            memory.add_user_message(user_response)
            memory.add_assistant_message(confirmation_question)

            return {
                "messages": [AIMessage(content=confirmation_question)],
                "awaiting_scenario_selection": True,
                "scenario_clarification_pending": False,
                "clarification_state": updated_state,
            }

        # More than one scenario remains - ask about the next one
        next_scenario = remaining_scenarios[0]
        next_question = next_scenario.get(
            "identifying_question",
            f"Is your situation related to: {next_scenario.get('title', 'this scenario')}?",
        )

        # Update clarification state
        updated_state = {
            "is_active": True,
            "attempt_count": attempt_count,
            "max_attempts": max_attempts,
            "original_question": original_question,
            "all_scenarios": all_scenarios,
            "remaining_scenarios": remaining_scenarios,
            "eliminated_scenarios": eliminated_scenarios,
            "user_responses": user_responses,
            "asked_questions": asked_questions + [next_question],
            "current_question": next_question,
            "gathered_context": gathered_context,
            "current_scenario_id": next_scenario.get("id"),
            "scenario_question_map": clarification_state.get(
                "scenario_question_map", {}
            ),
        }

        memory.add_user_message(user_response)
        memory.add_assistant_message(next_question)

        return {
            "messages": [AIMessage(content=next_question)],
            "awaiting_scenario_selection": True,
            "scenario_clarification_pending": False,
            "clarification_state": updated_state,
        }

    # ============ HANDLE NON-EXPLICIT RESPONSES (descriptions, partial info, etc.) ============
    try:
        analyzer = get_response_analyzer()

        # Evaluate user's response
        evaluation = analyzer.evaluate_user_response(
            remaining_scenarios=remaining_scenarios,
            user_response=user_response,
            previous_context=gathered_context,
            asked_question=last_question,
        )

        matched_id = evaluation.get("matched_scenario_id")
        confidence = evaluation.get("confidence_in_match", 0)
        eliminated_ids = evaluation.get("eliminated_scenario_ids", [])
        needs_more = evaluation.get("needs_more_info", True)
        suggested_question = evaluation.get("suggested_next_question", "")

        print(
            f"[ScenarioSelection] Attempt {attempt_count}/{max_attempts}, "
            f"Confidence: {confidence}, Eliminated: {eliminated_ids}, "
            f"Matched: {matched_id}, Remaining before filter: {len(remaining_scenarios)}"
        )

        # Filter out eliminated scenarios
        if eliminated_ids:
            remaining_scenarios = [
                s for s in remaining_scenarios if s.get("id") not in eliminated_ids
            ]
            print(
                f"[ScenarioSelection] After elimination: {len(remaining_scenarios)} scenarios remain"
            )

        # Check if we have a confident match
        if matched_id is not None and confidence >= 0.7:
            matched_scenario = None
            for s in all_scenarios:
                if s.get("id") == matched_id:
                    matched_scenario = s
                    break

            if matched_scenario:
                final_response = analyzer.generate_final_response(
                    matched_scenario, gathered_context
                )

                memory.add_user_message(user_response, state.get("detected_topics", []))
                memory.add_assistant_message(final_response)

                return {
                    "messages": [AIMessage(content=final_response)],
                    "awaiting_scenario_selection": False,
                    "scenario_clarification_pending": False,
                    "clarification_state": None,
                    "selected_scenario_id": matched_id,
                }

        # Check if only one scenario remains
        if len(remaining_scenarios) == 1:
            # Check if this is a confirmation response
            is_last_confirmation = clarification_state.get(
                "is_last_scenario_confirmation", False
            )

            if is_last_confirmation and is_explicit_no:
                # User said no to the last scenario too - all exhausted
                clarification_message = get_available_options_message()
                memory.add_user_message(user_response)
                memory.add_assistant_message(clarification_message)
                return {
                    "messages": [AIMessage(content=clarification_message)],
                    "awaiting_scenario_selection": False,
                    "scenario_clarification_pending": False,
                    "clarification_state": None,
                }

            # If we got here through elimination (not direct "no"), give the answer
            single_scenario = remaining_scenarios[0]
            final_response = analyzer.generate_final_response(
                single_scenario, gathered_context
            )

            memory.add_user_message(user_response, state.get("detected_topics", []))
            memory.add_assistant_message(final_response)

            return {
                "messages": [AIMessage(content=final_response)],
                "awaiting_scenario_selection": False,
                "scenario_clarification_pending": False,
                "clarification_state": None,
            }

        # Check if no scenarios remain
        if len(remaining_scenarios) == 0:
            clarification_message = get_available_options_message()
            memory.add_user_message(user_response)
            memory.add_assistant_message(clarification_message)

            return {
                "messages": [AIMessage(content=clarification_message)],
                "awaiting_scenario_selection": False,
                "scenario_clarification_pending": False,
                "clarification_state": None,
            }

        # Check if max attempts reached
        if attempt_count >= max_attempts:
            # Instead of giving best guess, ask for clarification
            clarification_message = get_available_options_message()

            memory.add_user_message(user_response)
            memory.add_assistant_message(clarification_message)

            return {
                "messages": [AIMessage(content=clarification_message)],
                "awaiting_scenario_selection": False,
                "scenario_clarification_pending": False,
                "clarification_state": None,
            }

        # Need more info - generate next question
        if needs_more and len(remaining_scenarios) > 1:
            # Use suggested question or generate new one
            if suggested_question and suggested_question not in asked_questions:
                next_question = suggested_question
                # Find scenario ID for this question
                next_scenario_id = None
                for s in remaining_scenarios:
                    if s.get("identifying_question") == suggested_question:
                        next_scenario_id = s.get("id")
                        break
            else:
                question_data = analyzer.generate_next_question(
                    remaining_scenarios=remaining_scenarios,
                    context=gathered_context,
                    asked_questions=asked_questions,
                )
                next_question = question_data.get("question", "")
                next_scenario_id = None

            if not next_question:
                # Use identifying question from first remaining scenario
                if remaining_scenarios:
                    next_scenario = remaining_scenarios[0]
                    next_question = next_scenario.get(
                        "identifying_question",
                        "Could you describe the error in more detail?",
                    )
                    next_scenario_id = next_scenario.get("id")

            # Update clarification state
            updated_state = {
                "is_active": True,
                "attempt_count": attempt_count,
                "max_attempts": max_attempts,
                "original_question": original_question,
                "all_scenarios": all_scenarios,
                "remaining_scenarios": remaining_scenarios,
                "eliminated_scenarios": clarification_state.get(
                    "eliminated_scenarios", []
                )
                + eliminated_ids,
                "user_responses": user_responses,
                "asked_questions": asked_questions + [next_question],
                "current_question": next_question,
                "gathered_context": gathered_context,
                "current_scenario_id": next_scenario_id,
                "scenario_question_map": clarification_state.get(
                    "scenario_question_map", {}
                ),
            }

            memory.add_user_message(user_response)
            memory.add_assistant_message(next_question)

            return {
                "messages": [AIMessage(content=next_question)],
                "awaiting_scenario_selection": True,
                "scenario_clarification_pending": False,
                "clarification_state": updated_state,
            }

        # Fallback - ask for clarification
        clarification_message = get_available_options_message()
        memory.add_user_message(user_response)
        memory.add_assistant_message(clarification_message)

        return {
            "messages": [AIMessage(content=clarification_message)],
            "awaiting_scenario_selection": False,
            "scenario_clarification_pending": False,
            "clarification_state": None,
        }

    except Exception as e:
        print(f"[ScenarioSelection] Error: {e}")
        import traceback

        traceback.print_exc()

        clarification_message = get_available_options_message()
        memory.add_user_message(user_response)
        memory.add_assistant_message(clarification_message)

        return {
            "messages": [AIMessage(content=clarification_message)],
            "awaiting_scenario_selection": False,
            "scenario_clarification_pending": False,
            "clarification_state": None,
        }


def save_to_memory(state: AgentState) -> dict:
    """Save the conversation turn to memory."""
    messages = state["messages"]
    context = state.get("context", {})
    session_id = context.get("session_id", "default")
    detected_topics = state.get("detected_topics", [])

    memory = memory_manager.get_or_create(session_id)

    user_msg = None
    assistant_msg = None

    for msg in messages:
        if isinstance(msg, HumanMessage):
            user_msg = msg.content
        elif isinstance(msg, AIMessage) and msg.content:
            assistant_msg = msg.content

    if user_msg:
        memory.add_user_message(user_msg, detected_topics)
    if assistant_msg:
        memory.add_assistant_message(assistant_msg, detected_topics)

    return {}


# ============================================
# ROUTING FUNCTIONS
# ============================================


def should_continue(state: AgentState) -> Literal["tools", "analyze_scenarios"]:
    """Determine if we should continue to tools or analyze response."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "analyze_scenarios"


def route_after_analysis(
    state: AgentState,
) -> Literal[
    "handle_greeting",
    "handle_closing",
    "handle_document_listing",
    "ask_clarification",
    "think_and_plan",
    "process_scenario_selection",
]:
    """Route based on analysis results."""

    print(f"\n[RouteAfterAnalysis] Making routing decision...")
    print(
        f"[RouteAfterAnalysis] scenario_clarification_pending: {state.get('scenario_clarification_pending', False)}"
    )
    print(
        f"[RouteAfterAnalysis] awaiting_scenario_selection: {state.get('awaiting_scenario_selection', False)}"
    )
    print(
        f"[RouteAfterAnalysis] clarification_state: {state.get('clarification_state') is not None}"
    )
    print(f"[RouteAfterAnalysis] interaction_mode: {state.get('interaction_mode')}")

    # Check if we should process scenario selection
    if state.get("scenario_clarification_pending", False):
        clarification_state = state.get("clarification_state")
        if clarification_state and clarification_state.get("is_active", False):
            print(f"[RouteAfterAnalysis] ➡️  Routing to process_scenario_selection")
            return "process_scenario_selection"
        elif state.get("awaiting_scenario_selection", False):
            print(
                f"[RouteAfterAnalysis] ➡️  Routing to process_scenario_selection (legacy)"
            )
            return "process_scenario_selection"

    mode = state.get("interaction_mode", InteractionMode.QUERY.value)

    if mode == InteractionMode.GREETING.value:
        print(f"[RouteAfterAnalysis] ➡️  Routing to handle_greeting")
        return "handle_greeting"

    if mode == InteractionMode.CLOSING.value:
        print(f"[RouteAfterAnalysis] ➡️  Routing to handle_closing")
        return "handle_closing"

    if mode == "document_listing":
        print(f"[RouteAfterAnalysis] ➡️  Routing to handle_document_listing")
        return "handle_document_listing"

    if state.get("clarification_needed", False):
        attempts = state.get("clarification_attempts", 0)
        if attempts < 2:
            print(f"[RouteAfterAnalysis] ➡️  Routing to ask_clarification")
            return "ask_clarification"

    print(f"[RouteAfterAnalysis] ➡️  Routing to think_and_plan (new question)")
    return "think_and_plan"


def route_after_validation(state: AgentState) -> Literal["handle_not_found", "agent"]:
    """Route based on search result validation."""
    if state.get("should_respond_not_found", False):
        return "handle_not_found"
    return "agent"


def route_after_scenario_analysis(
    state: AgentState,
) -> Literal["ask_scenario_clarification", "save_memory"]:
    """Route based on scenario analysis results."""
    if state.get("needs_scenario_clarification", False):
        return "ask_scenario_clarification"
    return "save_memory"
