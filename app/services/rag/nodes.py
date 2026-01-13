# app/services/rag/nodes.py
"""
Node functions - matching standalone agentic_ai_logic.py EXACTLY.
"""
import json
import random
import re
from typing import Literal, Optional, List

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode

from app.core.config import settings
from .state import AgentState
from .config import Config, InteractionMode, SearchQuality, ScenarioStatus
from .tools import tools


# ==================== LLM Setup ====================

llm = ChatOpenAI(
    model=getattr(settings, "openai_model", "gpt-4o"),
    temperature=0.1,
    openai_api_key=settings.openai_api_key,
)

llm_with_tools = llm.bind_tools(tools)

tool_node = ToolNode(tools)


# ==================== System Prompt - EXACT copy from standalone ====================

SYSTEM_PROMPT = """You are a highly specialized document-based support assistant. Your SOLE purpose is to provide information by STRICTLY and ONLY extracting or directly quoting content from the search results provided.

## CRITICAL WORKFLOW:

### STEP 1: ALWAYS SEARCH FIRST
For ANY user question, IMMEDIATELY call `search_and_analyze` tool.
DO NOT ask clarifying questions BEFORE searching.
DO NOT speculate about what scenarios might exist.

### STEP 2: INTERPRET TOOL RESULTS
After getting results from `search_and_analyze` or `get_scenario_answer` (these results will contain document snippets in their `documents` array):

**If the tool result indicates `found_answer: false` (from either tool):**
- Respond: "I don't have information about that in my knowledge base."
- Avoid offering further assistance unless explicitly asked, to remain strictly focused.

**If the tool result indicates `found_answer: true` AND `disambiguation_needed: false` (from `search_and_analyze`), OR if results come from `get_scenario_answer`:**
- Your task is to generate an answer by PRECISELY extracting or directly rephrasing specific sentences or bullet points that are EXPLICITLY and DIRECTLY stated within the `content` field of the `documents` from the tool's output.
- **ABSOLUTELY DO NOT:**
    - Introduce any external knowledge or information not present in the provided document snippets.
    - Make inferences, deductions, or assumptions.
    - Elaborate, expand, or add examples beyond what is explicitly written in the `documents` content.
    - Provide any information that is not directly traceable to the provided `documents` content.
- If the provided `documents` content does not contain a direct answer to a specific part of the user's question, you must explicitly state that "The available information does not specify..." or "I couldn't find details on..." for that particular aspect.
- Be concise, factual, and extremely literal in your interpretation and presentation of the source material.

**If the tool result indicates `found_answer: true` AND `disambiguation_needed: true` (from `search_and_analyze`):**
- Present the specific scenarios found (from the `scenarios` array in the tool output) EXACTLY as they are described (using titles and descriptions).
- Ask the user to choose which applies to them, using the `disambiguation_question` if provided, or construct a direct and clear question from the `scenarios` titles/descriptions.
- You MUST wait for their response.

### STEP 3: AFTER USER SELECTS SCENARIO
- The `get_scenario_answer` tool will be called automatically.
- Once its results are returned (a `ToolMessage` containing `documents`), apply the same strict extraction rules as described above for `found_answer: true`.

## ABSOLUTE, UNCOMPROMISING RULES FOR ALL RESPONSES:

1. **NEVER ask for clarification BEFORE searching.**
2. **NEVER speculate about scenarios that might exist; only present what tools explicitly report.**
3. **ONLY use information EXPLICITLY stated in the provided tool output's `documents` content for answers.**
   - Refer to the "ABSOLUTELY DO NOT" section above for detailed prohibitions.
   - If information is not in the documents, state its absence.
4. **If content is NOT found, say so clearly and briefly.**
5. **DO NOT mention "documents", "search results", "knowledge base", "according to the documents", "based on my information" etc. when giving an answer.** Just provide the extracted information naturally, but strictly from the source.
6. When asking for clarification (only for disambiguation), be direct and clear.
7. Maintain a neutral, factual, and objective tone.
"""


# ==================== Helper Classes - EXACT copy from standalone ====================


class QueryAnalyzer:
    """Analyzes user queries - simplified to avoid premature clarification."""

    @classmethod
    def is_greeting(cls, query: str) -> bool:
        greetings = [
            "hi",
            "hello",
            "hey",
            "good morning",
            "good afternoon",
            "good evening",
            "howdy",
            "greetings",
            "hi there",
            "hello there",
        ]
        return query.lower().strip().rstrip("!.,") in greetings

    @classmethod
    def is_closing(cls, query: str) -> bool:
        closings = [
            "bye",
            "goodbye",
            "see you",
            "thanks",
            "thank you",
            "that's all",
            "done",
            "exit",
            "quit",
            "thx",
        ]
        query_clean = query.lower().strip().rstrip("!.,")
        return any(c in query_clean for c in closings)

    @classmethod
    def is_too_short(cls, query: str) -> bool:
        """Check if query is too short to be meaningful."""
        return len(query.strip()) < 3

    @classmethod
    def is_scenario_selection(
        cls, query: str, available_options: List[str]
    ) -> Optional[str]:
        """Check if user's response is selecting a scenario from available options."""
        if not available_options:
            return None

        query_lower = query.lower().strip()

        # Check for numeric selection (1, 2, 3, etc.)
        if query_lower.isdigit():
            idx = int(query_lower) - 1
            if 0 <= idx < len(available_options):
                return available_options[idx]

        # Check for letter selection (a, b, c, etc.)
        if len(query_lower) == 1 and query_lower.isalpha():
            idx = ord(query_lower) - ord("a")
            if 0 <= idx < len(available_options):
                return available_options[idx]

        # Check for keyword match with options
        for option in available_options:
            option_lower = option.lower()
            if option_lower in query_lower or query_lower in option_lower:
                return option

            # Check for significant word overlap
            option_words = set(option_lower.split())
            query_words = set(query_lower.split())
            overlap = option_words.intersection(query_words)
            if len(overlap) >= min(2, len(option_words) // 2 + 1):
                return option

        return None


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


class NotFoundResponseGenerator:
    """Generates appropriate responses when information is not found."""

    RESPONSES = {
        "general": [
            "I don't have information about that in my knowledge base. Could you try asking about a different topic?",
            "I couldn't find any relevant information about this topic. Is there something else I can help you with?",
            "Sorry, I don't have data about that. Would you like to ask about something else?",
        ],
        "partial": [
            "I found some related information, but nothing that directly answers your question. Would you like me to share what I found?",
        ],
        "suggest_rephrase": [
            "I couldn't find a match for your query. Could you try rephrasing or being more specific?",
        ],
    }

    @classmethod
    def generate(
        cls, query: str, search_analysis: dict, available_topics: List[str] = None
    ) -> str:
        quality = search_analysis.get("quality", SearchQuality.NOT_FOUND.value)
        confidence = search_analysis.get("confidence", 0)

        if quality == SearchQuality.LOW.value and confidence > 0.1:
            response = random.choice(cls.RESPONSES["partial"])
        elif confidence == 0:
            response = random.choice(cls.RESPONSES["general"])
        else:
            response = random.choice(cls.RESPONSES["suggest_rephrase"])

        if available_topics and len(available_topics) > 0:
            topic_list = ", ".join(available_topics[:5])
            response += f"\n\nI can help you with topics like: {topic_list}."

        return response


# ==================== Helper Functions ====================


def get_user_message(messages) -> str:
    """Extract latest user message."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


# ==================== Node Functions - EXACT copy from standalone ====================


def analyze_input(state: AgentState) -> dict:
    """Analyze user input - simplified to avoid premature clarification."""
    messages = state["messages"]

    user_message = get_user_message(messages)

    print(f"\n[analyze_input] Message: '{user_message}'")
    print(
        f"[analyze_input] awaiting_scenario_selection: {state.get('awaiting_scenario_selection', False)}"
    )
    print(
        f"[analyze_input] current_scenario_options: {state.get('current_scenario_options', [])}"
    )

    if not user_message:
        return {"interaction_mode": InteractionMode.QUERY.value}

    # Check if we're awaiting scenario selection
    if state.get("awaiting_scenario_selection", False):
        current_options = state.get("current_scenario_options", [])
        selected = QueryAnalyzer.is_scenario_selection(user_message, current_options)

        if selected:
            print(f"[analyze_input] User selected scenario: '{selected}'")
            return {
                "interaction_mode": InteractionMode.DISAMBIGUATION.value,
                "selected_scenario": selected,
                "awaiting_scenario_selection": False,
            }
        else:
            # User's response didn't match options - treat as new query
            print(f"[analyze_input] No match for options, treating as new query")
            return {
                "interaction_mode": InteractionMode.QUERY.value,
                "awaiting_scenario_selection": False,
                "selected_scenario": user_message,  # Use their response as context
            }

    # Check for greetings
    if QueryAnalyzer.is_greeting(user_message):
        print(f"[analyze_input] Greeting detected")
        return {"interaction_mode": InteractionMode.GREETING.value}

    # Check for closings
    if QueryAnalyzer.is_closing(user_message):
        print(f"[analyze_input] Closing detected")
        return {"interaction_mode": InteractionMode.CLOSING.value}

    # Check for too short
    if QueryAnalyzer.is_too_short(user_message):
        print(f"[analyze_input] Query too short")
        return {
            "interaction_mode": InteractionMode.CLARIFICATION.value,
            "clarification_needed": True,
            "follow_up_questions": [
                "Could you please tell me more about what you're looking for?"
            ],
        }

    # Default: proceed to search
    print(f"[analyze_input] Normal query - proceeding to agent")
    return {
        "interaction_mode": InteractionMode.QUERY.value,
        "original_query": user_message,
    }


def handle_greeting(state: AgentState) -> dict:
    """Handle greeting messages."""
    greetings = [
        "Hello! I'm here to help you find information. What would you like to know?",
        "Hi there! How can I assist you today?",
        "Hey! I can answer questions based on available documentation. What do you need?",
    ]
    return {"messages": [AIMessage(content=random.choice(greetings))]}


def handle_closing(state: AgentState) -> dict:
    """Handle closing messages."""
    closings = [
        "Goodbye! Feel free to come back if you have more questions.",
        "Happy to help! Take care!",
        "Glad I could assist! Have a great day!",
    ]
    return {"messages": [AIMessage(content=random.choice(closings))]}


def ask_clarification(state: AgentState) -> dict:
    """Ask for clarification for very short/unclear queries."""
    follow_up = state.get("follow_up_questions", ["What would you like to know?"])
    return {
        "messages": [AIMessage(content=follow_up[0])],
        "pending_clarification": True,
    }


def agent(state: AgentState) -> dict:
    """Main agent - processes queries by searching first."""
    messages = state["messages"]

    context_info = ""

    # Add context for scenario selection flow
    if (
        state.get("selected_scenario")
        and state.get("interaction_mode") == InteractionMode.DISAMBIGUATION.value
    ):
        context_info = f"\n\nUser selected scenario: {state['selected_scenario']}"
        context_info += f"\nOriginal query: {state.get('original_query', '')}"
        context_info += "\nCall `get_scenario_answer` with this information."

    system = SystemMessage(content=SYSTEM_PROMPT + context_info)
    response = llm_with_tools.invoke([system] + list(messages))

    return {"messages": [response], "has_searched": True}


def validate_and_route(state: AgentState) -> dict:
    """Validate search results and determine routing."""
    messages = state["messages"]

    print(f"\n[validate_and_route] Checking tool results...")

    # Find the last tool message
    last_tool_result = None
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            try:
                last_tool_result = json.loads(msg.content)
                print(
                    f"[validate_and_route] found_answer={last_tool_result.get('found_answer')}, "
                    f"disambiguation_needed={last_tool_result.get('disambiguation_needed')}"
                )
                break
            except:
                continue

    if last_tool_result is None:
        print(f"[validate_and_route] No tool result found")
        return {"should_respond_not_found": False}

    found_answer = last_tool_result.get("found_answer", False)
    disambiguation_needed = last_tool_result.get("disambiguation_needed", False)

    if not found_answer:
        # No relevant information found
        message = last_tool_result.get(
            "message", "I don't have information about that."
        )
        print(f"[validate_and_route] Not found: {message}")
        return {
            "should_respond_not_found": True,
            "not_found_message": message,
            "found_relevant_info": False,
        }

    if disambiguation_needed:
        # Multiple scenarios found in content
        scenarios = last_tool_result.get("scenarios", [])
        options = [s.get("title", f"Option {i+1}") for i, s in enumerate(scenarios)]
        question = last_tool_result.get("disambiguation_question", "")

        print(f"[validate_and_route] Disambiguation needed! Scenarios: {options}")

        if not question and scenarios:
            question = "I found information about multiple scenarios:\n\n"
            for i, s in enumerate(scenarios, 1):
                question += f"{i}. **{s.get('title', f'Option {i}')}**\n"
                if s.get("description"):
                    question += f"   {s.get('description')}\n"
            question += "\nWhich one applies to your situation?"

        return {
            "has_multiple_scenarios": True,
            "detected_scenarios": scenarios,
            "disambiguation_question": question,
            "awaiting_scenario_selection": True,
            "current_scenario_options": options,
            "should_respond_not_found": False,
            "search_results": json.dumps(last_tool_result),
        }

    # Single scenario or clear answer - proceed normally
    print(f"[validate_and_route] Single answer - proceeding")
    return {
        "should_respond_not_found": False,
        "found_relevant_info": True,
        "search_results": json.dumps(last_tool_result),
    }


def handle_not_found(state: AgentState) -> dict:
    """Handle case when no relevant information was found."""
    message = state.get(
        "not_found_message",
        "I don't have information about that in my knowledge base. "
        "Is there something else I can help you with?",
    )
    print(f"[handle_not_found] {message}")
    return {"messages": [AIMessage(content=message)]}


def present_scenarios(state: AgentState) -> dict:
    """Present scenario options when disambiguation is needed."""
    question = state.get(
        "disambiguation_question",
        "I found multiple related scenarios. Could you specify which one you're asking about?",
    )
    print(f"[present_scenarios] Asking: {question[:80]}...")
    return {
        "messages": [AIMessage(content=question)],
        "awaiting_scenario_selection": True,
    }


# ==================== Routing Functions ====================


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Determine if we should continue to tools or end."""
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print(f"[should_continue] -> tools")
        return "tools"
    print(f"[should_continue] -> end")
    return "end"


def route_after_analysis(
    state: AgentState,
) -> Literal["handle_greeting", "handle_closing", "ask_clarification", "agent"]:
    """Route based on analysis results."""
    mode = state.get("interaction_mode", InteractionMode.QUERY.value)

    print(f"[route_after_analysis] Mode: {mode}")

    if mode == InteractionMode.GREETING.value:
        return "handle_greeting"
    if mode == InteractionMode.CLOSING.value:
        return "handle_closing"
    if mode == InteractionMode.CLARIFICATION.value:
        return "ask_clarification"

    return "agent"


def route_after_validation(
    state: AgentState,
) -> Literal["handle_not_found", "agent", "present_scenarios"]:
    """Route based on search result validation."""
    if state.get("should_respond_not_found", False):
        print(f"[route_after_validation] -> handle_not_found")
        return "handle_not_found"
    if state.get("has_multiple_scenarios", False) and state.get(
        "awaiting_scenario_selection", False
    ):
        print(f"[route_after_validation] -> present_scenarios")
        return "present_scenarios"
    print(f"[route_after_validation] -> agent")
    return "agent"
