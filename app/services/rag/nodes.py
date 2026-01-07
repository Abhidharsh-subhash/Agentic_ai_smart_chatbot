import json
import re
import random
from typing import Literal
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode

from app.core.config import settings
from app.services.embeddings import embedding_service
from .state import AgentState
from .config import InteractionMode, SearchQuality, SupportMode, Config
from .analyzers import QueryAnalyzer, NotFoundResponseGenerator
from .scenario_handler import scenario_handler, DetectedScenario
from .response_generator import response_generator
from .tools import tools
from .memory import memory_manager


# Initialize LLMs
llm = ChatOpenAI(
    model=getattr(settings, "openai_model", "gpt-4o"),
    temperature=0.0,
    openai_api_key=settings.openai_api_key,
)

llm_with_tools = llm.bind_tools(tools)

# Tool node
tool_node = ToolNode(tools)


# ============================================
# SIMPLE AGENT PROMPT - Document only
# ============================================

AGENT_SYSTEM_PROMPT = """You are a support assistant. Your ONLY job is to search for information using the search_documents tool.

RULES:
1. ALWAYS use search_documents to find information
2. Do NOT answer from your own knowledge
3. After searching, the system will process the results

Search for: {query}"""


# ============================================
# NODE FUNCTIONS
# ============================================


def analyze_input(state: AgentState) -> dict:
    """Analyze user input - check for greetings, closings, and scenario selections."""
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

    # Check greetings
    if QueryAnalyzer.is_greeting(user_message):
        return {"interaction_mode": InteractionMode.GREETING.value}

    # Check closings
    if QueryAnalyzer.is_closing(user_message):
        return {"interaction_mode": InteractionMode.CLOSING.value}

    # Check if user is responding to a scenario clarification
    if memory.has_pending_clarification():
        pending = memory.pending_clarification

        # Convert scenarios to DetectedScenario objects
        scenarios = [
            DetectedScenario(
                id=s.get("id", str(i + 1)),
                title=s.get("title", ""),
                condition=s.get("condition", ""),
                description=s.get("description", ""),
                keywords=s.get("keywords", []),
            )
            for i, s in enumerate(pending.scenarios)
        ]

        # Try to match user response to a scenario
        matched = scenario_handler.match_user_response_to_scenario(
            user_message, scenarios
        )

        if matched:
            print(f"[analyze_input] User selected scenario: {matched}")
            return {
                "interaction_mode": InteractionMode.SCENARIO_SELECTION.value,
                "selected_scenario_id": matched,
                "original_query": pending.original_query,
                "detected_scenarios": pending.scenarios,
                "raw_search_documents": pending.raw_documents,
                "awaiting_scenario_selection": False,
                "support_mode": SupportMode.SCENARIO_RESPONSE.value,
            }

        # User didn't select a number - might be providing more context
        # Increment attempts
        memory.increment_clarification_attempt()

        if pending.attempts >= Config.MAX_CLARIFICATION_ATTEMPTS:
            # Give up, clear pending, treat as new query
            memory.clear_pending_clarification()
        else:
            # Re-ask for clarification
            return {
                "interaction_mode": InteractionMode.AWAITING_SCENARIO.value,
                "scenario_question": pending.clarification_question,
                "detected_scenarios": pending.scenarios,
                "awaiting_scenario_selection": True,
            }

    # Normal query - will go through search
    return {
        "interaction_mode": InteractionMode.QUERY.value,
        "original_query": user_message,
        "support_mode": SupportMode.DIRECT_ANSWER.value,
    }


def think_and_plan(state: AgentState) -> dict:
    """Simple planning - just prepare the search query."""
    messages = state["messages"]

    user_question = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_question = msg.content
            break

    return {
        "planned_search_queries": [user_question] if user_question else [],
        "original_query": user_question or "",
    }


def handle_greeting(state: AgentState) -> dict:
    """Handle greetings."""
    context = state.get("context", {})
    session_id = context.get("session_id", "default")
    memory = memory_manager.get_or_create(session_id)

    if len(memory.turns) > 0:
        response = "Welcome back! How can I help you?"
    else:
        response = (
            "Hello! I'm your support assistant. I'll help you find information "
            "from our documentation. If your question has multiple possible answers, "
            "I'll ask you to clarify your situation first. How can I help you today?"
        )

    memory.add_assistant_message(response)
    return {"messages": [AIMessage(content=response)]}


def handle_closing(state: AgentState) -> dict:
    """Handle closings."""
    context = state.get("context", {})
    session_id = context.get("session_id", "default")
    memory = memory_manager.get_or_create(session_id)

    memory.clear_pending_clarification()

    response = (
        "You're welcome! Feel free to ask if you have more questions. Goodbye! 👋"
    )
    memory.add_assistant_message(response)

    return {"messages": [AIMessage(content=response)]}


def agent(state: AgentState) -> dict:
    """Main agent - just searches, doesn't generate answers."""
    messages = state["messages"]
    original_query = state.get("original_query", "")

    if not original_query:
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                original_query = msg.content
                break

    system = SystemMessage(content=AGENT_SYSTEM_PROMPT.format(query=original_query))

    response = llm_with_tools.invoke([system] + list(messages))

    return {"messages": [response], "has_searched": True}


def validate_search_results(state: AgentState) -> dict:
    """
    Validate search results and detect scenarios.
    This is STATELESS for scenario detection - same input = same output.
    """
    messages = state["messages"]
    context = state.get("context", {})
    session_id = context.get("session_id", "default")

    print(f"\n{'='*60}")
    print(f"[validate_search_results] Session: {session_id}")
    print(f"{'='*60}")

    # Get tool result
    last_tool_result = None
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            try:
                last_tool_result = json.loads(msg.content)
                break
            except:
                continue

    if last_tool_result is None:
        print(f"[validate_search_results] No tool result")
        return {"should_respond_not_found": False}

    should_respond = last_tool_result.get("should_respond", False)
    quality = last_tool_result.get("quality", SearchQuality.NOT_FOUND.value)
    documents = last_tool_result.get("documents", [])

    print(f"[validate_search_results] Should Respond: {should_respond}")
    print(f"[validate_search_results] Quality: {quality}")
    print(f"[validate_search_results] Docs: {len(documents)}")

    # Not found
    if not should_respond or quality == SearchQuality.NOT_FOUND.value:
        return {
            "should_respond_not_found": True,
            "not_found_message": NotFoundResponseGenerator.generate(
                query=state.get("original_query", ""), search_analysis=last_tool_result
            ),
            "found_relevant_info": False,
            "has_multiple_scenarios": False,
        }

    original_query = state.get("original_query", "")
    if not original_query:
        for msg in messages:
            if isinstance(msg, HumanMessage):
                original_query = msg.content
                break

    # FRESH scenario analysis - ignores conversation history
    print(f"[validate_search_results] Running FRESH scenario analysis...")
    try:
        scenario_result = scenario_handler.analyze_for_scenarios_sync(
            original_query, documents, session_id=session_id
        )
        print(
            f"[validate_search_results] Has scenarios: {scenario_result.has_multiple_scenarios}"
        )
        print(
            f"[validate_search_results] Scenario count: {len(scenario_result.scenarios)}"
        )

    except Exception as e:
        print(f"[validate_search_results] Scenario error: {e}")
        import traceback

        traceback.print_exc()
        scenario_result = None

    # Multiple scenarios - always ask for clarification
    if scenario_result and scenario_result.has_multiple_scenarios:
        print(
            f"[validate_search_results] MULTIPLE SCENARIOS - asking for clarification"
        )

        memory = memory_manager.get_or_create(session_id)

        scenarios_dict = [s.to_dict() for s in scenario_result.scenarios]

        # Store for follow-up
        memory.set_pending_clarification(
            original_query=original_query,
            clarification_type="scenario",
            scenarios=scenarios_dict,
            raw_documents=documents,
            clarification_question=scenario_result.clarification_question or "",
        )

        return {
            "should_respond_not_found": False,
            "has_multiple_scenarios": True,
            "detected_scenarios": scenarios_dict,
            "scenario_question": scenario_result.clarification_question,
            "awaiting_scenario_selection": True,
            "raw_search_documents": documents,
            "found_relevant_info": True,
            "support_mode": SupportMode.SCENARIO_SELECTION.value,
        }

    # Direct answer
    print(f"[validate_search_results] DIRECT ANSWER path")
    return {
        "should_respond_not_found": False,
        "has_multiple_scenarios": False,
        "found_relevant_info": True,
        "raw_search_documents": documents,
        "support_mode": SupportMode.DIRECT_ANSWER.value,
    }


def ask_scenario_clarification(state: AgentState) -> dict:
    """Ask user to select a scenario."""
    context = state.get("context", {})
    session_id = context.get("session_id", "default")
    memory = memory_manager.get_or_create(session_id)

    # Use pre-generated clarification question
    clarification_question = state.get("scenario_question")

    if not clarification_question:
        scenarios = state.get("detected_scenarios", [])
        clarification_question = response_generator.generate_clarification_question(
            state.get("original_query", ""), scenarios
        )

    memory.add_assistant_message(clarification_question, turn_type="clarification")

    return {
        "messages": [AIMessage(content=clarification_question)],
        "awaiting_scenario_selection": True,
    }


def handle_scenario_selection(state: AgentState) -> dict:
    """Handle user's scenario selection - generate answer for specific scenario."""
    context = state.get("context", {})
    session_id = context.get("session_id", "default")
    memory = memory_manager.get_or_create(session_id)

    selected_id = state.get("selected_scenario_id")
    original_query = state.get("original_query", "")
    scenarios = state.get("detected_scenarios", [])
    documents = state.get("raw_search_documents", [])

    print(f"[handle_scenario_selection] Selected: {selected_id}")
    print(f"[handle_scenario_selection] Original query: {original_query}")

    # Clear pending
    memory.clear_pending_clarification()

    # Find selected scenario
    selected_scenario = None
    try:
        idx = int(selected_id) - 1
        if 0 <= idx < len(scenarios):
            selected_scenario = scenarios[idx]
    except:
        for s in scenarios:
            if s.get("id") == selected_id:
                selected_scenario = s
                break

    if not selected_scenario:
        return {
            "messages": [
                AIMessage(
                    content="I couldn't identify which option you selected. Could you please reply with just the number?"
                )
            ],
            "awaiting_scenario_selection": True,
        }

    # Generate answer for selected scenario - DOCUMENT ONLY
    answer = response_generator.generate_scenario_answer(
        original_question=original_query,
        selected_scenario=selected_scenario,
        documents=documents,
        session_id=session_id,
    )

    memory.add_assistant_message(answer, turn_type="scenario_response")

    return {
        "messages": [AIMessage(content=answer)],
        "awaiting_scenario_selection": False,
        "has_multiple_scenarios": False,
    }


def handle_awaiting_scenario(state: AgentState) -> dict:
    """Re-ask for scenario selection."""
    context = state.get("context", {})
    session_id = context.get("session_id", "default")
    memory = memory_manager.get_or_create(session_id)

    if memory.has_pending_clarification():
        question = memory.pending_clarification.clarification_question
        response = f"I need to know your specific situation to help you. {question}"
    else:
        response = "Could you please specify which option applies to you by replying with a number?"

    memory.add_assistant_message(response, turn_type="clarification")
    return {"messages": [AIMessage(content=response)]}


def generate_direct_answer(state: AgentState) -> dict:
    """Generate direct answer from documents - NO external knowledge."""
    context = state.get("context", {})
    session_id = context.get("session_id", "default")
    memory = memory_manager.get_or_create(session_id)

    documents = state.get("raw_search_documents", [])
    original_query = state.get("original_query", "")

    print(f"[generate_direct_answer] Query: {original_query}")
    print(f"[generate_direct_answer] Docs: {len(documents)}")

    # Use strict response generator
    answer = response_generator.generate_direct_answer(
        question=original_query, documents=documents, session_id=session_id
    )

    memory.add_assistant_message(answer)

    return {"messages": [AIMessage(content=answer)]}


def handle_not_found(state: AgentState) -> dict:
    """Handle not found."""
    context = state.get("context", {})
    session_id = context.get("session_id", "default")
    memory = memory_manager.get_or_create(session_id)

    memory.clear_pending_clarification()

    message = state.get("not_found_message") or (
        "I don't have information about that in my knowledge base."
    )

    memory.add_assistant_message(message)
    return {"messages": [AIMessage(content=message)]}


def save_to_memory(state: AgentState) -> dict:
    """Save to memory."""
    messages = state["messages"]
    context = state.get("context", {})
    session_id = context.get("session_id", "default")

    memory = memory_manager.get_or_create(session_id)

    for msg in messages:
        if isinstance(msg, HumanMessage):
            memory.add_user_message(msg.content)
        elif isinstance(msg, AIMessage) and msg.content:
            # Avoid duplicates
            last = memory.get_last_assistant_response()
            if last != msg.content:
                memory.add_assistant_message(msg.content)

    return {}


# ============================================
# ROUTING
# ============================================


def should_continue(state: AgentState) -> Literal["tools", "save_memory"]:
    """Route after agent."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "save_memory"


def route_after_analysis(state: AgentState) -> str:
    """Route after input analysis."""
    mode = state.get("interaction_mode", InteractionMode.QUERY.value)

    if mode == InteractionMode.GREETING.value:
        return "handle_greeting"

    if mode == InteractionMode.CLOSING.value:
        return "handle_closing"

    if mode == InteractionMode.SCENARIO_SELECTION.value:
        return "handle_scenario_selection"

    if mode == InteractionMode.AWAITING_SCENARIO.value:
        return "handle_awaiting_scenario"

    return "think_and_plan"


def route_after_validation(state: AgentState) -> str:
    """Route after search validation."""
    if state.get("should_respond_not_found", False):
        return "handle_not_found"

    if state.get("has_multiple_scenarios", False):
        return "ask_scenario_clarification"

    return "generate_direct_answer"
