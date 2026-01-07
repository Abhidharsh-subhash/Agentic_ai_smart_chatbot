import json
import re
import random
import asyncio
from typing import Literal, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode

from app.core.config import settings
from app.services.embeddings import embedding_service
from .state import AgentState
from .config import InteractionMode, SearchQuality, SupportMode, Config
from .analyzers import QueryAnalyzer, NotFoundResponseGenerator
from .scenario_handler import scenario_handler, DetectedScenario
from .tools import tools
from .memory import memory_manager
from .scenario_handler import scenario_handler, DetectedScenario

# Initialize LLMs
llm = ChatOpenAI(
    model=getattr(settings, "openai_model", "gpt-4o"),
    temperature=0.0,
    openai_api_key=settings.openai_api_key,
)

thinking_llm = ChatOpenAI(
    model=getattr(settings, "openai_model", "gpt-4o"),
    temperature=0.0,
    openai_api_key=settings.openai_api_key,
)

llm_with_tools = llm.bind_tools(tools)

# Tool node
tool_node = ToolNode(tools)


# ============================================
# SYSTEM PROMPTS
# ============================================

THINKING_PROMPT = """Analyze the user's question and plan how to answer.

CONVERSATION CONTEXT:
{conversation_context}

PENDING SCENARIO: {pending_scenario}

CURRENT QUESTION: {user_question}

Respond ONLY with JSON:
{{"understanding": "what user asks", "is_follow_up": false, "is_scenario_selection": false, "search_queries": ["query1"], "reasoning": "why"}}"""


DIRECT_ANSWER_PROMPT = """You are a support assistant. Answer the user's question using ONLY the provided documents.

DOCUMENTS:
{documents}

USER QUESTION: {question}

RULES:
1. Use ONLY information from the documents
2. Be helpful and conversational
3. If the documents don't fully answer the question, say so
4. Don't add warnings or advice not in the documents

ANSWER:"""


SCENARIO_ANSWER_PROMPT = """Answer for the specific scenario the user selected.

ORIGINAL QUESTION: {original_query}
SELECTED SCENARIO: {selected_scenario}

RELEVANT DOCUMENTS:
{documents}

Provide the answer that applies specifically to this scenario.
Be detailed and helpful. Use ONLY document information.

ANSWER:"""


# ============================================
# NODE FUNCTIONS
# ============================================


def analyze_input(state: AgentState) -> dict:
    """Analyze user input for intent and check for scenario selection."""
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

    # Check for greetings
    if QueryAnalyzer.is_greeting(user_message):
        return {
            "interaction_mode": InteractionMode.GREETING.value,
            "clarification_needed": False,
        }

    # Check for closings
    if QueryAnalyzer.is_closing(user_message):
        return {
            "interaction_mode": InteractionMode.CLOSING.value,
            "clarification_needed": False,
        }

    # Check if we have a pending scenario and user is responding to it
    if memory.has_pending_scenario():
        pending = memory.pending_scenario

        # Try to match user response to a scenario
        scenarios_as_detected = [
            DetectedScenario(
                id=s.get("id", str(i + 1)),
                title=s.get("title", ""),
                condition=s.get("condition", ""),
                description=s.get("description", ""),
                keywords=s.get("keywords", []),
            )
            for i, s in enumerate(pending.scenarios)
        ]

        matched = scenario_handler.match_user_response_to_scenario(
            user_message, scenarios_as_detected
        )

        if matched:
            # User selected a scenario
            return {
                "interaction_mode": InteractionMode.SCENARIO_SELECTION.value,
                "selected_scenario_id": matched,
                "original_query": pending.original_query,
                "detected_scenarios": pending.scenarios,
                "raw_search_documents": pending.raw_documents,
                "awaiting_scenario_selection": False,
                "support_mode": SupportMode.SCENARIO_RESPONSE.value,
            }

        # User might be providing more context instead of selecting
        # Increment attempts and check if we should give up
        memory.increment_scenario_attempt()

        if pending.attempts >= Config.MAX_CLARIFICATION_ATTEMPTS:
            # Give up on clarification, try to answer with what we have
            memory.clear_pending_scenario()
            return {
                "interaction_mode": InteractionMode.QUERY.value,
                "original_query": f"{pending.original_query} - Context: {user_message}",
                "support_mode": SupportMode.DIRECT_ANSWER.value,
            }

        # Re-ask for clarification
        return {
            "interaction_mode": InteractionMode.AWAITING_SCENARIO.value,
            "scenario_question": pending.clarification_question,
            "detected_scenarios": pending.scenarios,
            "awaiting_scenario_selection": True,
        }

    # Standard query analysis
    analysis = QueryAnalyzer.analyze(user_message, context)

    return {
        "clarification_needed": not analysis["is_clear"],
        "interaction_mode": InteractionMode.QUERY.value,
        "original_query": user_message,
        "support_mode": SupportMode.DIRECT_ANSWER.value,
    }


def think_and_plan(state: AgentState) -> dict:
    """Analyze question and plan search strategy."""
    messages = state["messages"]
    context = state.get("context", {})
    session_id = context.get("session_id", "default")

    user_question = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_question = msg.content
            break

    if not user_question:
        return {"thinking": None, "planned_search_queries": []}

    memory = memory_manager.get_or_create(session_id)
    conversation_context = memory.get_context_window(n_turns=4)

    pending_info = "None"
    if memory.has_pending_scenario():
        pending_info = f"Query: {memory.pending_scenario.original_query}"

    prompt = THINKING_PROMPT.format(
        conversation_context=conversation_context,
        pending_scenario=pending_info,
        user_question=user_question,
    )

    try:
        response = thinking_llm.invoke([SystemMessage(content=prompt)])
        result = _parse_json_response(response.content)

        thinking = {
            "understanding": result.get("understanding", user_question),
            "search_queries": result.get("search_queries", [user_question]),
            "reasoning": result.get("reasoning", ""),
            "is_follow_up": result.get("is_follow_up", False),
            "is_scenario_selection": result.get("is_scenario_selection", False),
        }

        return {
            "thinking": thinking,
            "planned_search_queries": thinking["search_queries"] or [user_question],
        }
    except Exception as e:
        print(f"Thinking error: {e}")
        return {
            "thinking": {"search_queries": [user_question]},
            "planned_search_queries": [user_question],
        }


def handle_greeting(state: AgentState) -> dict:
    """Handle greeting messages."""
    context = state.get("context", {})
    session_id = context.get("session_id", "default")
    memory = memory_manager.get_or_create(session_id)

    if len(memory.turns) > 0:
        response = "Welcome back! How can I help you today?"
    else:
        response = (
            "Hello! I'm your support assistant. Ask me anything, and I'll find "
            "the right answer for you. If there are different options depending "
            "on your situation, I'll ask you to clarify. How can I help?"
        )

    memory.add_assistant_message(response)
    return {"messages": [AIMessage(content=response)]}


def handle_closing(state: AgentState) -> dict:
    """Handle closing messages."""
    context = state.get("context", {})
    session_id = context.get("session_id", "default")
    memory = memory_manager.get_or_create(session_id)

    memory.clear_pending_scenario()

    response = random.choice(
        [
            "You're welcome! Feel free to come back anytime. 👋",
            "Happy to help! Take care!",
        ]
    )

    memory.add_assistant_message(response)
    return {"messages": [AIMessage(content=response)]}


def agent(state: AgentState) -> dict:
    """Main agent node - calls tools for search."""
    messages = state["messages"]
    context = state.get("context", {})
    session_id = context.get("session_id", "default")

    memory = memory_manager.get_or_create(session_id)
    conversation_context = memory.get_context_window(n_turns=3)

    system_prompt = f"""You are a support assistant. Search for information to answer user questions.

CONVERSATION CONTEXT:
{conversation_context}

INSTRUCTIONS:
1. Use search_documents tool to find relevant information
2. Provide answers ONLY from search results
3. If multiple scenarios exist, the system will handle clarification

Always search first before answering."""

    response = llm_with_tools.invoke(
        [SystemMessage(content=system_prompt)] + list(messages)
    )

    return {"messages": [response], "has_searched": True}


def validate_search_results(state: AgentState) -> dict:
    """Validate search results and check for multiple scenarios using LLM."""
    messages = state["messages"]
    context = state.get("context", {})
    session_id = context.get("session_id", "default")

    # Get last tool result
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
    documents = last_tool_result.get("documents", [])

    # Not found case
    if not should_respond or quality == SearchQuality.NOT_FOUND.value:
        not_found_msg = NotFoundResponseGenerator.generate(
            query=state.get("original_query", ""), search_analysis=last_tool_result
        )
        return {
            "should_respond_not_found": True,
            "not_found_message": not_found_msg,
            "found_relevant_info": False,
            "has_multiple_scenarios": False,
        }

    # Get original query
    original_query = state.get("original_query", "")
    if not original_query:
        for msg in messages:
            if isinstance(msg, HumanMessage):
                original_query = msg.content
                break

    # Run SYNCHRONOUS scenario analysis
    try:
        scenario_result = scenario_handler.analyze_for_scenarios_sync(
            original_query, documents
        )
    except Exception as e:
        print(f"Scenario analysis error: {e}")
        import traceback

        traceback.print_exc()
        scenario_result = None

    # Handle scenario result
    if scenario_result and scenario_result.has_multiple_scenarios:
        memory = memory_manager.get_or_create(session_id)

        # Store pending scenario
        scenarios_dict = [s.to_dict() for s in scenario_result.scenarios]

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

    # Direct answer path
    return {
        "should_respond_not_found": False,
        "has_multiple_scenarios": False,
        "found_relevant_info": True,
        "raw_search_documents": documents,
        "support_mode": SupportMode.DIRECT_ANSWER.value,
        "direct_answer": scenario_result.direct_answer if scenario_result else None,
    }


def ask_scenario_clarification(state: AgentState) -> dict:
    """Ask user to select which scenario applies."""
    context = state.get("context", {})
    session_id = context.get("session_id", "default")
    memory = memory_manager.get_or_create(session_id)

    clarification_question = state.get("scenario_question")
    scenarios = state.get("detected_scenarios", [])

    if not clarification_question:
        # Generate default question
        options = "\n".join(
            [
                f"{i+1}. {s.get('condition') or s.get('title', f'Option {i+1}')}"
                for i, s in enumerate(scenarios[:5])
            ]
        )
        clarification_question = (
            f"I found information for different situations. Which applies to you?\n\n"
            f"{options}\n\n"
            f"Reply with a number or describe your situation."
        )

    memory.add_assistant_message(clarification_question, turn_type="clarification")

    return {
        "messages": [AIMessage(content=clarification_question)],
        "awaiting_scenario_selection": True,
    }


def handle_scenario_selection(state: AgentState) -> dict:
    """Handle when user selects a specific scenario."""
    context = state.get("context", {})
    session_id = context.get("session_id", "default")
    memory = memory_manager.get_or_create(session_id)

    selected_id = state.get("selected_scenario_id")
    original_query = state.get("original_query", "")
    scenarios = state.get("detected_scenarios", [])
    documents = state.get("raw_search_documents", [])

    # Clear pending scenario
    memory.clear_pending_scenario()

    # Find selected scenario
    selected_scenario = None
    try:
        idx = int(selected_id) - 1
        if 0 <= idx < len(scenarios):
            selected_scenario = scenarios[idx]
    except (ValueError, TypeError):
        for s in scenarios:
            if s.get("id") == selected_id:
                selected_scenario = s
                break

    if not selected_scenario:
        return {
            "messages": [
                AIMessage(
                    content="I couldn't identify which option you selected. Could you please try again?"
                )
            ],
            "awaiting_scenario_selection": True,
        }

    # Generate answer for selected scenario
    docs_text = "\n\n---\n\n".join([d.get("content", "") for d in documents[:3]])

    prompt = SCENARIO_ANSWER_PROMPT.format(
        original_query=original_query,
        selected_scenario=f"{selected_scenario.get('title', '')}: {selected_scenario.get('condition', '')}",
        documents=docs_text[:3000],
    )

    try:
        response = llm.invoke([SystemMessage(content=prompt)])
        answer = response.content.strip()
    except Exception as e:
        print(f"Scenario answer error: {e}")
        answer = "I had trouble generating the specific answer. Please try rephrasing your question."

    # Store context for follow-ups
    memory.set_scenario_context(
        {"scenario": selected_scenario, "original_query": original_query}
    )

    memory.add_assistant_message(answer, turn_type="scenario_response")

    return {
        "messages": [AIMessage(content=answer)],
        "awaiting_scenario_selection": False,
        "has_multiple_scenarios": False,
    }


def generate_direct_answer(state: AgentState) -> dict:
    """Generate a direct answer from documents when no scenarios detected."""
    context = state.get("context", {})
    session_id = context.get("session_id", "default")
    memory = memory_manager.get_or_create(session_id)

    documents = state.get("raw_search_documents", [])
    original_query = state.get("original_query", "")

    # Check if we already have a direct answer from scenario analysis
    direct_answer = state.get("direct_answer")

    if direct_answer and len(direct_answer) > 50:
        # Use the pre-generated direct answer
        answer = direct_answer
    else:
        # Generate answer from documents
        docs_text = "\n\n---\n\n".join([d.get("content", "") for d in documents[:3]])

        prompt = DIRECT_ANSWER_PROMPT.format(
            documents=docs_text[:3000], question=original_query
        )

        try:
            response = llm.invoke([SystemMessage(content=prompt)])
            answer = response.content.strip()
        except Exception as e:
            print(f"Direct answer error: {e}")
            answer = "I found relevant information but had trouble formulating the answer. Please try again."

    memory.add_assistant_message(answer)
    return {"messages": [AIMessage(content=answer)]}


def handle_not_found(state: AgentState) -> dict:
    """Handle case when no relevant information was found."""
    context = state.get("context", {})
    session_id = context.get("session_id", "default")
    memory = memory_manager.get_or_create(session_id)

    memory.clear_pending_scenario()

    not_found_message = state.get("not_found_message") or (
        "I don't have information about that. Could you try rephrasing or ask about something else?"
    )

    memory.add_assistant_message(not_found_message)
    return {"messages": [AIMessage(content=not_found_message)]}


def handle_awaiting_scenario(state: AgentState) -> dict:
    """Handle when we're still waiting for scenario selection."""
    context = state.get("context", {})
    session_id = context.get("session_id", "default")
    memory = memory_manager.get_or_create(session_id)

    # Re-ask the clarification question
    if memory.has_pending_scenario():
        question = memory.pending_scenario.clarification_question

        response = (
            f"I'm not sure which option applies to you. {question}\n\n"
            f"Or you can rephrase your question with more details."
        )
    else:
        response = "Could you please provide more details about your situation?"

    memory.add_assistant_message(response, turn_type="clarification")
    return {"messages": [AIMessage(content=response)]}


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
    if assistant_msg and not any(
        isinstance(m, AIMessage) and m.content == assistant_msg
        for m in state.get("messages", [])[-3:]
    ):
        memory.add_assistant_message(assistant_msg, detected_topics)

    return {}


# ============================================
# HELPER FUNCTIONS
# ============================================


def _parse_json_response(response: str) -> dict:
    """Parse JSON from LLM response."""
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", response)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {}


# ============================================
# ROUTING FUNCTIONS
# ============================================


def should_continue(state: AgentState) -> Literal["tools", "save_memory"]:
    """Determine if we should continue to tools or end."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "save_memory"


def route_after_analysis(state: AgentState) -> str:
    """Route based on analysis results."""
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
    """Route based on search result validation."""
    if state.get("should_respond_not_found", False):
        return "handle_not_found"

    if state.get("has_multiple_scenarios", False):
        return "ask_scenario_clarification"

    return "generate_direct_answer"
