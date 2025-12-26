import json
import random
from typing import Literal
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode

from app.core.config import settings
from .state import AgentState
from .config import InteractionMode, SearchQuality
from .analyzers import QueryAnalyzer, NotFoundResponseGenerator
from .tools import tools


# Initialize LLM
llm = ChatOpenAI(
    model=settings.openai_model if hasattr(settings, "openai_model") else "gpt-4o",
    temperature=0.1,
    openai_api_key=settings.openai_api_key,
)
llm_with_tools = llm.bind_tools(tools)


# System prompt
SYSTEM_PROMPT = """You are a document-based support assistant. You can ONLY provide information that exists in the search results.

## ABSOLUTE RULES:

### RULE 1: SEARCH FIRST
- Call `search_documents` for EVERY user question
- Wait for search results before responding

### RULE 2: ONLY USE SEARCH RESULTS
- You can ONLY use information from the `documents` array in search results
- NEVER use your general knowledge
- NEVER make up information

### RULE 3: CHECK `should_respond` FLAG
- If search returns `"should_respond": false` → DO NOT answer the question
- Instead, say you don't have information about this topic

### RULE 4: RESPONSE FORMAT
When you DO have information:
- Provide clear, direct answers
- Use only content from the documents

When you DON'T have information:
- Clearly state you don't have this information
- Suggest the user try a different question"""


# Tool node
tool_node = ToolNode(tools)


def analyze_input(state: AgentState) -> dict:
    """Analyze user input for intent and clarity."""
    messages = state["messages"]
    context = state.get("context", {})

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

    # Handle pending clarification
    if state.get("pending_clarification", False):
        return {
            "clarification_needed": False,
            "pending_clarification": False,
            "interaction_mode": InteractionMode.QUERY.value,
        }

    # Analyze query clarity
    analysis = QueryAnalyzer.analyze(user_message, context)
    needs_clarification = not analysis["is_clear"] and analysis["confidence"] < 0.3

    return {
        "clarification_needed": needs_clarification,
        "clarification_reason": analysis.get("clarification_type", ""),
        "follow_up_questions": analysis.get("follow_up_questions", []),
        "pending_clarification": needs_clarification,
        "original_query": user_message if needs_clarification else "",
        "search_confidence": analysis["confidence"],
        "interaction_mode": (
            InteractionMode.CLARIFICATION.value
            if needs_clarification
            else InteractionMode.QUERY.value
        ),
    }


def handle_greeting(state: AgentState) -> dict:
    """Handle greeting messages."""
    greetings = [
        "Hello! I'm your support assistant. I can answer questions based on the available documentation. How can I help you today?",
        "Hi there! I'm here to help you find information from our knowledge base. What would you like to know?",
    ]
    return {"messages": [AIMessage(content=random.choice(greetings))]}


def handle_closing(state: AgentState) -> dict:
    """Handle closing messages."""
    closings = [
        "You're welcome! Feel free to come back if you have more questions. Have a great day! 👋",
        "Happy to help! Don't hesitate to ask if anything else comes up. Take care!",
    ]
    return {"messages": [AIMessage(content=random.choice(closings))]}


def ask_clarification(state: AgentState) -> dict:
    """Ask for clarification when needed."""
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
    """Main agent node."""
    messages = state["messages"]
    topic_history = state.get("topic_history", [])

    context_info = ""
    if topic_history:
        context_info = f"\n\nRecent topics: {', '.join(topic_history[-3:])}"

    if state.get("original_query"):
        context_info += f"\nOriginal question: {state['original_query']}"

    system = SystemMessage(content=SYSTEM_PROMPT + context_info)
    response = llm_with_tools.invoke([system] + list(messages))

    return {"messages": [response], "has_searched": True}


def validate_search_results(state: AgentState) -> dict:
    """Validate search results and determine if we should respond."""
    messages = state["messages"]

    # Find the last tool message
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
    not_found_message = state.get("not_found_message")

    if not not_found_message:
        not_found_message = (
            "I don't have information about that topic in my knowledge base. "
            "Would you like to ask about something else?"
        )

    return {"messages": [AIMessage(content=not_found_message)]}


# Routing functions
def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Determine if we should continue to tools or end."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"


def route_after_analysis(
    state: AgentState,
) -> Literal["handle_greeting", "handle_closing", "ask_clarification", "agent"]:
    """Route based on analysis results."""
    mode = state.get("interaction_mode", InteractionMode.QUERY.value)

    if mode == InteractionMode.GREETING.value:
        return "handle_greeting"

    if mode == InteractionMode.CLOSING.value:
        return "handle_closing"

    if state.get("clarification_needed", False):
        attempts = state.get("clarification_attempts", 0)
        if attempts < 2:
            return "ask_clarification"

    return "agent"


def route_after_validation(state: AgentState) -> Literal["handle_not_found", "agent"]:
    """Route based on search result validation."""
    if state.get("should_respond_not_found", False):
        return "handle_not_found"
    return "agent"
