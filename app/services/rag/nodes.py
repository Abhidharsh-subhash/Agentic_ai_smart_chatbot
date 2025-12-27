import json
import random
from typing import Literal
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode

from app.core.config import settings
from app.services.embeddings import embedding_service
from .state import AgentState
from .config import InteractionMode, SearchQuality
from .analyzers import QueryAnalyzer, NotFoundResponseGenerator
from .tools import tools
from .memory import memory_manager


# Initialize LLMs
llm = ChatOpenAI(
    model=settings.openai_model if hasattr(settings, "openai_model") else "gpt-4o",
    temperature=0.1,
    openai_api_key=settings.openai_api_key,
)

# Separate LLM for thinking (can use cheaper model)
thinking_llm = ChatOpenAI(
    model=settings.openai_model if hasattr(settings, "openai_model") else "gpt-4o",
    temperature=0.0,
    openai_api_key=settings.openai_api_key,
)

llm_with_tools = llm.bind_tools(tools)


# ============================================
# SYSTEM PROMPTS
# ============================================

THINKING_PROMPT = """You are an analytical assistant. Your job is to understand the user's question and plan how to answer it.

## Conversation Context:
{conversation_context}

## Topics Discussed Previously:
{topics_discussed}

## Current User Question:
{user_question}

## Your Task:
Analyze this question step by step and respond in JSON format:

{{
    "understanding": "What the user is really asking about (be specific)",
    "is_follow_up": true/false (is this a follow-up to previous conversation?),
    "referenced_context": "If follow-up, what previous context is being referenced?",
    "key_topics": ["topic1", "topic2"],
    "search_queries": ["optimized search query 1", "optimized search query 2"],
    "reasoning": "Why these search queries will find the answer"
}}

Think carefully:
1. Is this a follow-up question referencing previous conversation?
2. What specific information does the user need?
3. What are the best search terms to find this information?
4. If it's a follow-up, incorporate context from previous discussion.

Respond ONLY with valid JSON."""


AGENT_SYSTEM_PROMPT = """You are a document-based support assistant with conversation memory.

## CONVERSATION CONTEXT:
{conversation_context}

## YOUR ANALYSIS (Chain of Thought):
{thinking_output}

## RULES:

### RULE 1: USE CONTEXT
- Remember the conversation history
- If this is a follow-up question, connect your answer to previous discussion
- Reference previous topics when relevant

### RULE 2: SEARCH FIRST
- Call `search_documents` for questions
- Use the optimized search queries from your analysis
- Wait for search results before responding

### RULE 3: ONLY USE SEARCH RESULTS
- You can ONLY use information from search results
- NEVER use general knowledge
- NEVER make up information

### RULE 4: CHECK `should_respond` FLAG
- If `"should_respond": false` → say you don't have information

### RULE 5: RESPONSE FORMAT
- Provide clear, direct answers
- If follow-up, connect to previous context
- Be conversational and helpful

### RULE 6: FOR FOLLOW-UP QUESTIONS
- Acknowledge the connection to previous discussion
- Use context from previous turns
- Provide coherent, contextual responses"""


# Tool node
tool_node = ToolNode(tools)


# ============================================
# NODE FUNCTIONS
# ============================================


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

    # Get memory for this session
    memory = memory_manager.get_or_create(session_id)

    # Check if this is a follow-up question
    is_follow_up = memory.is_likely_follow_up(user_message)

    # Store current query info
    last_user = memory.get_last_user_query()
    last_assistant = memory.get_last_assistant_response()

    # Check for greetings
    if QueryAnalyzer.is_greeting(user_message):
        return {
            "interaction_mode": InteractionMode.GREETING.value,
            "clarification_needed": False,
            "is_follow_up_question": False,
        }

    # Check for closings
    if QueryAnalyzer.is_closing(user_message):
        return {
            "interaction_mode": InteractionMode.CLOSING.value,
            "clarification_needed": False,
            "is_follow_up_question": False,
        }

    # Handle pending clarification
    if state.get("pending_clarification", False):
        return {
            "clarification_needed": False,
            "pending_clarification": False,
            "interaction_mode": InteractionMode.QUERY.value,
            "is_follow_up_question": True,
            "last_user_query": last_user or "",
            "last_assistant_response": last_assistant or "",
        }

    # Analyze query clarity
    analysis = QueryAnalyzer.analyze(user_message, context)
    needs_clarification = not analysis["is_clear"] and analysis["confidence"] < 0.3

    return {
        "clarification_needed": needs_clarification,
        "clarification_reason": analysis.get("clarification_type", ""),
        "follow_up_questions": analysis.get("follow_up_questions", []),
        "pending_clarification": needs_clarification,
        "original_query": user_message if needs_clarification else user_message,
        "search_confidence": analysis["confidence"],
        "interaction_mode": (
            InteractionMode.CLARIFICATION.value
            if needs_clarification
            else InteractionMode.QUERY.value
        ),
        "is_follow_up_question": is_follow_up,
        "last_user_query": last_user or "",
        "last_assistant_response": last_assistant or "",
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
            "planned_search_queries": [],
        }

    # Get conversation memory
    memory = memory_manager.get_or_create(session_id)
    conversation_context = memory.get_context_window(n_turns=4)
    topics_discussed = memory.get_topics_discussed()

    # Build thinking prompt
    prompt = THINKING_PROMPT.format(
        conversation_context=conversation_context,
        topics_discussed=topics_discussed,
        user_question=user_question,
    )

    try:
        response = thinking_llm.invoke([SystemMessage(content=prompt)])

        # Parse JSON response
        thinking_output = json.loads(response.content)

        # Validate required fields
        thinking = {
            "understanding": thinking_output.get("understanding", user_question),
            "key_topics": thinking_output.get("key_topics", []),
            "search_queries": thinking_output.get("search_queries", [user_question]),
            "reasoning": thinking_output.get("reasoning", ""),
            "is_follow_up": thinking_output.get("is_follow_up", False),
            "referenced_context": thinking_output.get("referenced_context", ""),
        }

        return {
            "thinking": thinking,
            "planned_search_queries": thinking["search_queries"],
            "detected_topics": thinking["key_topics"],
            "is_follow_up_question": thinking["is_follow_up"],
            "follow_up_context": thinking["referenced_context"] or "",
        }

    except (json.JSONDecodeError, Exception) as e:
        print(f"[CoT] Error in thinking: {e}")
        # Fallback - use original query
        return {
            "thinking": {
                "understanding": user_question,
                "key_topics": [],
                "search_queries": [user_question],
                "reasoning": "Direct search",
                "is_follow_up": state.get("is_follow_up_question", False),
                "referenced_context": "",
            },
            "planned_search_queries": [user_question],
        }


def handle_greeting(state: AgentState) -> dict:
    """Handle greeting messages."""
    context = state.get("context", {})
    session_id = context.get("session_id", "default")
    memory = memory_manager.get_or_create(session_id)

    # Check if returning user
    if len(memory.turns) > 0:
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

    # Add to memory
    memory.add_assistant_message(response)

    return {"messages": [AIMessage(content=response)]}


def handle_closing(state: AgentState) -> dict:
    """Handle closing messages."""
    context = state.get("context", {})
    session_id = context.get("session_id", "default")
    memory = memory_manager.get_or_create(session_id)

    # Personalized closing based on conversation
    if len(memory.turns) > 4:
        topics = memory.get_topics_discussed()
        closings = [
            f"You're welcome! We covered a lot about {topics}. Feel free to come back anytime! 👋",
            "Happy to help! Don't hesitate to ask if anything else comes up. Take care!",
        ]
    else:
        closings = [
            "You're welcome! Feel free to come back if you have more questions. Have a great day! 👋",
            "Happy to help! Don't hesitate to ask if anything else comes up. Take care!",
        ]

    response = random.choice(closings)
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
                "There are currently no documents in the knowledge base. "
                "Please upload some documents first to start asking questions."
            )
        else:
            files = result.get("files", [])
            total_files = result.get("total_files", 0)
            total_chunks = result.get("total_chunks", 0)

            message = f"📂 **Available Documents in Knowledge Base**\n\n"
            message += f"**Total Files:** {total_files}\n"
            message += f"**Total Indexed Chunks:** {total_chunks}\n\n"
            message += "**Documents:**\n"

            files_by_type = {}
            for file in files:
                file_type = file.get("file_type", "unknown").upper()
                if file_type not in files_by_type:
                    files_by_type[file_type] = []
                files_by_type[file_type].append(file)

            for file_type, type_files in files_by_type.items():
                message += f"\n**{file_type} Files:**\n"
                for file in type_files:
                    filename = file.get("filename", "Unknown")
                    chunk_count = file.get("chunk_count", 0)
                    message += f"  • {filename} ({chunk_count} chunks)\n"

            message += "\n💡 You can ask questions about any of these documents!"

        # Add to memory
        memory.add_assistant_message(message, topics=["documents", "knowledge base"])

        return {"messages": [AIMessage(content=message)]}

    except Exception as e:
        print(f"Error listing documents: {e}")
        import traceback

        traceback.print_exc()

        return {
            "messages": [
                AIMessage(
                    content="I encountered an error while retrieving the document list. Please try again."
                )
            ]
        }


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
    """Main agent node with CoT context."""
    messages = state["messages"]
    context = state.get("context", {})
    session_id = context.get("session_id", "default")
    topic_history = state.get("topic_history", [])
    thinking = state.get("thinking", {})

    # Get conversation memory
    memory = memory_manager.get_or_create(session_id)
    conversation_context = memory.get_context_window(n_turns=4)

    # Format thinking output for prompt
    if thinking:
        thinking_str = f"""
Understanding: {thinking.get('understanding', 'N/A')}
Is Follow-up: {thinking.get('is_follow_up', False)}
Referenced Context: {thinking.get('referenced_context', 'None')}
Key Topics: {', '.join(thinking.get('key_topics', []))}
Search Strategy: {thinking.get('reasoning', 'Direct search')}
Planned Queries: {', '.join(thinking.get('search_queries', []))}
"""
    else:
        thinking_str = "No prior analysis available."

    # Build system prompt with context
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
        # Check if follow-up and provide contextual response
        if state.get("is_follow_up_question"):
            not_found_message = (
                "I don't have additional information about that in my knowledge base. "
                "Would you like to ask about a different aspect, or try a different topic?"
            )
        else:
            not_found_message = (
                "I don't have information about that topic in my knowledge base. "
                "Would you like to ask about something else?"
            )

    # Add to memory
    memory.add_assistant_message(not_found_message)

    return {"messages": [AIMessage(content=not_found_message)]}


def save_to_memory(state: AgentState) -> dict:
    """Save the conversation turn to memory."""
    messages = state["messages"]
    context = state.get("context", {})
    session_id = context.get("session_id", "default")
    detected_topics = state.get("detected_topics", [])

    memory = memory_manager.get_or_create(session_id)

    # Find user message and assistant response
    user_msg = None
    assistant_msg = None

    for msg in messages:
        if isinstance(msg, HumanMessage):
            user_msg = msg.content
        elif isinstance(msg, AIMessage) and msg.content:
            assistant_msg = msg.content

    # Save to memory
    if user_msg:
        memory.add_user_message(user_msg, detected_topics)
    if assistant_msg:
        memory.add_assistant_message(assistant_msg, detected_topics)

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


def route_after_analysis(
    state: AgentState,
) -> Literal[
    "handle_greeting",
    "handle_closing",
    "handle_document_listing",
    "ask_clarification",
    "think_and_plan",
]:
    """Route based on analysis results."""
    mode = state.get("interaction_mode", InteractionMode.QUERY.value)

    if mode == InteractionMode.GREETING.value:
        return "handle_greeting"

    if mode == InteractionMode.CLOSING.value:
        return "handle_closing"

    if mode == "document_listing":
        return "handle_document_listing"

    if state.get("clarification_needed", False):
        attempts = state.get("clarification_attempts", 0)
        if attempts < 2:
            return "ask_clarification"

    # Go to Chain of Thought first
    return "think_and_plan"


def route_after_validation(state: AgentState) -> Literal["handle_not_found", "agent"]:
    """Route based on search result validation."""
    if state.get("should_respond_not_found", False):
        return "handle_not_found"
    return "agent"
