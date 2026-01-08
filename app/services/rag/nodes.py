# app/services/rag/nodes.py
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


# ============================================
# HELPER FUNCTIONS
# ============================================


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

    # Check if this is a scenario clarification response
    is_scenario_clarification = state.get("awaiting_scenario_selection", False)

    if is_scenario_clarification:
        # User is responding to a scenario clarification question
        return {
            "interaction_mode": InteractionMode.QUERY.value,
            "clarification_needed": False,
            "is_follow_up_question": True,
            "user_scenario_context": user_message,
            "scenario_clarification_pending": True,  # Flag to route to scenario matching
            "last_user_query": state.get("original_query", ""),
            "last_assistant_response": state.get("original_full_response", ""),
        }

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
            "awaiting_scenario_selection": False,
        }

    # Check for closings
    if QueryAnalyzer.is_closing(user_message):
        return {
            "interaction_mode": InteractionMode.CLOSING.value,
            "clarification_needed": False,
            "is_follow_up_question": False,
            "awaiting_scenario_selection": False,
        }

    # Handle pending clarification (from vague query, not scenarios)
    if state.get("pending_clarification", False) and not is_scenario_clarification:
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
        "awaiting_scenario_selection": False,
        "scenario_clarification_pending": False,
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
    NEW NODE: Analyze the agent's response for multiple scenarios.
    If multiple scenarios detected, prepare clarification flow.
    """
    messages = state["messages"]
    context = state.get("context", {})

    # Skip if this is already a scenario clarification response
    if state.get("scenario_clarification_pending", False):
        return {
            "needs_scenario_clarification": False,
        }

    # Get the last AI response
    last_ai_response = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            last_ai_response = msg.content
            break

    if not last_ai_response:
        return {"needs_scenario_clarification": False}

    # Get original user question
    user_question = None
    for msg in messages:
        if isinstance(msg, HumanMessage):
            user_question = msg.content
            break

    if not user_question:
        return {"needs_scenario_clarification": False}

    # Analyze the response for multiple scenarios
    try:
        analyzer = get_response_analyzer()
        analysis = analyzer.analyze_response(last_ai_response, user_question)

        if (
            analysis.get("has_multiple_scenarios", False)
            and analysis.get("scenario_count", 0) > 1
        ):
            scenarios = analysis.get("scenarios", [])
            clarification_q = analysis.get("clarification_question", "")
            options = analysis.get("clarification_options", [])

            print(
                f"[ScenarioAnalysis] Detected {len(scenarios)} scenarios, asking for clarification"
            )

            return {
                "needs_scenario_clarification": True,
                "response_analysis": analysis,
                "scenario_clarification_question": clarification_q,
                "scenario_options": options,
                "original_full_response": last_ai_response,
                "parsed_scenarios": scenarios,
                "awaiting_scenario_selection": True,
                "original_query": user_question,
            }
        else:
            print("[ScenarioAnalysis] Single scenario or direct answer detected")
            return {
                "needs_scenario_clarification": False,
                "response_analysis": analysis,
            }

    except Exception as e:
        print(f"[ScenarioAnalysis] Error: {e}")
        import traceback

        traceback.print_exc()
        return {"needs_scenario_clarification": False}


def ask_scenario_clarification(state: AgentState) -> dict:
    """
    NEW NODE: Ask the user to clarify which scenario applies to them.
    """
    context = state.get("context", {})
    session_id = context.get("session_id", "default")
    memory = memory_manager.get_or_create(session_id)

    original_question = state.get("original_query", "your question")
    scenarios = state.get("parsed_scenarios", [])
    clarification_question = state.get("scenario_clarification_question", "")
    options = state.get("scenario_options", [])

    # Generate a natural clarification message
    prompt = SCENARIO_CLARIFICATION_PROMPT.format(
        original_question=original_question,
        scenarios=json.dumps(scenarios, indent=2),
        clarification_question=clarification_question,
    )

    try:
        response = analyzer_llm.invoke([SystemMessage(content=prompt)])
        clarification_message = response.content.strip()
    except Exception as e:
        # Fallback message
        clarification_message = (
            f"I found several possible situations that might apply to your question. "
            f"{clarification_question}\n\n"
        )
        if options:
            clarification_message += "The main possibilities are:\n"
            for i, opt in enumerate(options[:5], 1):
                clarification_message += f"{i}. {opt}\n"
            clarification_message += "\nWhich one matches your situation?"

    # Save to memory
    user_msg = None
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            user_msg = msg.content
            break

    if user_msg:
        memory.add_user_message(user_msg, state.get("detected_topics", []))
    memory.add_assistant_message(clarification_message)

    return {
        "messages": [AIMessage(content=clarification_message)],
        "awaiting_scenario_selection": True,
        "pending_clarification": False,  # Different from vague query clarification
    }


def process_scenario_selection(state: AgentState) -> dict:
    """
    NEW NODE: Process user's scenario selection and provide focused response.
    """
    context = state.get("context", {})
    session_id = context.get("session_id", "default")
    memory = memory_manager.get_or_create(session_id)

    user_clarification = state.get("user_scenario_context", "")
    scenarios = state.get("parsed_scenarios", [])
    original_question = state.get("original_query", "")
    full_response = state.get("original_full_response", "")

    if not user_clarification or not scenarios:
        # No clarification provided, return full response
        return {
            "awaiting_scenario_selection": False,
            "scenario_clarification_pending": False,
        }

    # Match user's clarification to a scenario
    try:
        analyzer = get_response_analyzer()
        match_result = analyzer.match_user_to_scenario(
            scenarios, user_clarification, original_question
        )

        matched_id = match_result.get("matched_scenario_id")
        confidence = match_result.get("confidence", 0)
        needs_more = match_result.get("needs_more_info", False)

        if matched_id is not None and confidence > 0.5:
            # Find the matched scenario
            matched_scenario = None
            for s in scenarios:
                if s.get("id") == matched_id:
                    matched_scenario = s
                    break

            if matched_scenario:
                # Generate focused response
                focused_response = analyzer.generate_focused_response(
                    matched_scenario, original_question, user_clarification
                )

                # Save to memory
                memory.add_user_message(
                    user_clarification, state.get("detected_topics", [])
                )
                memory.add_assistant_message(focused_response)

                return {
                    "messages": [AIMessage(content=focused_response)],
                    "awaiting_scenario_selection": False,
                    "scenario_clarification_pending": False,
                    "selected_scenario_id": matched_id,
                }

        # Couldn't match - ask for more info or provide full response
        if needs_more:
            follow_up = (
                "I'm not quite sure which situation applies based on that. "
                "Could you provide a bit more detail about what's happening? "
                "For example, what error message are you seeing, or what steps led to this issue?"
            )
            memory.add_user_message(user_clarification)
            memory.add_assistant_message(follow_up)

            return {
                "messages": [AIMessage(content=follow_up)],
                "awaiting_scenario_selection": True,  # Keep waiting
                "scenario_clarification_pending": False,
            }
        else:
            # Provide the full response as fallback
            memory.add_user_message(user_clarification)
            memory.add_assistant_message(full_response)

            return {
                "messages": [AIMessage(content=full_response)],
                "awaiting_scenario_selection": False,
                "scenario_clarification_pending": False,
            }

    except Exception as e:
        print(f"[ScenarioSelection] Error: {e}")
        # Fallback - return full response
        memory.add_user_message(user_clarification)
        memory.add_assistant_message(full_response)

        return {
            "messages": [AIMessage(content=full_response)],
            "awaiting_scenario_selection": False,
            "scenario_clarification_pending": False,
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

    # Check if user is responding to scenario clarification
    if state.get("scenario_clarification_pending", False):
        return "process_scenario_selection"

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
