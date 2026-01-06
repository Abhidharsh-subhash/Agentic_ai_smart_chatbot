import json
import re
import random
from typing import Literal, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode

from app.core.config import settings
from app.services.embeddings import embedding_service
from .state import AgentState
from .config import InteractionMode, SearchQuality, SupportMode, Config
from .analyzers import QueryAnalyzer, NotFoundResponseGenerator, ScenarioDetector
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

llm_with_tools = llm.bind_tools(tools)


# ============================================
# SYSTEM PROMPTS
# ============================================

THINKING_PROMPT = """Analyze the user's question and plan how to search for the answer.

Conversation Context:
{conversation_context}

Topics Discussed:
{topics_discussed}

Pending Clarification: {pending_clarification}

Current Question:
{user_question}

Respond with ONLY a JSON object:
{{"understanding": "what the user is asking", "is_follow_up": false, "is_scenario_response": false, "referenced_context": "", "key_topics": [], "search_queries": ["query1"], "reasoning": "why these queries"}}"""


AGENT_SYSTEM_PROMPT = """You are a SUPPORT AGENT assistant. You help users by finding information and asking clarifying questions when needed.

## CONVERSATION CONTEXT:
{conversation_context}

## YOUR ANALYSIS:
{thinking_output}

## SUPPORT AGENT RULES:

### RULE 1: ASK FOR CLARIFICATION WHEN NEEDED
- If search results show MULTIPLE SCENARIOS or CONDITIONS, ASK the user which applies to them
- Use the clarification_question from the search results
- Present options clearly numbered (1, 2, 3, etc.)
- Wait for user to specify before giving a detailed answer

### RULE 2: DOCUMENT-ONLY RESPONSES
- ONLY provide information from the search results
- Do NOT add information from general knowledge
- Do NOT add warnings or advice not in documents

### RULE 3: SCENARIO HANDLING
When `has_multiple_scenarios` is true in search results:
1. DO NOT try to answer all scenarios
2. Present the clarification question to the user
3. Ask them to choose which situation applies

### RULE 4: SEARCH FIRST
- Call `search_documents` with the question
- Check `has_multiple_scenarios` field
- If true → ask for clarification
- If false and `should_respond: true` → provide answer

### RULE 5: BE CONVERSATIONAL
- Acknowledge the user's situation
- Be helpful and patient
- If they selected a scenario, provide the specific answer for that scenario

### EXAMPLE FLOW:
User: "How do I reset my password?"
Search Result: has_multiple_scenarios: true (different methods for admin vs regular user)
You: "I can help with password reset! To give you the right steps, could you tell me:
1. Are you an admin user?
2. Are you a regular user?
3. Are you locked out of your account?
Please reply with the number or describe your situation."

User: "2"
You: [Provide specific steps for regular user password reset from documents]

Remember: When in doubt, ASK for clarification rather than guessing!"""


SCENARIO_RESPONSE_PROMPT = """You are processing a user's scenario selection.

Original Question: {original_query}
User Selected: Scenario {scenario_id}

Available Scenarios:
{scenarios}

Relevant Documents:
{documents}

Provide ONLY the answer that applies to the selected scenario {scenario_id}.
Be specific and use only information from the documents.
Do NOT mention other scenarios unless directly relevant."""


# Tool node
tool_node = ToolNode(tools)


# ============================================
# HELPER FUNCTIONS
# ============================================


def extract_json_from_response(response_text: str) -> dict:
    """Extract JSON from LLM response."""
    if not response_text:
        raise ValueError("Empty response")

    text = response_text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

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

    json_match = re.search(r"\{[^{}]*\}", text)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

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


# ============================================
# NODE FUNCTIONS
# ============================================


def analyze_input(state: AgentState) -> dict:
    """Analyze user input for intent, clarity, and scenario selection."""
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

    # Analyze the query
    analysis = QueryAnalyzer.analyze(user_message, context)

    # Check if this is a scenario selection response
    if analysis.get("is_scenario_selection") and memory.has_pending_clarification():
        pending = memory.pending_clarification
        selected = analysis.get("selected_scenario")

        return {
            "interaction_mode": InteractionMode.SCENARIO_SELECTION.value,
            "selected_scenario_id": selected,
            "original_query": pending.original_query,
            "detected_scenarios": pending.scenarios,
            "raw_search_documents": pending.raw_documents,
            "awaiting_scenario_selection": False,
            "support_mode": SupportMode.FOLLOW_UP.value,
        }

    # Check if we're waiting for scenario selection but got a descriptive response
    if memory.has_pending_clarification():
        pending = memory.pending_clarification

        # Check if the response might be describing a scenario
        if pending.clarification_type == "scenario":
            # Try to match the response to a scenario
            matched_scenario = _match_response_to_scenario(
                user_message, pending.scenarios
            )
            if matched_scenario:
                return {
                    "interaction_mode": InteractionMode.SCENARIO_SELECTION.value,
                    "selected_scenario_id": matched_scenario,
                    "original_query": pending.original_query,
                    "detected_scenarios": pending.scenarios,
                    "raw_search_documents": pending.raw_documents,
                    "awaiting_scenario_selection": False,
                    "support_mode": SupportMode.FOLLOW_UP.value,
                }

            # Increment attempt and check if we should give up
            memory.increment_clarification_attempt()
            if pending.attempts >= Config.MAX_CLARIFICATION_ATTEMPTS:
                memory.clear_pending_clarification()
                # Proceed with best effort answer

    # Standard query analysis
    is_follow_up = memory.is_likely_follow_up(user_message)
    last_user = memory.get_last_user_query()
    last_assistant = memory.get_last_assistant_response()

    # Check for vague queries
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
        "support_mode": SupportMode.DIRECT_ANSWER.value,
    }


def _match_response_to_scenario(response: str, scenarios: list) -> Optional[str]:
    """Try to match a descriptive response to one of the scenarios."""
    response_lower = response.lower()

    best_match = None
    best_score = 0

    for scenario in scenarios:
        condition = scenario.get("condition", "").lower()
        description = scenario.get("description", "").lower()

        # Calculate word overlap
        response_words = set(response_lower.split())
        condition_words = set(condition.split())
        description_words = set(description.split())

        overlap = len(response_words & (condition_words | description_words))

        if overlap > best_score and overlap >= 2:  # At least 2 words match
            best_score = overlap
            best_match = scenario.get("id")

    return best_match


def think_and_plan(state: AgentState) -> dict:
    """Chain of Thought: Analyze question and plan search strategy."""
    messages = state["messages"]
    context = state.get("context", {})
    session_id = context.get("session_id", "default")

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

    memory = memory_manager.get_or_create(session_id)
    conversation_context = memory.get_context_window(n_turns=4)
    topics_discussed = memory.get_topics_discussed()

    # Check for pending clarification
    pending_info = "None"
    if memory.has_pending_clarification():
        pending = memory.pending_clarification
        pending_info = f"Type: {pending.clarification_type}, Original Query: {pending.original_query}"

    prompt = THINKING_PROMPT.format(
        conversation_context=conversation_context,
        topics_discussed=topics_discussed,
        pending_clarification=pending_info,
        user_question=user_question,
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
            "is_scenario_response": thinking_output.get("is_scenario_response", False),
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
                "is_scenario_response": False,
                "referenced_context": "",
            },
            "planned_search_queries": [user_question],
        }


def handle_greeting(state: AgentState) -> dict:
    """Handle greeting messages."""
    context = state.get("context", {})
    session_id = context.get("session_id", "default")
    memory = memory_manager.get_or_create(session_id)

    if len(memory.turns) > 0:
        greetings = [
            "Welcome back! How can I help you today?",
            "Hello again! What would you like to know?",
        ]
    else:
        greetings = [
            "Hello! I'm your support assistant. I'll help you find the information you need. "
            "If your question has multiple possible answers, I'll ask you some clarifying questions "
            "to make sure I give you the most relevant response. How can I help you today?",
            "Hi there! I'm here to help. Feel free to ask me anything, and I'll make sure to "
            "understand your specific situation before providing an answer. What would you like to know?",
        ]

    response = random.choice(greetings)
    memory.add_assistant_message(response)

    return {"messages": [AIMessage(content=response)]}


def handle_closing(state: AgentState) -> dict:
    """Handle closing messages."""
    context = state.get("context", {})
    session_id = context.get("session_id", "default")
    memory = memory_manager.get_or_create(session_id)

    # Clear any pending clarifications
    memory.clear_pending_clarification()

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
                "There are currently no documents in the knowledge base."
            )
        else:
            files = result.get("files", [])
            total_files = result.get("total_files", 0)

            message = f"📂 **Available Documents** ({total_files} files)\n\n"

            for file in files[:10]:
                filename = file.get("filename", "Unknown")
                message += f"  • {filename}\n"

            message += "\n💡 You can ask questions about any of these documents!"

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
    """Ask for clarification when needed."""
    follow_up_questions = state.get("follow_up_questions", [])
    attempts = state.get("clarification_attempts", 0)

    if attempts >= Config.MAX_CLARIFICATION_ATTEMPTS:
        return {
            "messages": [AIMessage(content="Let me search with what I have...")],
            "clarification_needed": False,
            "pending_clarification": False,
        }

    message = (
        follow_up_questions[0]
        if follow_up_questions
        else "Could you provide more details about your specific situation?"
    )

    return {
        "messages": [AIMessage(content=message)],
        "pending_clarification": True,
        "clarification_attempts": attempts + 1,
    }


def agent(state: AgentState) -> dict:
    """Main agent node - handles queries and scenario detection."""
    messages = state["messages"]
    context = state.get("context", {})
    session_id = context.get("session_id", "default")
    thinking = state.get("thinking", {})

    memory = memory_manager.get_or_create(session_id)
    conversation_context = memory.get_context_window(n_turns=4)

    if thinking:
        thinking_str = f"""
Understanding: {thinking.get('understanding', 'N/A')}
Is Follow-up: {thinking.get('is_follow_up', False)}
Is Scenario Response: {thinking.get('is_scenario_response', False)}
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


def handle_scenario_selection(state: AgentState) -> dict:
    """Handle when user selects a specific scenario."""
    context = state.get("context", {})
    session_id = context.get("session_id", "default")
    memory = memory_manager.get_or_create(session_id)

    selected_id = state.get("selected_scenario_id")
    original_query = state.get("original_query", "")
    scenarios = state.get("detected_scenarios", [])
    raw_documents = state.get("raw_search_documents", [])

    # Clear pending clarification
    memory.clear_pending_clarification()

    # Find the selected scenario
    selected_scenario = None
    for s in scenarios:
        if s.get("id") == selected_id:
            selected_scenario = s
            break

    if not selected_scenario and scenarios:
        # Try to get by index
        try:
            idx = int(selected_id) - 1
            if 0 <= idx < len(scenarios):
                selected_scenario = scenarios[idx]
        except:
            pass

    # Build prompt for scenario-specific response
    scenarios_text = "\n".join(
        [
            f"{s.get('id', i+1)}. {s.get('condition', s.get('description', 'N/A'))}"
            for i, s in enumerate(scenarios)
        ]
    )

    docs_text = "\n\n".join([doc.get("content", "") for doc in raw_documents[:3]])

    prompt = SCENARIO_RESPONSE_PROMPT.format(
        original_query=original_query,
        scenario_id=selected_id,
        scenarios=scenarios_text,
        documents=docs_text,
    )

    try:
        response = llm.invoke(
            [
                SystemMessage(content=prompt),
                HumanMessage(content=f"Provide the answer for scenario {selected_id}"),
            ]
        )

        answer = (
            response.content
            if response.content
            else "I couldn't generate a specific answer for that scenario."
        )

        # Update memory with user's context
        if selected_scenario:
            memory.update_user_context(
                "user_scenario", selected_scenario.get("condition", "")
            )

        memory.add_assistant_message(answer, turn_type="scenario_response")

        return {
            "messages": [AIMessage(content=answer)],
            "awaiting_scenario_selection": False,
            "has_multiple_scenarios": False,
        }

    except Exception as e:
        print(f"Error generating scenario response: {e}")
        return {
            "messages": [
                AIMessage(
                    content="I understood your selection but had trouble generating the specific answer. "
                    "Could you please rephrase your original question?"
                )
            ]
        }


def validate_search_results(state: AgentState) -> dict:
    """Validate search results and check for multiple scenarios."""
    messages = state["messages"]
    context = state.get("context", {})
    session_id = context.get("session_id", "default")
    memory = memory_manager.get_or_create(session_id)

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

    # NEW: Check for multiple scenarios
    has_multiple_scenarios = last_tool_result.get("has_multiple_scenarios", False)
    scenarios = last_tool_result.get("scenarios", [])
    clarification_question = last_tool_result.get("clarification_question")
    documents = last_tool_result.get("documents", [])

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
            "has_multiple_scenarios": False,
        }

    # If multiple scenarios detected, set up for clarification
    if has_multiple_scenarios and scenarios:
        original_query = state.get("original_query", "")

        # Store in memory for follow-up
        memory.set_pending_clarification(
            original_query=original_query,
            clarification_type="scenario",
            scenarios=scenarios,
            raw_documents=documents,
        )

        return {
            "should_respond_not_found": False,
            "has_multiple_scenarios": True,
            "detected_scenarios": scenarios,
            "scenario_question": clarification_question,
            "awaiting_scenario_selection": True,
            "raw_search_documents": documents,
            "search_quality": quality,
            "search_confidence": confidence,
            "found_relevant_info": True,
            "support_mode": SupportMode.SCENARIO_SELECTION.value,
        }

    return {
        "should_respond_not_found": False,
        "search_quality": quality,
        "search_confidence": confidence,
        "found_relevant_info": True,
        "has_multiple_scenarios": False,
        "support_mode": SupportMode.DIRECT_ANSWER.value,
    }


def ask_scenario_clarification(state: AgentState) -> dict:
    """Ask user to select which scenario applies to them."""
    context = state.get("context", {})
    session_id = context.get("session_id", "default")
    memory = memory_manager.get_or_create(session_id)

    clarification_question = state.get("scenario_question")
    scenarios = state.get("detected_scenarios", [])

    if clarification_question:
        message = clarification_question
    else:
        # Build a default clarification question
        message = "I found information that applies to different situations. Which of these applies to you?\n\n"
        for i, scenario in enumerate(scenarios[:5], 1):
            condition = scenario.get("condition", scenario.get("description", ""))
            message += f"{i}. {condition}\n"
        message += "\nPlease reply with the number or describe your situation."

    memory.add_assistant_message(message, turn_type="clarification")

    return {
        "messages": [AIMessage(content=message)],
        "awaiting_scenario_selection": True,
    }


def handle_not_found(state: AgentState) -> dict:
    """Handle case when no relevant information was found."""
    context = state.get("context", {})
    session_id = context.get("session_id", "default")
    memory = memory_manager.get_or_create(session_id)

    # Clear any pending clarification
    memory.clear_pending_clarification()

    not_found_message = state.get("not_found_message")

    if not not_found_message:
        not_found_message = (
            "I don't have information about that in my knowledge base. "
            "Could you try rephrasing your question or ask about a different topic?"
        )

    memory.add_assistant_message(not_found_message)
    return {"messages": [AIMessage(content=not_found_message)]}


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
    "handle_scenario_selection",
]:
    """Route based on analysis results."""
    mode = state.get("interaction_mode", InteractionMode.QUERY.value)

    if mode == InteractionMode.GREETING.value:
        return "handle_greeting"

    if mode == InteractionMode.CLOSING.value:
        return "handle_closing"

    if mode == "document_listing":
        return "handle_document_listing"

    # NEW: Handle scenario selection
    if mode == InteractionMode.SCENARIO_SELECTION.value:
        return "handle_scenario_selection"

    if state.get("clarification_needed", False):
        attempts = state.get("clarification_attempts", 0)
        if attempts < Config.MAX_CLARIFICATION_ATTEMPTS:
            return "ask_clarification"

    return "think_and_plan"


def route_after_validation(
    state: AgentState,
) -> Literal["handle_not_found", "ask_scenario_clarification", "agent"]:
    """Route based on search result validation."""
    if state.get("should_respond_not_found", False):
        return "handle_not_found"

    # NEW: Route to scenario clarification if multiple scenarios detected
    if state.get("has_multiple_scenarios", False) and state.get(
        "awaiting_scenario_selection", False
    ):
        return "ask_scenario_clarification"

    return "agent"
