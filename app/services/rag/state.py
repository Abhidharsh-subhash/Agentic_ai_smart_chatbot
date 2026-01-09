# app/services/rag/state.py
import operator
from typing import TypedDict, Annotated, Sequence, List, Optional
from langchain_core.messages import BaseMessage


class ThinkingOutput(TypedDict):
    """Structure for Chain of Thought output."""

    understanding: str
    key_topics: List[str]
    search_queries: List[str]
    reasoning: str
    is_follow_up: bool
    referenced_context: Optional[str]


class ScenarioInfo(TypedDict):
    """Structure for a single scenario."""

    id: int
    title: str
    condition: str
    solution: str
    keywords: List[str]
    differentiating_question: str  # Question that identifies this scenario


class ClarificationState(TypedDict):
    """State for multi-turn clarification flow."""

    is_active: bool
    attempt_count: int
    max_attempts: int
    original_question: str
    all_scenarios: List[ScenarioInfo]
    remaining_scenarios: List[ScenarioInfo]  # Scenarios still possible
    eliminated_scenarios: List[int]  # IDs of eliminated scenarios
    user_responses: List[str]  # All user responses in this flow
    asked_questions: List[str]  # Questions we've already asked
    current_question: str
    gathered_context: str  # Accumulated context from user
    current_scenario_id: Optional[int]
    scenario_question_map: dict  # Maps scenario ID to its identifying question
    is_last_scenario_confirmation: bool  # NEW: Flag when asking about the last scenario


class ResponseAnalysis(TypedDict):
    """Structure for response analysis output."""

    has_multiple_scenarios: bool
    scenario_count: int
    scenarios: List[ScenarioInfo]
    clarification_question: str
    confidence: float
    reasoning: str


class AgentState(TypedDict):
    """State definition for the RAG agent."""

    messages: Annotated[Sequence[BaseMessage], operator.add]
    context: dict

    # Clarification handling (for vague queries)
    clarification_needed: bool
    clarification_reason: str
    follow_up_questions: List[str]
    pending_clarification: bool
    original_query: str
    clarification_attempts: int

    # Intent & Understanding
    user_intent: str
    detected_topics: List[str]
    sentiment: str

    # Interaction tracking
    interaction_mode: str
    conversation_history: List[dict]
    topic_history: List[str]

    # Search state
    search_confidence: float
    search_quality: str
    has_searched: bool
    search_results: str
    found_relevant_info: bool
    best_match_score: float

    # Response control
    should_respond_not_found: bool
    not_found_message: str

    # Chain of Thought
    thinking: Optional[ThinkingOutput]
    planned_search_queries: List[str]

    # Conversation Memory
    conversation_summary: str
    last_assistant_response: str
    last_user_query: str
    is_follow_up_question: bool
    follow_up_context: str

    # ============ Multi-Turn Scenario Clarification ============
    response_analysis: Optional[ResponseAnalysis]
    needs_scenario_clarification: bool
    scenario_clarification_question: str
    scenario_options: List[str]
    original_full_response: str
    parsed_scenarios: List[ScenarioInfo]
    awaiting_scenario_selection: bool
    selected_scenario_id: Optional[int]
    scenario_clarification_pending: bool
    user_scenario_context: str

    # NEW: Multi-turn clarification tracking
    clarification_state: Optional[ClarificationState]
