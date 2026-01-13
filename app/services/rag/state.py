# app/services/rag/state.py
import operator
from typing import TypedDict, Annotated, Sequence, List, Optional
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """State definition - matching standalone exactly."""

    messages: Annotated[Sequence[BaseMessage], operator.add]
    context: dict

    # Clarification handling
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

    # Scenario/disambiguation state
    has_multiple_scenarios: bool
    detected_scenarios: List[dict]
    scenario_status: str
    disambiguation_question: str
    selected_scenario: Optional[str]
    disambiguation_depth: int
    scenario_context: List[dict]
    awaiting_scenario_selection: bool
    filtered_search_results: str
    current_scenario_options: List[str]
