import operator
from typing import TypedDict, Annotated, Sequence, List, Optional, Dict
from langchain_core.messages import BaseMessage


class ThinkingOutput(TypedDict):
    """Structure for Chain of Thought output."""

    understanding: str
    search_queries: List[str]
    reasoning: str
    is_follow_up: bool
    is_scenario_selection: bool


class AgentState(TypedDict):
    """State definition for the RAG agent."""

    # Messages
    messages: Annotated[Sequence[BaseMessage], operator.add]
    context: dict

    # Query handling
    original_query: str
    clarification_needed: bool
    clarification_reason: str

    # Intent & Mode
    interaction_mode: str
    support_mode: str
    detected_topics: List[str]

    # Search state
    has_searched: bool
    found_relevant_info: bool
    should_respond_not_found: bool
    not_found_message: str
    raw_search_documents: List[Dict]

    # Direct answer (when no scenarios)
    direct_answer: Optional[str]

    # Scenario handling
    has_multiple_scenarios: bool
    detected_scenarios: List[Dict]
    awaiting_scenario_selection: bool
    selected_scenario_id: Optional[str]
    scenario_question: str

    # Thinking/Planning
    thinking: Optional[ThinkingOutput]
    planned_search_queries: List[str]

    # Follow-up context
    is_follow_up_question: bool
    follow_up_context: str
