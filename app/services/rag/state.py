import operator
from typing import TypedDict, Annotated, Sequence, List, Optional
from langchain_core.messages import BaseMessage


class ScenarioInfo(TypedDict):
    """Structure for a detected scenario."""

    id: str
    condition: str
    description: str
    solution: str


class ThinkingOutput(TypedDict):
    """Structure for Chain of Thought output."""

    understanding: str
    key_topics: List[str]
    search_queries: List[str]
    reasoning: str
    is_follow_up: bool
    referenced_context: Optional[str]


class AgentState(TypedDict):
    """State definition for the RAG agent."""

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

    # Chain of Thought
    thinking: Optional[ThinkingOutput]
    planned_search_queries: List[str]

    # Conversation Memory
    conversation_summary: str
    last_assistant_response: str
    last_user_query: str
    is_follow_up_question: bool
    follow_up_context: str

    # ============ NEW: Scenario/Clarification Handling ============
    has_multiple_scenarios: bool
    detected_scenarios: List[ScenarioInfo]
    awaiting_scenario_selection: bool
    selected_scenario_id: Optional[str]
    scenario_question: str
    raw_search_documents: List[dict]  # Store raw docs for scenario-specific response

    # Support Agent Mode
    support_mode: str  # "direct_answer", "need_clarification", "scenario_selection"
    clarification_context: dict  # Store context about what we're clarifying
