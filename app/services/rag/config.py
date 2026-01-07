from enum import Enum


class Config:
    """Configuration constants for RAG pipeline."""

    # Confidence thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.7
    MEDIUM_CONFIDENCE_THRESHOLD = 0.4
    LOW_CONFIDENCE_THRESHOLD = 0.2

    # Score thresholds (lower is better for FAISS L2 distance)
    EXCELLENT_SCORE = 0.6
    GOOD_SCORE = 0.85
    ACCEPTABLE_SCORE = 1.2
    NOT_FOUND_SCORE_THRESHOLD = 1.5

    # Minimum number of relevant results needed
    MIN_RELEVANT_RESULTS = 1

    # Search settings
    DEFAULT_NUM_RESULTS = 10

    # Scenario detection settings
    MIN_SCENARIOS_FOR_CLARIFICATION = 2
    MAX_CLARIFICATION_ATTEMPTS = 3
    MAX_CHARS_FOR_DIRECT_ANSWER = 450

    # NEW: LLM-based scenario detection
    USE_LLM_SCENARIO_DETECTION = True
    SCENARIO_DETECTION_THRESHOLD = 0.6


class SearchQuality(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    MODERATE = "moderate"
    LOW = "low"
    NOT_FOUND = "not_found"


class InteractionMode(str, Enum):
    GREETING = "greeting"
    QUERY = "query"
    CLARIFICATION = "clarification"
    NOT_FOUND = "not_found"
    CLOSING = "closing"
    SCENARIO_SELECTION = "scenario_selection"
    AWAITING_SCENARIO = "awaiting_scenario"


class SupportMode(str, Enum):
    """Support agent response modes."""

    DIRECT_ANSWER = "direct_answer"
    NEED_CLARIFICATION = "need_clarification"
    SCENARIO_SELECTION = "scenario_selection"
    FOLLOW_UP = "follow_up"
    SCENARIO_RESPONSE = "scenario_response"
