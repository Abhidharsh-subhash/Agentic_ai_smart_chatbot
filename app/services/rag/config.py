# app/services/rag/config.py
from enum import Enum


class Config:
    """Configuration constants - matching standalone logic exactly."""

    # Confidence thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.7
    MEDIUM_CONFIDENCE_THRESHOLD = 0.4
    LOW_CONFIDENCE_THRESHOLD = 0.2

    # Score thresholds (lower is better for FAISS L2 distance)
    EXCELLENT_SCORE = 0.5
    GOOD_SCORE = 0.7
    ACCEPTABLE_SCORE = 1.0
    NOT_FOUND_SCORE_THRESHOLD = 1.2

    # Minimum relevant results
    MIN_RELEVANT_RESULTS = 1

    # Scenario settings
    MIN_SCENARIOS_FOR_DISAMBIGUATION = 2
    MAX_SCENARIOS_TO_SHOW = 5

    # Redis
    REDIS_SESSION_PREFIX = "chat:session"
    REDIS_TTL_SECONDS = 86400 * 7


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
    DISAMBIGUATION = "disambiguation"


class ScenarioStatus(str, Enum):
    SINGLE = "single"
    MULTIPLE = "multiple"
    NONE = "none"
    RESOLVED = "resolved"
