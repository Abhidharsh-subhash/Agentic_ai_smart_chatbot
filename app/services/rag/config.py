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

    # If best score is above this, consider it "not found"
    NOT_FOUND_SCORE_THRESHOLD = 1.

    # Minimum number of relevant results needed
    MIN_RELEVANT_RESULTS = 1

    # Search settings
    DEFAULT_NUM_RESULTS = 10


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
