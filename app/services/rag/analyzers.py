import re
import random
from typing import List, Optional
from .config import Config, SearchQuality


class SearchResultAnalyzer:
    """Analyzes search results to determine quality and relevance."""

    @classmethod
    def analyze(cls, results: list, query: str) -> dict:
        """
        Analyze search results and determine quality.

        Returns dict with:
        - found_relevant_info: bool
        - confidence: float (0-1)
        - quality: SearchQuality
        - best_score: float
        - should_respond: bool
        - reason: str
        """
        if not results:
            return {
                "found_relevant_info": False,
                "confidence": 0.0,
                "quality": SearchQuality.NOT_FOUND.value,
                "best_score": float("inf"),
                "should_respond": False,
                "reason": "No search results returned",
                "relevant_count": 0,
            }

        # Extract scores (lower is better for FAISS L2)
        scores = [float(score) for _, score in results]
        best_score = min(scores)
        avg_score = sum(scores) / len(scores)

        # Count relevant results
        relevant_results = [s for s in scores if s < Config.ACCEPTABLE_SCORE]
        relevant_count = len(relevant_results)

        # Determine quality and confidence
        if best_score < Config.EXCELLENT_SCORE:
            quality = SearchQuality.EXCELLENT.value
            confidence = 0.95
        elif best_score < Config.GOOD_SCORE:
            quality = SearchQuality.GOOD.value
            confidence = 0.8
        elif best_score < Config.ACCEPTABLE_SCORE:
            quality = SearchQuality.MODERATE.value
            confidence = 0.5
        elif best_score < Config.NOT_FOUND_SCORE_THRESHOLD:
            quality = SearchQuality.LOW.value
            confidence = 0.25
        else:
            quality = SearchQuality.NOT_FOUND.value
            confidence = 0.0

        # Check keyword overlap
        query_keywords = set(query.lower().split())
        keyword_matches = 0

        for doc, _ in results:
            content_lower = doc.page_content.lower()
            matches = sum(
                1 for kw in query_keywords if kw in content_lower and len(kw) > 3
            )
            keyword_matches = max(keyword_matches, matches)

        keyword_relevance = keyword_matches / max(len(query_keywords), 1)

        # Final determination
        should_respond = (
            quality != SearchQuality.NOT_FOUND.value
            and relevant_count >= Config.MIN_RELEVANT_RESULTS
            and (
                confidence > Config.LOW_CONFIDENCE_THRESHOLD or keyword_relevance > 0.3
            )
        )

        # Reason for decision
        if not should_respond:
            if quality == SearchQuality.NOT_FOUND.value:
                reason = "No relevant information found in knowledge base"
            elif relevant_count < Config.MIN_RELEVANT_RESULTS:
                reason = "Insufficient relevant results"
            else:
                reason = "Low confidence in search results"
        else:
            reason = "Relevant information found"

        return {
            "found_relevant_info": should_respond,
            "confidence": confidence,
            "quality": quality,
            "best_score": best_score,
            "avg_score": avg_score,
            "should_respond": should_respond,
            "reason": reason,
            "relevant_count": relevant_count,
            "keyword_relevance": keyword_relevance,
        }


class QueryAnalyzer:
    """Analyzes user queries for clarity and intent."""

    TRULY_VAGUE_PATTERNS = [
        r"^(it|this|that|these|those)[\?\.]?$",
        r"^(what|how|why|where|when)[\?\.]?$",
        r"^(help|more|info|details)[\?\.]?$",
        r"^tell me[\?\.]?$",
        r"^explain[\?\.]?$",
        r"^show me[\?\.]?$",
    ]

    GREETINGS = [
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "howdy",
        "greetings",
        "hi there",
        "hello there",
    ]

    CLOSINGS = [
        "bye",
        "goodbye",
        "see you",
        "thanks",
        "thank you",
        "that's all",
        "done",
        "exit",
        "quit",
        "thx",
    ]

    @classmethod
    def analyze(cls, query: str, context: dict = None) -> dict:
        """Analyze query for clarity."""
        query_lower = query.lower().strip()

        analysis = {
            "is_clear": True,
            "issues": [],
            "clarification_type": None,
            "follow_up_questions": [],
            "confidence": 0.8,
        }

        if len(query_lower) < 2:
            analysis["is_clear"] = False
            analysis["issues"].append("empty_query")
            analysis["clarification_type"] = "incomplete"
            analysis["follow_up_questions"].append(
                "I'm here to help! What would you like to know about?"
            )
            analysis["confidence"] = 0.0
            return analysis

        for pattern in cls.TRULY_VAGUE_PATTERNS:
            if re.match(pattern, query_lower):
                has_context = context and context.get("last_topic")
                if not has_context:
                    analysis["is_clear"] = False
                    analysis["issues"].append("vague_reference")
                    analysis["clarification_type"] = "vague"
                    analysis["follow_up_questions"].append(
                        "Could you please be more specific about what you'd like to know?"
                    )
                    analysis["confidence"] = 0.2
                    return analysis

        return analysis

    @classmethod
    def is_greeting(cls, query: str) -> bool:
        return query.lower().strip().rstrip("!.,") in cls.GREETINGS

    @classmethod
    def is_closing(cls, query: str) -> bool:
        query_clean = query.lower().strip().rstrip("!.,")
        return any(c in query_clean for c in cls.CLOSINGS)


class NotFoundResponseGenerator:
    """Generates appropriate not-found responses."""

    RESPONSES = {
        "general": [
            "I don't have information about that in my knowledge base. Could you try asking about a different topic?",
            "I couldn't find any relevant information about this topic. Is there something else I can help you with?",
            "That topic doesn't appear to be covered in the documentation I have access to.",
        ],
        "partial": [
            "I found some related information, but nothing that directly answers your question. Would you like me to share what I found?",
        ],
        "suggest_rephrase": [
            "I couldn't find a match for your query. Could you try rephrasing it or being more specific?",
            "No results found. Perhaps try using different keywords?",
        ],
    }

    @classmethod
    def generate(
        cls,
        query: str,
        search_analysis: dict,
        available_topics: Optional[List[str]] = None,
    ) -> str:
        """Generate an appropriate not-found response."""
        quality = search_analysis.get("quality", SearchQuality.NOT_FOUND.value)
        confidence = search_analysis.get("confidence", 0)

        if quality == SearchQuality.LOW.value and confidence > 0.1:
            response = random.choice(cls.RESPONSES["partial"])
        elif confidence == 0:
            response = random.choice(cls.RESPONSES["general"])
        else:
            response = random.choice(cls.RESPONSES["suggest_rephrase"])

        if available_topics and len(available_topics) > 0:
            topic_list = ", ".join(available_topics[:5])
            response += f"\n\nI can help you with topics like: {topic_list}."

        return response


class ResponseSanitizer:
    """Sanitize responses to remove file references."""

    GENERAL_KNOWLEDGE_PATTERNS = [
        r"(?i)\bbased on my (general )?knowledge\b",
        r"(?i)\bgenerally speaking\b",
        r"(?i)\bin most cases\b",
        r"(?i)\bfrom what I know\b",
    ]

    FILE_PATTERNS = [
        r"\b[\w\-]+\.(pdf|docx?|txt|xlsx?|pptx?|csv|json|xml)\b",
        r"\(source:\s*[^)]+\)",
        r"\[source:\s*[^\]]+\]",
        r"(?i)according to the [\w\s]+ document[,:]?\s*",
    ]

    @classmethod
    def contains_general_knowledge(cls, response: str) -> bool:
        for pattern in cls.GENERAL_KNOWLEDGE_PATTERNS:
            if re.search(pattern, response):
                return True
        return False

    @classmethod
    def sanitize(cls, response: str) -> str:
        if not response:
            return response

        sanitized = response
        for pattern in cls.FILE_PATTERNS:
            sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)

        sanitized = re.sub(r"\s{2,}", " ", sanitized)
        sanitized = re.sub(r"\s+([.,!?])", r"\1", sanitized)
        return sanitized.strip()
