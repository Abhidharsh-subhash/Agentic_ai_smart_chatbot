# app/services/rag/analyzers.py
import re
import random
import json
from typing import List, Optional, Dict
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


class HallucinationDetector:
    """Detect and filter hallucinated content."""

    HALLUCINATION_INDICATORS = [
        r"(?i)this could lead to",
        r"(?i)it('s| is) important to",
        r"(?i)generally speaking",
        r"(?i)in most cases",
        r"(?i)typically,",
        r"(?i)usually,",
        r"(?i)you should (also )?consider",
        r"(?i)to avoid (any )?(potential )?issues",
        r"(?i)it('s| is) (also )?recommended",
        r"(?i)best practice",
        r"(?i)keep in mind",
        r"(?i)be aware that",
        r"(?i)note that",
        r"(?i)remember that",
        r"(?i)from (my|general) knowledge",
        r"(?i)based on (my|general) (knowledge|understanding)",
        r"(?i)as a general rule",
        r"(?i)in general,",
        r"(?i)commonly,",
        r"(?i)often,",
        r"(?i)may (also )?result in",
        r"(?i)could (also )?cause",
        r"(?i)might (also )?lead to",
        r"(?i)to ensure",
        r"(?i)to prevent",
        r"(?i)for (your )?safety",
        r"(?i)as soon as possible",
        r"(?i)it would be (wise|advisable)",
    ]

    @classmethod
    def contains_hallucination(cls, response: str) -> bool:
        """Check if response contains hallucination indicators."""
        for pattern in cls.HALLUCINATION_INDICATORS:
            if re.search(pattern, response):
                return True
        return False

    @classmethod
    def get_hallucination_phrases(cls, response: str) -> list:
        """Get list of detected hallucination phrases."""
        found = []
        for pattern in cls.HALLUCINATION_INDICATORS:
            matches = re.findall(pattern, response)
            found.extend(matches)
        return found


# ============================================
# NEW: Response Scenario Analyzer
# ============================================


class ResponseScenarioAnalyzer:
    """
    Analyzes LLM responses to detect multiple scenarios that need user clarification.
    Uses LLM to dynamically understand response structure.
    """

    ANALYSIS_PROMPT = """You are a response analyzer. Analyze the following response to determine if it contains multiple distinct scenarios, conditions, or cases that require user clarification.

## RESPONSE TO ANALYZE:
{response}

## USER'S ORIGINAL QUESTION:
{user_question}

## YOUR TASK:
Determine if this response presents multiple distinct scenarios where the user needs to clarify which one applies to their situation.

Signs of multiple scenarios:
- Numbered lists with different conditions (1., 2., 3., etc.)
- "If X... then Y" patterns
- Multiple root causes with different solutions
- Different cases or situations described
- Phrases like "could be because of", "reasons include", "depends on"

## RESPOND WITH ONLY A JSON OBJECT (no markdown, no explanation):

If multiple scenarios detected (3 or more distinct scenarios):
{{
    "has_multiple_scenarios": true,
    "scenario_count": <number>,
    "scenarios": [
        {{
            "id": 1,
            "title": "Brief title for this scenario",
            "description": "What condition/situation this describes",
            "solution": "The solution for this scenario",
            "keywords": ["keyword1", "keyword2"]
        }}
    ],
    "clarification_question": "A natural question to ask user to identify their scenario",
    "clarification_options": ["Option 1 description", "Option 2 description"],
    "confidence": 0.95,
    "reasoning": "Why this needs clarification"
}}

If single scenario or direct answer (fewer than 3 scenarios):
{{
    "has_multiple_scenarios": false,
    "scenario_count": 1,
    "scenarios": [],
    "clarification_question": "",
    "clarification_options": [],
    "confidence": 0.95,
    "reasoning": "Why this is a direct answer"
}}"""

    MATCH_SCENARIO_PROMPT = """Based on the user's clarification, identify which scenario best matches their situation.

## AVAILABLE SCENARIOS:
{scenarios}

## USER'S CLARIFICATION:
{user_clarification}

## ORIGINAL QUESTION:
{original_question}

Respond with ONLY a JSON object:
{{
    "matched_scenario_id": <id number or null if no clear match>,
    "confidence": <0.0 to 1.0>,
    "matched_keywords": ["keywords that matched"],
    "reasoning": "Why this scenario matches",
    "needs_more_info": false
}}

If no scenario clearly matches, set matched_scenario_id to null and needs_more_info to true."""

    FOCUSED_RESPONSE_PROMPT = """Based on the user's situation, provide a focused, helpful response.

## USER'S ORIGINAL QUESTION:
{original_question}

## USER'S SPECIFIC SITUATION:
{user_context}

## RELEVANT SCENARIO:
Title: {scenario_title}
Description: {scenario_description}
Solution: {scenario_solution}

Generate a response that:
1. Acknowledges their specific situation briefly
2. Provides the solution clearly and directly
3. Is concise and actionable
4. Does NOT mention other scenarios

Keep it helpful and under 100 words."""

    def __init__(self, llm=None):
        self._llm = llm

    @property
    def llm(self):
        """Lazy load LLM to avoid import issues."""
        if self._llm is None:
            from langchain_openai import ChatOpenAI
            from app.core.config import settings

            self._llm = ChatOpenAI(
                model=getattr(settings, "openai_model", "gpt-4o"),
                temperature=0.0,
                openai_api_key=settings.openai_api_key,
            )
        return self._llm

    def analyze_response(self, response: str, user_question: str) -> Dict:
        """
        Analyze a response to detect if it contains multiple scenarios.

        Returns:
            Dict with analysis results including scenarios and clarification question
        """
        from langchain_core.messages import SystemMessage

        if not response or len(response) < 100:
            return self._single_scenario_result(
                "Response too short for multiple scenarios"
            )

        prompt = self.ANALYSIS_PROMPT.format(
            response=response, user_question=user_question
        )

        try:
            result = self.llm.invoke([SystemMessage(content=prompt)])
            response_text = result.content.strip()

            # Parse JSON from response
            analysis = self._extract_json(response_text)

            # Validate required fields
            if not isinstance(analysis, dict):
                return self._single_scenario_result("Invalid analysis format")

            # Ensure all required fields exist
            analysis.setdefault("has_multiple_scenarios", False)
            analysis.setdefault("scenario_count", 0)
            analysis.setdefault("scenarios", [])
            analysis.setdefault("clarification_question", "")
            analysis.setdefault("clarification_options", [])
            analysis.setdefault("confidence", 0.0)
            analysis.setdefault("reasoning", "")

            # Only consider it multi-scenario if 3+ scenarios
            if analysis.get("scenario_count", 0) < 3:
                analysis["has_multiple_scenarios"] = False

            return analysis

        except Exception as e:
            print(f"[ResponseAnalyzer] Error analyzing response: {e}")
            return self._single_scenario_result(f"Analysis error: {str(e)}")

    def match_user_to_scenario(
        self, scenarios: List[Dict], user_clarification: str, original_question: str
    ) -> Dict:
        """
        Match user's clarification to the most relevant scenario.

        Returns:
            Dict with matched_scenario_id and confidence
        """
        from langchain_core.messages import SystemMessage

        if not scenarios:
            return {
                "matched_scenario_id": None,
                "confidence": 0.0,
                "needs_more_info": True,
                "reasoning": "No scenarios available",
            }

        # Format scenarios for prompt
        scenarios_text = json.dumps(scenarios, indent=2)

        prompt = self.MATCH_SCENARIO_PROMPT.format(
            scenarios=scenarios_text,
            user_clarification=user_clarification,
            original_question=original_question,
        )

        try:
            result = self.llm.invoke([SystemMessage(content=prompt)])
            response_text = result.content.strip()

            match_result = self._extract_json(response_text)

            # Validate
            match_result.setdefault("matched_scenario_id", None)
            match_result.setdefault("confidence", 0.0)
            match_result.setdefault("needs_more_info", True)
            match_result.setdefault("reasoning", "")
            match_result.setdefault("matched_keywords", [])

            return match_result

        except Exception as e:
            print(f"[ResponseAnalyzer] Error matching scenario: {e}")
            return {
                "matched_scenario_id": None,
                "confidence": 0.0,
                "needs_more_info": True,
                "reasoning": f"Matching error: {str(e)}",
            }

    def generate_focused_response(
        self, scenario: Dict, original_question: str, user_context: str
    ) -> str:
        """
        Generate a focused response for a specific scenario.
        """
        from langchain_core.messages import SystemMessage

        prompt = self.FOCUSED_RESPONSE_PROMPT.format(
            original_question=original_question,
            user_context=user_context,
            scenario_title=scenario.get("title", "N/A"),
            scenario_description=scenario.get("description", "N/A"),
            scenario_solution=scenario.get("solution", "N/A"),
        )

        try:
            result = self.llm.invoke([SystemMessage(content=prompt)])
            return result.content.strip()
        except Exception as e:
            # Fallback to direct scenario info
            return f"{scenario.get('description', '')}\n\nSolution: {scenario.get('solution', 'Please try again.')}"

    def _extract_json(self, text: str) -> Dict:
        """Extract JSON from LLM response."""
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to extract from code blocks
        patterns = [
            r"```json\s*([\s\S]*?)\s*```",
            r"```\s*([\s\S]*?)\s*```",
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return json.loads(match.group(1).strip())
                except (json.JSONDecodeError, IndexError):
                    continue

        # Try to find JSON object
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

    def _single_scenario_result(self, reason: str) -> Dict:
        """Return a standard single-scenario result."""
        return {
            "has_multiple_scenarios": False,
            "scenario_count": 1,
            "scenarios": [],
            "clarification_question": "",
            "clarification_options": [],
            "confidence": 1.0,
            "reasoning": reason,
        }


# Singleton instance (lazy initialization)
response_scenario_analyzer = ResponseScenarioAnalyzer()
