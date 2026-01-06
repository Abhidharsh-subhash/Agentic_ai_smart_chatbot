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

        scores = [float(score) for _, score in results]
        best_score = min(scores)
        avg_score = sum(scores) / len(scores)

        relevant_results = [s for s in scores if s < Config.ACCEPTABLE_SCORE]
        relevant_count = len(relevant_results)

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

        query_keywords = set(query.lower().split())
        keyword_matches = 0

        for doc, _ in results:
            content_lower = doc.page_content.lower()
            matches = sum(
                1 for kw in query_keywords if kw in content_lower and len(kw) > 3
            )
            keyword_matches = max(keyword_matches, matches)

        keyword_relevance = keyword_matches / max(len(query_keywords), 1)

        should_respond = (
            quality != SearchQuality.NOT_FOUND.value
            and relevant_count >= Config.MIN_RELEVANT_RESULTS
            and (
                confidence > Config.LOW_CONFIDENCE_THRESHOLD or keyword_relevance > 0.3
            )
        )

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


class ScenarioDetector:
    """Detect and extract multiple scenarios/conditions from search results."""

    # Patterns that indicate conditional/scenario-based content
    SCENARIO_PATTERNS = [
        # Conditional patterns
        (r"(?i)if\s+you\s+(have|are|want|need|don't|do not)\s+([^,.:]+)", "condition"),
        (r"(?i)in\s+case\s+(of|you)\s+([^,.:]+)", "condition"),
        (r"(?i)when\s+(you|the|your)\s+([^,.:]+)", "condition"),
        (
            r"(?i)for\s+(new|existing|individual|business|corporate|first[- ]time)\s+([^,.:]+)",
            "user_type",
        ),
        # Explicit scenario markers
        (
            r"(?i)(option|scenario|method|type|case|situation)\s*(\d+|[a-c])\s*[:\-]?\s*([^.]+)",
            "explicit",
        ),
        # Either/or patterns
        (r"(?i)(either|or)\s+([^,.:]+)", "alternative"),
        # Dependency patterns
        (r"(?i)depending\s+on\s+(your|the|whether)\s+([^,.:]+)", "dependency"),
        (r"(?i)based\s+on\s+(your|the)\s+([^,.:]+)", "dependency"),
        # Multiple solution patterns
        (r"(?i)(solution|step|approach)\s*(\d+)\s*[:\-]", "numbered"),
        # Conditional then patterns
        (r"(?i)if\s+([^,]+),\s*(then\s+)?([^.]+)", "if_then"),
    ]

    # Keywords that often indicate multiple scenarios
    MULTI_SCENARIO_KEYWORDS = [
        "depends",
        "depending",
        "based on",
        "varies",
        "different",
        "options",
        "alternatives",
        "either",
        "or",
        "choose",
        "if you have",
        "if you are",
        "in case of",
        "when you",
        "for new",
        "for existing",
        "type a",
        "type b",
        "option 1",
        "option 2",
    ]

    @classmethod
    def detect_scenarios(cls, documents: List[dict], query: str) -> Dict:
        """
        Analyze documents to detect if there are multiple scenarios/conditions.
        """
        if not documents:
            return {
                "has_multiple_scenarios": False,
                "scenarios": [],
                "clarification_question": None,
                "confidence": 0.0,
            }

        all_content = " ".join([doc.get("content", "") for doc in documents])
        content_lower = all_content.lower()

        # --- NEW LOGIC START: Check for Direct Answer Override ---

        # 1. Length Check: If the total content is short, it's likely a single definition
        # that happens to have "if/else" inside it (like your fee example).
        is_short_content = len(all_content) < Config.MAX_CHARS_FOR_DIRECT_ANSWER

        # 2. Intent Check: "What is" questions usually expect a definition,
        # whereas "How do I" or "My X failed" usually expect a specific path.
        query_lower = query.lower().strip()
        is_factual_question = (
            query_lower.startswith("what is")
            or query_lower.startswith("what are")
            or query_lower.startswith("how much")
            or query_lower.startswith("cost of")
            or "fee" in query_lower
            or "price" in query_lower
        )

        # If it is a short factual answer, bypass scenario detection
        if is_short_content and is_factual_question:
            return {
                "has_multiple_scenarios": False,
                "scenarios": [],
                "clarification_question": None,
                "confidence": 0.0,
            }
        # --- NEW LOGIC END ---

        # Check for multi-scenario keywords
        keyword_matches = sum(
            1 for kw in cls.MULTI_SCENARIO_KEYWORDS if kw in content_lower
        )

        # Extract scenarios using patterns
        scenarios = []
        scenario_id = 1

        for pattern, pattern_type in cls.SCENARIO_PATTERNS:
            matches = re.finditer(pattern, all_content)
            for match in matches:
                scenario = cls._extract_scenario_from_match(
                    match, pattern_type, scenario_id
                )
                if scenario and not cls._is_duplicate_scenario(scenario, scenarios):
                    scenarios.append(scenario)
                    scenario_id += 1

        # Also try to detect scenarios using LLM-friendly structure detection
        structured_scenarios = cls._detect_structured_scenarios(documents)
        for s in structured_scenarios:
            if not cls._is_duplicate_scenario(s, scenarios):
                s["id"] = str(scenario_id)
                scenarios.append(s)
                scenario_id += 1

        # Determine if we have multiple meaningful scenarios
        has_multiple = len(scenarios) >= Config.MIN_SCENARIOS_FOR_CLARIFICATION

        # --- NEW LOGIC: Secondary Safety Check ---
        # Even if we found scenarios via Regex, if the content is extremely short,
        # it's annoying to ask for clarification. Just show the text.
        if has_multiple and len(all_content) < 300:
            has_multiple = False
        # ----------------------------------------

        # Generate clarification question
        clarification_question = None
        if has_multiple:
            clarification_question = cls._generate_clarification_question(
                scenarios, query
            )

        return {
            "has_multiple_scenarios": has_multiple,
            "scenarios": scenarios[:5],  # Limit to 5 scenarios
            "clarification_question": clarification_question,
            "confidence": min(
                0.9, 0.3 + (keyword_matches * 0.1) + (len(scenarios) * 0.15)
            ),
        }

    @classmethod
    def _extract_scenario_from_match(
        cls, match, pattern_type: str, scenario_id: int
    ) -> Optional[dict]:
        """Extract scenario information from regex match."""
        try:
            groups = match.groups()

            if pattern_type == "condition" and len(groups) >= 2:
                return {
                    "id": str(scenario_id),
                    "condition": f"{groups[0]} {groups[1]}".strip(),
                    "description": match.group(0).strip(),
                    "type": pattern_type,
                }
            elif pattern_type == "explicit" and len(groups) >= 3:
                return {
                    "id": str(scenario_id),
                    "condition": f"{groups[0]} {groups[1]}".strip(),
                    "description": (
                        groups[2].strip() if groups[2] else match.group(0).strip()
                    ),
                    "type": pattern_type,
                }
            elif pattern_type == "if_then" and len(groups) >= 3:
                return {
                    "id": str(scenario_id),
                    "condition": groups[0].strip(),
                    "description": (groups[2] or "").strip(),
                    "type": pattern_type,
                }
            elif pattern_type in ["dependency", "user_type", "alternative"]:
                return {
                    "id": str(scenario_id),
                    "condition": " ".join(g for g in groups if g).strip(),
                    "description": match.group(0).strip(),
                    "type": pattern_type,
                }
        except Exception:
            pass
        return None

    @classmethod
    def _detect_structured_scenarios(cls, documents: List[dict]) -> List[dict]:
        """Detect scenarios from document structure (bullets, numbers, etc.)."""
        scenarios = []

        for doc in documents:
            content = doc.get("content", "")

            # Look for bullet points or numbered lists that might be scenarios
            lines = content.split("\n")
            current_scenarios = []

            for line in lines:
                line = line.strip()
                # Check for list items that look like scenarios
                if re.match(
                    r"^[\-\*\•]\s*(?:If|When|For|In case)", line, re.IGNORECASE
                ):
                    current_scenarios.append(
                        {
                            "condition": line.lstrip("-*• "),
                            "description": line,
                            "type": "structured",
                        }
                    )
                elif re.match(
                    r"^\d+[\.\)]\s*(?:If|When|For|In case)", line, re.IGNORECASE
                ):
                    current_scenarios.append(
                        {
                            "condition": re.sub(r"^\d+[\.\)]\s*", "", line),
                            "description": line,
                            "type": "structured",
                        }
                    )

            scenarios.extend(current_scenarios)

        return scenarios

    @classmethod
    def _is_duplicate_scenario(cls, new_scenario: dict, existing: List[dict]) -> bool:
        """Check if scenario is a duplicate of existing ones."""
        new_cond = new_scenario.get("condition", "").lower()
        for s in existing:
            existing_cond = s.get("condition", "").lower()
            # Check for significant overlap
            if new_cond in existing_cond or existing_cond in new_cond:
                return True
            # Check word overlap
            new_words = set(new_cond.split())
            existing_words = set(existing_cond.split())
            if (
                len(new_words) > 2
                and len(new_words & existing_words) / len(new_words) > 0.7
            ):
                return True
        return False

    @classmethod
    def _generate_clarification_question(
        cls, scenarios: List[dict], original_query: str
    ) -> str:
        """Generate a clarification question based on detected scenarios."""
        if not scenarios:
            return "Could you provide more details about your specific situation?"

        # Build options from scenarios
        options = []
        for i, scenario in enumerate(scenarios[:4], 1):
            condition = scenario.get("condition", scenario.get("description", ""))
            # Clean and truncate
            condition = re.sub(r"\s+", " ", condition).strip()
            if len(condition) > 100:
                condition = condition[:97] + "..."
            options.append(f"{i}. {condition}")

        question = (
            f"I found information that applies to different situations. "
            f"To give you the most accurate answer, could you tell me which scenario applies to you?\n\n"
            + "\n".join(options)
            + "\n\nPlease reply with the number or describe your situation."
        )

        return question


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

    # Patterns that indicate user is selecting a scenario
    SCENARIO_SELECTION_PATTERNS = [
        r"^[1-5]$",  # Just a number
        r"^option\s*[1-5]$",
        r"^scenario\s*[1-5]$",
        r"^the\s*(first|second|third|fourth|fifth)\s*one",
        r"^(first|second|third|fourth|fifth)\s*option",
        r"^number\s*[1-5]$",
        r"^#\s*[1-5]$",
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
            "is_scenario_selection": False,
            "selected_scenario": None,
        }

        # Check if this is a scenario selection response
        scenario_selection = cls._check_scenario_selection(query_lower)
        if scenario_selection:
            analysis["is_scenario_selection"] = True
            analysis["selected_scenario"] = scenario_selection
            return analysis

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
    def _check_scenario_selection(cls, query: str) -> Optional[str]:
        """Check if the query is selecting a scenario."""
        query_clean = query.strip().lower()

        # Direct number
        if query_clean.isdigit() and 1 <= int(query_clean) <= 5:
            return query_clean

        # Pattern matching
        for pattern in cls.SCENARIO_SELECTION_PATTERNS:
            match = re.match(pattern, query_clean, re.IGNORECASE)
            if match:
                # Extract the number
                num_match = re.search(r"[1-5]", query_clean)
                if num_match:
                    return num_match.group()
                # Handle word numbers
                word_to_num = {
                    "first": "1",
                    "second": "2",
                    "third": "3",
                    "fourth": "4",
                    "fifth": "5",
                }
                for word, num in word_to_num.items():
                    if word in query_clean:
                        return num

        return None

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
        ],
        "partial": [
            "I found some related information, but nothing that directly answers your question. Would you like me to share what I found?",
        ],
        "suggest_rephrase": [
            "I couldn't find a match for your query. Could you try rephrasing it or being more specific?",
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

    FILE_PATTERNS = [
        r"\b[\w\-]+\.(pdf|docx?|txt|xlsx?|pptx?|csv|json|xml)\b",
        r"\(source:\s*[^)]+\)",
        r"\[source:\s*[^\]]+\]",
        r"(?i)according to the [\w\s]+ document[,:]?\s*",
    ]

    @classmethod
    def contains_general_knowledge(cls, response: str) -> bool:
        patterns = [
            r"(?i)\bbased on my (general )?knowledge\b",
            r"(?i)\bgenerally speaking\b",
            r"(?i)\bin most cases\b",
        ]
        for pattern in patterns:
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
        r"(?i)you should (also )?consider",
        r"(?i)to avoid (any )?(potential )?issues",
        r"(?i)best practice",
        r"(?i)from (my|general) knowledge",
    ]

    @classmethod
    def contains_hallucination(cls, response: str) -> bool:
        for pattern in cls.HALLUCINATION_INDICATORS:
            if re.search(pattern, response):
                return True
        return False
