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


# app/services/rag/analyzers.py

# ... existing code ...


class ClarificationContextAnalyzer:
    """
    Analyzes if user input is a clarification response or a new unrelated question.
    Uses LLM to make intelligent context-aware decisions.
    """

    CONTEXT_ANALYSIS_PROMPT = """You are analyzing a conversation to determine if the user is responding to a clarification question or asking a completely new question.

CURRENT CONVERSATION CONTEXT:
- Original Question: {original_question}
- Last Clarification Question Asked: {last_question}
- Topic Being Discussed: {topics}

USER'S NEW INPUT:
"{user_input}"

TASK: Determine if the user's input is:
1. A RESPONSE to the clarification question (answering yes/no, describing their situation, providing details about the original topic)
2. A NEW QUESTION that is unrelated to the current clarification flow

IMPORTANT INDICATORS:

For CLARIFICATION RESPONSE (is_clarification_response=true):
- Short answers like "yes", "no", "nope", "correct", "that's right"
- Descriptions of their specific situation related to the original question
- References to errors, issues, or problems mentioned in the original question
- Elaborations on the same topic
- "It says...", "I'm seeing...", "The error shows..."

For NEW QUESTION (is_new_question=true):
- Completely different topics (e.g., original was about errors, new is about how to apply)
- Questions starting with "how do I...", "what is...", "can you tell me about...", "how can I..."
- Questions that make sense on their own without the previous context
- Topic shifts that don't relate to the current problem
- General inquiries unrelated to troubleshooting the original issue

Respond with ONLY a JSON object (no markdown, no explanation):
{{"is_clarification_response": true/false, "is_new_question": true/false, "confidence": 0.0-1.0, "reasoning": "brief explanation", "detected_intent": "clarification_response" or "new_question"}}"""

    def __init__(self, llm=None):
        self._llm = llm

    @property
    def llm(self):
        if self._llm is None:
            from langchain_openai import ChatOpenAI
            from app.core.config import settings

            self._llm = ChatOpenAI(
                model=getattr(settings, "openai_model", "gpt-4o"),
                temperature=0.0,
                openai_api_key=settings.openai_api_key,
            )
        return self._llm

    def analyze_input_context(
        self,
        user_input: str,
        original_question: str,
        last_clarification_question: str,
        topics: List[str] = None,
    ) -> Dict:
        """
        Analyze if user input is a clarification response or new question.

        Returns:
            Dict with:
            - is_clarification_response: bool
            - is_new_question: bool
            - confidence: float (0-1)
            - reasoning: str
            - detected_intent: str
        """
        from langchain_core.messages import SystemMessage

        # Quick check for obvious cases to save LLM calls
        user_lower = user_input.lower().strip()

        # Obvious clarification responses
        obvious_clarification = [
            "yes",
            "no",
            "nope",
            "yep",
            "yeah",
            "correct",
            "that's right",
            "not that",
            "neither",
            "none",
            "both",
            "all of them",
            "the first one",
            "the second one",
            "option 1",
            "option 2",
            "n",
            "y",
        ]

        if user_lower in obvious_clarification:
            return {
                "is_clarification_response": True,
                "is_new_question": False,
                "confidence": 0.95,
                "reasoning": "Obvious clarification response (yes/no pattern)",
                "detected_intent": "clarification_response",
            }

        # Use LLM for nuanced cases
        prompt = self.CONTEXT_ANALYSIS_PROMPT.format(
            original_question=original_question,
            last_question=last_clarification_question,
            topics=", ".join(topics) if topics else "Not specified",
            user_input=user_input,
        )

        try:
            result = self.llm.invoke([SystemMessage(content=prompt)])
            analysis = self._extract_json(result.content.strip())

            # Set defaults
            analysis.setdefault("is_clarification_response", False)
            analysis.setdefault("is_new_question", True)
            analysis.setdefault("confidence", 0.5)
            analysis.setdefault("detected_intent", "new_question")
            analysis.setdefault("reasoning", "")

            # Ensure mutual exclusivity
            if analysis["is_clarification_response"] and analysis["is_new_question"]:
                # If both are true, go with higher confidence option
                analysis["is_new_question"] = analysis["confidence"] < 0.5
                analysis["is_clarification_response"] = analysis["confidence"] >= 0.5

            return analysis

        except Exception as e:
            print(f"[ClarificationContextAnalyzer] Error: {e}")
            # Fallback heuristics
            return self._fallback_analysis(user_input, original_question)

    def _fallback_analysis(self, user_input: str, original_question: str) -> Dict:
        """Fallback heuristics when LLM fails."""
        user_lower = user_input.lower().strip()

        # Check for question patterns that indicate new questions
        new_question_patterns = [
            "how can i",
            "how do i",
            "how to",
            "what is",
            "what are",
            "can you tell me",
            "can you explain",
            "i want to",
            "i need to",
            "tell me about",
            "explain",
            "help me with",
        ]

        for pattern in new_question_patterns:
            if user_lower.startswith(pattern):
                return {
                    "is_clarification_response": False,
                    "is_new_question": True,
                    "confidence": 0.7,
                    "reasoning": f"Input starts with new question pattern: '{pattern}'",
                    "detected_intent": "new_question",
                }

        # Check if input contains question marks (likely a new question)
        if "?" in user_input and len(user_input.split()) > 5:
            return {
                "is_clarification_response": False,
                "is_new_question": True,
                "confidence": 0.6,
                "reasoning": "Input contains question mark and is longer than typical clarification",
                "detected_intent": "new_question",
            }

        # Short responses are likely clarifications
        if len(user_input.split()) <= 3:
            return {
                "is_clarification_response": True,
                "is_new_question": False,
                "confidence": 0.6,
                "reasoning": "Short response, likely clarification",
                "detected_intent": "clarification_response",
            }

        # Default to clarification for medium-length responses
        return {
            "is_clarification_response": True,
            "is_new_question": False,
            "confidence": 0.5,
            "reasoning": "Fallback - treating as clarification",
            "detected_intent": "clarification_response",
        }

    def _extract_json(self, text: str) -> Dict:
        """Extract JSON from response."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        for pattern in [r"```json\s*([\s\S]*?)\s*```", r"```\s*([\s\S]*?)\s*```"]:
            match = re.search(pattern, text)
            if match:
                try:
                    return json.loads(match.group(1).strip())
                except:
                    continue

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
                        except:
                            break

        raise ValueError(f"No JSON found: {text[:100]}")


# Create singleton instance
clarification_context_analyzer = ClarificationContextAnalyzer()


class ResponseScenarioAnalyzer:
    """
    Analyzes LLM responses and manages multi-turn clarification.
    Progressively narrows down scenarios based on user responses.
    """

    ANALYSIS_PROMPT = """Analyze this response for multiple scenarios that need clarification.

RESPONSE:
{response}

USER QUESTION:
{user_question}

For each scenario, identify:
1. A clear title
2. The specific condition/situation
3. The solution
4. Keywords that would indicate this scenario
5. A specific YES/NO question that would identify this scenario

Respond with ONLY a JSON object:

{{
    "has_multiple_scenarios": true/false,
    "scenario_count": <number>,
    "scenarios": [
        {{
            "id": 1,
            "title": "Short title",
            "condition": "When this happens",
            "solution": "Do this to fix",
            "keywords": ["error", "message", "keywords"],
            "identifying_question": "A yes/no question that identifies this scenario"
        }}
    ],
    "first_question": "The best first question to start narrowing down (based on most common scenario or biggest differentiator)"
}}

Make identifying_question specific like:
- "Is the application showing as already existing in the system?"
- "Are you seeing an error about the passport number?"
- "Is the sponsor's license expired?"

NOT generic like "What error are you seeing?" """

    EVALUATE_RESPONSE_PROMPT = """Based on the user's response, determine which scenarios are still possible.

REMAINING SCENARIOS:
{scenarios}

USER'S RESPONSE:
{user_response}

PREVIOUS CONTEXT:
{previous_context}

QUESTION THAT WAS ASKED:
{asked_question}

Analyze and respond with ONLY a JSON object:

{{
    "interpretation": "What the user's response means",
    "eliminated_scenario_ids": [list of scenario IDs that are now eliminated],
    "likely_scenario_ids": [list of scenario IDs that seem likely based on response],
    "confidence_in_match": 0.0-1.0,
    "matched_scenario_id": null or ID if confident match,
    "needs_more_info": true/false,
    "suggested_next_question": "If needs_more_info, what to ask next (based on remaining scenarios)",
    "reasoning": "Why this interpretation"
}}

Rules:
- If user says "yes" to a scenario's identifying question, that scenario is likely
- If user says "no" or indicates something doesn't apply, eliminate those scenarios
- If user gives partial info, narrow down based on keywords
- If user says "I don't know" or is vague, try a different angle
- confidence_in_match should be >= 0.7 to consider it a match
- suggested_next_question should be specific to differentiate remaining scenarios"""

    GENERATE_NEXT_QUESTION_PROMPT = """Generate the next clarifying question to narrow down these scenarios.

REMAINING SCENARIOS:
{scenarios}

WHAT WE ALREADY KNOW:
{context}

QUESTIONS ALREADY ASKED:
{asked_questions}

Generate ONE specific question that:
1. Helps differentiate between the remaining scenarios
2. Is different from questions already asked
3. Is clear and easy to answer
4. Focuses on observable symptoms/errors

Respond with ONLY a JSON object:
{{
    "question": "The next question to ask",
    "targets_scenarios": [list of scenario IDs this question helps identify],
    "reasoning": "Why this question helps"
}}"""

    FINAL_RESPONSE_PROMPT = """Generate a direct, helpful response for this matched scenario.

USER'S SITUATION: {user_context}
MATCHED SCENARIO: {scenario_title}
CONDITION: {condition}
SOLUTION: {solution}

Give a concise, actionable response (2-3 sentences max). 
Start directly with the solution, no preamble like "Based on your situation..."."""

    def __init__(self, llm=None):
        self._llm = llm

    @property
    def llm(self):
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
        """Initial analysis of response for multiple scenarios."""
        from langchain_core.messages import SystemMessage

        if not response or len(response) < 100:
            return self._single_scenario_result("Response too short")

        prompt = self.ANALYSIS_PROMPT.format(
            response=response, user_question=user_question
        )

        try:
            result = self.llm.invoke([SystemMessage(content=prompt)])
            analysis = self._extract_json(result.content.strip())

            analysis.setdefault("has_multiple_scenarios", False)
            analysis.setdefault("scenario_count", 0)
            analysis.setdefault("scenarios", [])
            analysis.setdefault("first_question", "")

            if analysis.get("scenario_count", 0) < 3:
                analysis["has_multiple_scenarios"] = False

            return analysis

        except Exception as e:
            print(f"[ResponseAnalyzer] Analysis error: {e}")
            return self._single_scenario_result(str(e))

    def evaluate_user_response(
        self,
        remaining_scenarios: List[Dict],
        user_response: str,
        previous_context: str,
        asked_question: str,
    ) -> Dict:
        """Evaluate user's response and narrow down scenarios."""
        from langchain_core.messages import SystemMessage

        if not remaining_scenarios:
            return {
                "matched_scenario_id": None,
                "needs_more_info": False,
                "eliminated_scenario_ids": [],
                "confidence_in_match": 0.0,
            }

        prompt = self.EVALUATE_RESPONSE_PROMPT.format(
            scenarios=json.dumps(remaining_scenarios, indent=2),
            user_response=user_response,
            previous_context=previous_context or "No previous context",
            asked_question=asked_question or "Initial question",
        )

        try:
            result = self.llm.invoke([SystemMessage(content=prompt)])
            evaluation = self._extract_json(result.content.strip())

            evaluation.setdefault("eliminated_scenario_ids", [])
            evaluation.setdefault("likely_scenario_ids", [])
            evaluation.setdefault("confidence_in_match", 0.0)
            evaluation.setdefault("matched_scenario_id", None)
            evaluation.setdefault("needs_more_info", True)
            evaluation.setdefault("suggested_next_question", "")
            evaluation.setdefault("interpretation", "")

            return evaluation

        except Exception as e:
            print(f"[ResponseAnalyzer] Evaluation error: {e}")
            return {
                "matched_scenario_id": None,
                "needs_more_info": True,
                "eliminated_scenario_ids": [],
                "confidence_in_match": 0.0,
                "suggested_next_question": "",
            }

    def generate_next_question(
        self, remaining_scenarios: List[Dict], context: str, asked_questions: List[str]
    ) -> Dict:
        """Generate the next differentiating question."""
        from langchain_core.messages import SystemMessage

        if not remaining_scenarios:
            return {"question": "", "reasoning": "No scenarios left"}

        # If only one scenario left, we should just provide the answer
        if len(remaining_scenarios) == 1:
            return {
                "question": "",
                "single_scenario": remaining_scenarios[0],
                "reasoning": "Only one scenario remaining",
            }

        prompt = self.GENERATE_NEXT_QUESTION_PROMPT.format(
            scenarios=json.dumps(remaining_scenarios, indent=2),
            context=context or "No context yet",
            asked_questions=(
                json.dumps(asked_questions) if asked_questions else "None yet"
            ),
        )

        try:
            result = self.llm.invoke([SystemMessage(content=prompt)])
            question_data = self._extract_json(result.content.strip())

            question_data.setdefault("question", "")
            question_data.setdefault("reasoning", "")

            return question_data

        except Exception as e:
            print(f"[ResponseAnalyzer] Question generation error: {e}")
            # Fallback: use identifying question from first remaining scenario
            if remaining_scenarios:
                fallback_q = remaining_scenarios[0].get(
                    "identifying_question",
                    "Could you provide more details about what you're experiencing?",
                )
                return {"question": fallback_q, "reasoning": "Fallback question"}
            return {"question": "", "reasoning": "Error"}

    def generate_final_response(self, scenario: Dict, user_context: str) -> str:
        """Generate the final response for matched scenario."""
        from langchain_core.messages import SystemMessage

        prompt = self.FINAL_RESPONSE_PROMPT.format(
            user_context=user_context,
            scenario_title=scenario.get("title", ""),
            condition=scenario.get("condition", ""),
            solution=scenario.get("solution", ""),
        )

        try:
            result = self.llm.invoke([SystemMessage(content=prompt)])
            return result.content.strip()
        except Exception:
            return scenario.get("solution", "Please try again or contact support.")

    def _extract_json(self, text: str) -> Dict:
        """Extract JSON from response."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        for pattern in [r"```json\s*([\s\S]*?)\s*```", r"```\s*([\s\S]*?)\s*```"]:
            match = re.search(pattern, text)
            if match:
                try:
                    return json.loads(match.group(1).strip())
                except:
                    continue

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
                        except:
                            break

        raise ValueError(f"No JSON found: {text[:100]}")

    def _single_scenario_result(self, reason: str) -> Dict:
        return {
            "has_multiple_scenarios": False,
            "scenario_count": 1,
            "scenarios": [],
            "first_question": "",
            "confidence": 1.0,
            "reasoning": reason,
        }


response_scenario_analyzer = ResponseScenarioAnalyzer()
