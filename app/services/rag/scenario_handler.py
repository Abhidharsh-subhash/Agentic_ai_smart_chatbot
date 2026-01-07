"""
Scenario Handler - Manages dynamic scenario detection and clarification.
Uses LLM to analyze search results for multiple answer paths.
"""

import json
import re
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from app.core.config import settings


@dataclass
class DetectedScenario:
    """Represents a detected scenario/option in the answer."""

    id: str
    title: str
    condition: str
    description: str
    keywords: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "title": self.title,
            "condition": self.condition,
            "description": self.description,
            "keywords": self.keywords,
        }


@dataclass
class ScenarioAnalysisResult:
    """Result of scenario analysis."""

    has_multiple_scenarios: bool
    scenarios: List[DetectedScenario]
    clarification_question: Optional[str]
    direct_answer: Optional[str]
    confidence: float
    reasoning: str


class ScenarioHandler:
    """
    Handles dynamic scenario detection and clarification using LLM.
    Supports both sync and async operations.
    """

    SCENARIO_ANALYSIS_PROMPT = """Analyze the following search results for a user query and determine if there are multiple scenarios, options, or conditions that require clarification.

## USER QUERY:
{query}

## SEARCH RESULTS:
{search_results}

## ANALYSIS TASK:
1. Determine if the answer varies based on different conditions, user types, or scenarios
2. If YES: Extract each distinct scenario with its condition and relevant answer
3. If NO: The answer is direct and applies to all cases

## RESPOND WITH ONLY THIS JSON (no markdown, no extra text):
{{
    "has_multiple_scenarios": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation of why",
    "scenarios": [
        {{
            "id": "1",
            "title": "short title",
            "condition": "when this applies",
            "description": "the answer for this scenario",
            "keywords": ["keyword1", "keyword2"]
        }}
    ],
    "direct_answer": "if has_multiple_scenarios is false, provide the direct answer here",
    "clarification_question": "if has_multiple_scenarios is true, generate a natural question asking user which scenario applies"
}}

## EXAMPLES OF MULTIPLE SCENARIOS:
- "If you are an admin..." vs "If you are a regular user..."
- "For new customers..." vs "For existing customers..."
- "Option A: ..." vs "Option B: ..."
- Different steps based on device type, account type, etc.

## EXAMPLES OF DIRECT ANSWERS:
- Simple facts: "The fee is $50"
- Single process: "To reset password, click forgot password..."
- Definitions: "A widget is..."

Be conservative - only flag as multiple scenarios if the answer truly differs based on user situation."""

    CLARIFICATION_GENERATION_PROMPT = """Generate a friendly, natural clarification question for the user.

USER'S ORIGINAL QUESTION: {query}

DETECTED SCENARIOS:
{scenarios}

Generate a question that:
1. Acknowledges what the user asked
2. Explains you found information for different situations
3. Lists the options clearly with numbers
4. Asks them to choose or describe their situation

Keep it conversational and helpful. Don't be robotic.

RESPOND WITH ONLY THE CLARIFICATION QUESTION TEXT (no JSON, no quotes):"""

    SCENARIO_EXTRACTION_PROMPT = """Extract the specific answer for the selected scenario.

ORIGINAL QUESTION: {query}

USER SELECTED: {selection}

ALL SCENARIOS:
{scenarios}

RELEVANT DOCUMENTS:
{documents}

Provide ONLY the answer relevant to the selected scenario.
Be specific and detailed.
Use only information from the documents.

RESPONSE:"""

    def __init__(self):
        self._llm = None

    @property
    def llm(self):
        """Lazy initialization of LLM."""
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=getattr(settings, "openai_model", "gpt-4o"),
                temperature=0.0,
                openai_api_key=settings.openai_api_key,
            )
        return self._llm

    def analyze_for_scenarios_sync(
        self, query: str, search_results: List[Dict], use_llm: bool = True
    ) -> ScenarioAnalysisResult:
        """
        Synchronous version: Analyze search results to detect if multiple scenarios exist.

        Args:
            query: User's question
            search_results: List of document dicts with 'content' key
            use_llm: Whether to use LLM analysis (more accurate but slower)
        """
        if not search_results:
            return ScenarioAnalysisResult(
                has_multiple_scenarios=False,
                scenarios=[],
                clarification_question=None,
                direct_answer=None,
                confidence=0.0,
                reasoning="No search results",
            )

        # Combine search results
        combined_content = "\n\n---\n\n".join(
            [doc.get("content", "") for doc in search_results[:5]]
        )

        # Quick heuristic check first
        quick_check = self._quick_scenario_check(combined_content, query)

        if not quick_check["might_have_scenarios"]:
            # Likely direct answer - skip LLM analysis
            return ScenarioAnalysisResult(
                has_multiple_scenarios=False,
                scenarios=[],
                clarification_question=None,
                direct_answer=combined_content[:1000],
                confidence=0.9,
                reasoning="Quick check: appears to be direct answer",
            )

        if use_llm:
            return self._llm_scenario_analysis_sync(query, combined_content)
        else:
            return self._regex_scenario_analysis(query, combined_content)

    async def analyze_for_scenarios(
        self, query: str, search_results: List[Dict], use_llm: bool = True
    ) -> ScenarioAnalysisResult:
        """
        Async version: Analyze search results to detect if multiple scenarios exist.
        """
        if not search_results:
            return ScenarioAnalysisResult(
                has_multiple_scenarios=False,
                scenarios=[],
                clarification_question=None,
                direct_answer=None,
                confidence=0.0,
                reasoning="No search results",
            )

        combined_content = "\n\n---\n\n".join(
            [doc.get("content", "") for doc in search_results[:5]]
        )

        quick_check = self._quick_scenario_check(combined_content, query)

        if not quick_check["might_have_scenarios"]:
            return ScenarioAnalysisResult(
                has_multiple_scenarios=False,
                scenarios=[],
                clarification_question=None,
                direct_answer=combined_content[:1000],
                confidence=0.9,
                reasoning="Quick check: appears to be direct answer",
            )

        if use_llm:
            return await self._llm_scenario_analysis_async(query, combined_content)
        else:
            return self._regex_scenario_analysis(query, combined_content)

    def _quick_scenario_check(self, content: str, query: str) -> Dict:
        """Quick heuristic check before LLM analysis."""
        content_lower = content.lower()
        query_lower = query.lower()

        # Indicators of potential multiple scenarios
        scenario_indicators = [
            r"\bif you (are|have|want|need)\b",
            r"\b(option|scenario|case|situation)\s*[1-9a-c]\b",
            r"\bfor (new|existing|individual|business|admin|regular)\b",
            r"\b(either|alternatively|or you can)\b",
            r"\bdepending on\b",
            r"\bbased on (your|the)\b",
            r"\b(method|approach|way)\s*[1-9]\b",
            r"•\s*(if|when|for)\b",
        ]

        indicator_count = sum(
            1 for pattern in scenario_indicators if re.search(pattern, content_lower)
        )

        is_short = len(content) < 400

        is_factual_query = any(
            query_lower.startswith(p)
            for p in [
                "what is",
                "what are",
                "how much",
                "when is",
                "where is",
                "who is",
            ]
        )

        might_have_scenarios = indicator_count >= 2 and not (
            is_short and is_factual_query
        )

        return {
            "might_have_scenarios": might_have_scenarios,
            "indicator_count": indicator_count,
            "is_short": is_short,
            "is_factual_query": is_factual_query,
        }

    def _llm_scenario_analysis_sync(
        self, query: str, content: str
    ) -> ScenarioAnalysisResult:
        """Synchronous LLM analysis for scenarios."""
        prompt = self.SCENARIO_ANALYSIS_PROMPT.format(
            query=query, search_results=content[:4000]
        )

        try:
            # Use sync invoke
            response = self.llm.invoke([SystemMessage(content=prompt)])
            result = self._parse_scenario_response(response.content)

            return self._build_analysis_result(result)
        except Exception as e:
            print(f"LLM scenario analysis error: {e}")
            return self._regex_scenario_analysis(query, content)

    async def _llm_scenario_analysis_async(
        self, query: str, content: str
    ) -> ScenarioAnalysisResult:
        """Async LLM analysis for scenarios."""
        prompt = self.SCENARIO_ANALYSIS_PROMPT.format(
            query=query, search_results=content[:4000]
        )

        try:
            response = await self.llm.ainvoke([SystemMessage(content=prompt)])
            result = self._parse_scenario_response(response.content)

            return self._build_analysis_result(result)
        except Exception as e:
            print(f"LLM scenario analysis error: {e}")
            return self._regex_scenario_analysis(query, content)

    def _build_analysis_result(self, result: Dict) -> ScenarioAnalysisResult:
        """Build ScenarioAnalysisResult from parsed response."""
        return ScenarioAnalysisResult(
            has_multiple_scenarios=result.get("has_multiple_scenarios", False),
            scenarios=[
                DetectedScenario(
                    id=s.get("id", str(i + 1)),
                    title=s.get("title", f"Option {i+1}"),
                    condition=s.get("condition", ""),
                    description=s.get("description", ""),
                    keywords=s.get("keywords", []),
                )
                for i, s in enumerate(result.get("scenarios", []))
            ],
            clarification_question=result.get("clarification_question"),
            direct_answer=result.get("direct_answer"),
            confidence=result.get("confidence", 0.5),
            reasoning=result.get("reasoning", ""),
        )

    def _regex_scenario_analysis(
        self, query: str, content: str
    ) -> ScenarioAnalysisResult:
        """Fallback regex-based scenario detection."""
        scenarios = []

        patterns = [
            (r"(?:Option|Scenario|Case)\s*(\d+)[:\s]+([^.]+\.)", "numbered"),
            (r"If you (are|have|want)\s+([^,.:]+)[,:]?\s*([^.]+\.)", "conditional"),
            (
                r"For (new|existing|admin|regular)\s+([^,.:]+)[,:]?\s*([^.]+\.)",
                "user_type",
            ),
        ]

        scenario_id = 1
        for pattern, ptype in patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                groups = match.groups()
                scenarios.append(
                    DetectedScenario(
                        id=str(scenario_id),
                        title=f"{ptype.title()} {scenario_id}",
                        condition=" ".join(g for g in groups[:2] if g),
                        description=match.group(0)[:200],
                        keywords=[],
                    )
                )
                scenario_id += 1

        unique_scenarios = self._deduplicate_scenarios(scenarios)
        has_multiple = len(unique_scenarios) >= 2

        clarification_question = None
        if has_multiple:
            options = "\n".join(
                [
                    f"{i+1}. {s.condition or s.title}"
                    for i, s in enumerate(unique_scenarios[:5])
                ]
            )
            clarification_question = (
                f"I found different information depending on your situation. "
                f"Which of these applies to you?\n\n{options}\n\n"
                f"Please reply with the number or describe your situation."
            )

        return ScenarioAnalysisResult(
            has_multiple_scenarios=has_multiple,
            scenarios=unique_scenarios[:5],
            clarification_question=clarification_question,
            direct_answer=None if has_multiple else content[:500],
            confidence=0.6 if has_multiple else 0.8,
            reasoning="Regex pattern matching",
        )

    def _deduplicate_scenarios(
        self, scenarios: List[DetectedScenario]
    ) -> List[DetectedScenario]:
        """Remove duplicate/similar scenarios."""
        unique = []
        for s in scenarios:
            is_dup = False
            for existing in unique:
                s_words = set(s.condition.lower().split())
                e_words = set(existing.condition.lower().split())
                if len(s_words) > 0 and len(s_words & e_words) / len(s_words) > 0.7:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(s)
        return unique

    def _parse_scenario_response(self, response: str) -> Dict:
        """Parse LLM response to extract scenario information."""
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass

        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        brace_match = re.search(r"\{[\s\S]*\}", response)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        return {
            "has_multiple_scenarios": False,
            "scenarios": [],
            "direct_answer": response[:500],
            "confidence": 0.5,
            "reasoning": "Could not parse LLM response",
        }

    def generate_clarification_question_sync(
        self, query: str, scenarios: List[DetectedScenario]
    ) -> str:
        """Synchronous: Generate a natural clarification question."""
        if not scenarios:
            return "Could you provide more details about your specific situation?"

        scenarios_text = "\n".join(
            [f"{s.id}. {s.title}: {s.condition}" for s in scenarios[:5]]
        )

        prompt = self.CLARIFICATION_GENERATION_PROMPT.format(
            query=query, scenarios=scenarios_text
        )

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content.strip()
        except Exception as e:
            print(f"Error generating clarification: {e}")
            return self._fallback_clarification_question(scenarios)

    async def generate_clarification_question(
        self, query: str, scenarios: List[DetectedScenario]
    ) -> str:
        """Async: Generate a natural clarification question."""
        if not scenarios:
            return "Could you provide more details about your specific situation?"

        scenarios_text = "\n".join(
            [f"{s.id}. {s.title}: {s.condition}" for s in scenarios[:5]]
        )

        prompt = self.CLARIFICATION_GENERATION_PROMPT.format(
            query=query, scenarios=scenarios_text
        )

        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            return response.content.strip()
        except Exception as e:
            print(f"Error generating clarification: {e}")
            return self._fallback_clarification_question(scenarios)

    def _fallback_clarification_question(
        self, scenarios: List[DetectedScenario]
    ) -> str:
        """Generate fallback clarification question."""
        options = "\n".join(
            [f"{s.id}. {s.condition or s.title}" for s in scenarios[:5]]
        )
        return (
            f"I found information for different situations. "
            f"Which applies to you?\n\n{options}\n\n"
            f"Reply with a number or describe your case."
        )

    def extract_scenario_answer_sync(
        self, query: str, selection: str, scenarios: List[Dict], documents: List[Dict]
    ) -> str:
        """Synchronous: Extract the specific answer for a selected scenario."""
        scenarios_text = "\n".join(
            [
                f"{s.get('id', i+1)}. {s.get('title', '')}: {s.get('condition', '')} - {s.get('description', '')[:200]}"
                for i, s in enumerate(scenarios)
            ]
        )

        docs_text = "\n\n---\n\n".join([d.get("content", "") for d in documents[:3]])

        prompt = self.SCENARIO_EXTRACTION_PROMPT.format(
            query=query,
            selection=selection,
            scenarios=scenarios_text,
            documents=docs_text[:3000],
        )

        try:
            response = self.llm.invoke([SystemMessage(content=prompt)])
            return response.content.strip()
        except Exception as e:
            print(f"Error extracting scenario answer: {e}")
            return "I had trouble extracting the specific answer. Could you please rephrase your question?"

    async def extract_scenario_answer(
        self, query: str, selection: str, scenarios: List[Dict], documents: List[Dict]
    ) -> str:
        """Async: Extract the specific answer for a selected scenario."""
        scenarios_text = "\n".join(
            [
                f"{s.get('id', i+1)}. {s.get('title', '')}: {s.get('condition', '')} - {s.get('description', '')[:200]}"
                for i, s in enumerate(scenarios)
            ]
        )

        docs_text = "\n\n---\n\n".join([d.get("content", "") for d in documents[:3]])

        prompt = self.SCENARIO_EXTRACTION_PROMPT.format(
            query=query,
            selection=selection,
            scenarios=scenarios_text,
            documents=docs_text[:3000],
        )

        try:
            response = await self.llm.ainvoke([SystemMessage(content=prompt)])
            return response.content.strip()
        except Exception as e:
            print(f"Error extracting scenario answer: {e}")
            return "I had trouble extracting the specific answer. Could you please rephrase your question?"

    def match_user_response_to_scenario(
        self, response: str, scenarios: List[DetectedScenario]
    ) -> Optional[str]:
        """Match user's follow-up response to a scenario."""
        response_clean = response.strip().lower()

        # Direct number selection
        if response_clean.isdigit():
            num = int(response_clean)
            if 1 <= num <= len(scenarios):
                return str(num)

        # "Option X" or "Scenario X" patterns
        num_match = re.search(r"(?:option|scenario|number|#)\s*(\d+)", response_clean)
        if num_match:
            num = int(num_match.group(1))
            if 1 <= num <= len(scenarios):
                return str(num)

        # Word numbers
        word_to_num = {
            "first": "1",
            "second": "2",
            "third": "3",
            "fourth": "4",
            "fifth": "5",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
        }
        for word, num in word_to_num.items():
            if word in response_clean:
                if int(num) <= len(scenarios):
                    return num

        # Try keyword matching
        best_match = None
        best_score = 0
        response_words = set(response_clean.split())

        for i, scenario in enumerate(scenarios):
            keywords = set(k.lower() for k in scenario.keywords)
            condition_words = set(scenario.condition.lower().split())
            title_words = set(scenario.title.lower().split())

            all_scenario_words = keywords | condition_words | title_words
            overlap = len(response_words & all_scenario_words)

            if overlap > best_score and overlap >= 2:
                best_score = overlap
                best_match = str(i + 1)

        return best_match


# Singleton instance
scenario_handler = ScenarioHandler()
