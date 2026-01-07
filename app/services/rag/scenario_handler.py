"""
Scenario Handler - Consistent detection regardless of conversation history.
"""

import json
import re
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

from app.core.config import settings
from .config import Config


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
    Handles scenario detection consistently.
    Always detects scenarios the same way regardless of conversation history.
    """

    # Deterministic prompt - no conversation context to ensure consistency
    SCENARIO_DETECTION_PROMPT = """Analyze these search results and determine if there are MULTIPLE DISTINCT SCENARIOS that require different answers.

## USER QUESTION:
{query}

## SEARCH RESULTS:
{search_results}

## ANALYSIS RULES:
1. Look for DISTINCT scenarios/conditions (e.g., "If you are X..." vs "If you are Y...")
2. Each scenario should have a DIFFERENT solution/answer
3. Ignore minor variations - only count truly different paths
4. Be CONSISTENT - the same content should always produce the same result

## SCENARIO INDICATORS TO LOOK FOR:
- "If you are/have..." patterns
- "Scenario 1:", "Scenario 2:" labels
- "For [type] users..." patterns
- "Option A:", "Option B:" patterns
- "In case of..." patterns

## RESPOND WITH ONLY THIS JSON:
{{
    "has_multiple_scenarios": true/false,
    "scenario_count": 0,
    "scenarios": [
        {{
            "id": "1",
            "title": "brief title",
            "condition": "when this applies",
            "description": "what to do in this scenario"
        }}
    ],
    "reasoning": "why I detected these scenarios"
}}

IMPORTANT: Be deterministic. The same input should ALWAYS produce the same output."""

    def __init__(self):
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=getattr(settings, "openai_model", "gpt-4o"),
                temperature=0.0,  # Deterministic - crucial for consistency
                openai_api_key=settings.openai_api_key,
            )
        return self._llm

    def analyze_for_scenarios_sync(
        self, query: str, search_results: List[Dict], session_id: str = ""
    ) -> ScenarioAnalysisResult:
        """
        Analyze search results for multiple scenarios.
        This is STATELESS - does not consider conversation history.
        """
        print(f"\n[ScenarioHandler] === FRESH ANALYSIS (no history) ===")
        print(f"[ScenarioHandler] Query: {query}")
        print(f"[ScenarioHandler] Documents: {len(search_results)}")

        if not search_results:
            return self._empty_result("No search results")

        # Combine document content
        combined_content = "\n\n---\n\n".join(
            [doc.get("content", "") for doc in search_results[:5]]
        )

        # Step 1: Quick regex check for scenarios
        regex_scenarios = self._regex_scenario_detection(combined_content)
        print(f"[ScenarioHandler] Regex found {len(regex_scenarios)} scenarios")

        # Step 2: If regex finds scenarios, use LLM to validate and structure
        if len(regex_scenarios) >= Config.MIN_SCENARIOS_FOR_CLARIFICATION:
            return self._llm_analysis(query, combined_content, session_id)

        # Step 3: If regex finds nothing, still do LLM check for edge cases
        if self._might_have_scenarios(combined_content):
            return self._llm_analysis(query, combined_content, session_id)

        # Direct answer - no scenarios
        print(f"[ScenarioHandler] No scenarios detected, returning direct answer")
        return ScenarioAnalysisResult(
            has_multiple_scenarios=False,
            scenarios=[],
            clarification_question=None,
            direct_answer=combined_content[:1000],
            confidence=0.9,
            reasoning="No multiple scenarios detected",
        )

    def _regex_scenario_detection(self, content: str) -> List[Dict]:
        """Deterministic regex-based scenario detection."""
        scenarios = []

        # Pattern 1: "Scenario X:" format
        scenario_pattern = r"Scenario\s*(\d+)\s*[:\-]\s*([^\n]+)"
        for match in re.finditer(scenario_pattern, content, re.IGNORECASE):
            scenarios.append(
                {
                    "id": match.group(1),
                    "title": f"Scenario {match.group(1)}",
                    "condition": match.group(2).strip(),
                    "description": match.group(0),
                }
            )

        # Pattern 2: "If you are/have..." format
        if_pattern = (
            r"[Ii]f\s+(?:you\s+)?(are|have|want|need)\s+([^,.:]+)[,.:]\s*([^.]+\.?)"
        )
        for i, match in enumerate(re.finditer(if_pattern, content)):
            scenarios.append(
                {
                    "id": str(len(scenarios) + 1),
                    "title": f"Condition {len(scenarios) + 1}",
                    "condition": f"If you {match.group(1)} {match.group(2)}".strip(),
                    "description": match.group(3).strip()[:200],
                }
            )

        # Pattern 3: "Option/Case X:" format
        option_pattern = r"(?:Option|Case|Method)\s*(\d+|[A-C])\s*[:\-]\s*([^\n]+)"
        for match in re.finditer(option_pattern, content, re.IGNORECASE):
            scenarios.append(
                {
                    "id": str(len(scenarios) + 1),
                    "title": f"Option {match.group(1)}",
                    "condition": match.group(2).strip(),
                    "description": match.group(0),
                }
            )

        # Deduplicate
        seen = set()
        unique = []
        for s in scenarios:
            key = s["condition"].lower()[:50]
            if key not in seen:
                seen.add(key)
                unique.append(s)

        return unique

    def _might_have_scenarios(self, content: str) -> bool:
        """Quick check if content might have scenarios."""
        indicators = [
            r"\bscenario\s*\d",
            r"\boption\s*\d",
            r"\bcase\s*\d",
            r"\bif you (are|have|want|need)\b",
            r"\bdepending on\b",
            r"\bbased on your\b",
        ]

        content_lower = content.lower()
        matches = sum(1 for p in indicators if re.search(p, content_lower))

        return matches >= 2

    def _llm_analysis(
        self, query: str, content: str, session_id: str = ""
    ) -> ScenarioAnalysisResult:
        """LLM-based scenario analysis - deterministic."""
        print(f"[ScenarioHandler] Running LLM analysis...")

        prompt = self.SCENARIO_DETECTION_PROMPT.format(
            query=query, search_results=content[:4000]
        )

        try:
            response = self.llm.invoke([SystemMessage(content=prompt)])
            result = self._parse_response(response.content)

            scenarios = [
                DetectedScenario(
                    id=s.get("id", str(i + 1)),
                    title=s.get("title", f"Scenario {i+1}"),
                    condition=s.get("condition", ""),
                    description=s.get("description", ""),
                    keywords=[],
                )
                for i, s in enumerate(result.get("scenarios", []))
            ]

            has_multiple = len(scenarios) >= Config.MIN_SCENARIOS_FOR_CLARIFICATION

            print(f"[ScenarioHandler] LLM detected {len(scenarios)} scenarios")
            print(f"[ScenarioHandler] Has multiple: {has_multiple}")

            clarification = None
            if has_multiple:
                clarification = self._generate_clarification(query, scenarios)

            return ScenarioAnalysisResult(
                has_multiple_scenarios=has_multiple,
                scenarios=scenarios,
                clarification_question=clarification,
                direct_answer=None if has_multiple else content[:1000],
                confidence=result.get("confidence", 0.8),
                reasoning=result.get("reasoning", "LLM analysis"),
            )

        except Exception as e:
            print(f"[ScenarioHandler] LLM error: {e}")
            # Fallback to regex-only
            regex_scenarios = self._regex_scenario_detection(content)
            return self._build_from_regex(query, content, regex_scenarios)

    def _generate_clarification(
        self, query: str, scenarios: List[DetectedScenario]
    ) -> str:
        """Generate clarification question."""
        options = "\n".join(
            [f"{i+1}. {s.condition or s.title}" for i, s in enumerate(scenarios[:5])]
        )

        return (
            f"I found different solutions depending on your specific situation. "
            f"Which of these applies to you?\n\n{options}\n\n"
            f"Please reply with a number or describe your situation."
        )

    def _build_from_regex(
        self, query: str, content: str, regex_scenarios: List[Dict]
    ) -> ScenarioAnalysisResult:
        """Build result from regex detection."""
        scenarios = [
            DetectedScenario(
                id=s.get("id", str(i + 1)),
                title=s.get("title", f"Scenario {i+1}"),
                condition=s.get("condition", ""),
                description=s.get("description", ""),
                keywords=[],
            )
            for i, s in enumerate(regex_scenarios)
        ]

        has_multiple = len(scenarios) >= Config.MIN_SCENARIOS_FOR_CLARIFICATION

        clarification = None
        if has_multiple:
            clarification = self._generate_clarification(query, scenarios)

        return ScenarioAnalysisResult(
            has_multiple_scenarios=has_multiple,
            scenarios=scenarios,
            clarification_question=clarification,
            direct_answer=None if has_multiple else content[:1000],
            confidence=0.7,
            reasoning="Regex-based detection",
        )

    def _parse_response(self, response: str) -> Dict:
        """Parse LLM response."""
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass

        # Try code block
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response)
        if match:
            try:
                return json.loads(match.group(1))
            except:
                pass

        # Try brace match
        match = re.search(r"\{[\s\S]*\}", response)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                pass

        return {"has_multiple_scenarios": False, "scenarios": []}

    def _empty_result(self, reason: str) -> ScenarioAnalysisResult:
        """Return empty result."""
        return ScenarioAnalysisResult(
            has_multiple_scenarios=False,
            scenarios=[],
            clarification_question=None,
            direct_answer=None,
            confidence=0.0,
            reasoning=reason,
        )

    def match_user_response_to_scenario(
        self, response: str, scenarios: List[DetectedScenario]
    ) -> Optional[str]:
        """Match user selection to scenario."""
        response_clean = response.strip().lower()

        # Direct number
        if response_clean.isdigit():
            num = int(response_clean)
            if 1 <= num <= len(scenarios):
                return str(num)

        # "Option X" pattern
        match = re.search(r"(?:option|scenario|number|#)\s*(\d+)", response_clean)
        if match:
            num = int(match.group(1))
            if 1 <= num <= len(scenarios):
                return str(num)

        # Word numbers
        words = {"first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5"}
        for word, num in words.items():
            if word in response_clean and int(num) <= len(scenarios):
                return num

        # Keyword matching
        response_words = set(response_clean.split())
        best_match = None
        best_score = 0

        for i, scenario in enumerate(scenarios):
            scenario_words = set(scenario.condition.lower().split())
            overlap = len(response_words & scenario_words)
            if overlap > best_score and overlap >= 2:
                best_score = overlap
                best_match = str(i + 1)

        return best_match


# Singleton
scenario_handler = ScenarioHandler()
