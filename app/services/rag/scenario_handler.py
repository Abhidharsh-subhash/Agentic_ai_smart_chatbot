"""
Scenario Handler - Consistent detection with concise clarification questions.
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
    title: str  # Short title (2-5 words)
    condition: str  # Brief condition
    description: str  # Full description for answer generation
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
    Handles scenario detection with concise clarification questions.
    """

    # Updated prompt for CONCISE scenario titles
    SCENARIO_DETECTION_PROMPT = """Analyze these search results and identify DISTINCT SCENARIOS that need different solutions.

## USER QUESTION:
{query}

## SEARCH RESULTS:
{search_results}

## YOUR TASK:
1. Find distinct scenarios/conditions with DIFFERENT solutions
2. Create SHORT titles (2-5 words max) for each scenario
3. Be consistent - same content = same result

## RESPOND WITH ONLY THIS JSON:
{{
    "has_multiple_scenarios": true/false,
    "scenarios": [
        {{
            "id": "1",
            "title": "SHORT 2-5 word title",
            "condition": "brief one-line condition",
            "description": "full description for answering"
        }}
    ],
    "reasoning": "brief explanation"
}}

## EXAMPLE TITLES (keep them this short):
- "Same sponsor exists"
- "Different sponsor"
- "Duplicate passport"
- "Expired license"
- "Invalid format"
- "New customer"
- "Existing customer"
- "Admin user"
- "Regular user"

Keep titles SHORT and CLEAR."""

    def __init__(self):
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=getattr(settings, "openai_model", "gpt-4o"),
                temperature=0.0,
                openai_api_key=settings.openai_api_key,
            )
        return self._llm

    def analyze_for_scenarios_sync(
        self, query: str, search_results: List[Dict], session_id: str = ""
    ) -> ScenarioAnalysisResult:
        """Analyze search results for multiple scenarios."""
        print(f"\n[ScenarioHandler] Analyzing: {query}")

        if not search_results:
            return self._empty_result("No search results")

        combined_content = "\n\n---\n\n".join(
            [doc.get("content", "") for doc in search_results[:5]]
        )

        # Quick regex check
        regex_scenarios = self._regex_scenario_detection(combined_content)
        print(f"[ScenarioHandler] Regex found: {len(regex_scenarios)} scenarios")

        if len(regex_scenarios) >= Config.MIN_SCENARIOS_FOR_CLARIFICATION:
            result = self._llm_analysis(query, combined_content, session_id)
        elif self._might_have_scenarios(combined_content):
            result = self._llm_analysis(query, combined_content, session_id)
        else:
            result = ScenarioAnalysisResult(
                has_multiple_scenarios=False,
                scenarios=[],
                clarification_question=None,
                direct_answer=combined_content[:1000],
                confidence=0.9,
                reasoning="No multiple scenarios detected",
            )

        # Generate concise clarification if needed
        if result.has_multiple_scenarios and result.scenarios:
            result.clarification_question = self._generate_concise_clarification(
                query, result.scenarios
            )

        return result

    def _generate_concise_clarification(
        self, query: str, scenarios: List[DetectedScenario]
    ) -> str:
        """Generate a CONCISE clarification question."""
        # Short intro (1 sentence)
        intro = "Which situation applies to you?"

        # Short numbered options (title only, max 5)
        options = []
        for i, s in enumerate(scenarios[:5], 1):
            # Use title if short enough, otherwise truncate condition
            label = s.title if len(s.title) <= 30 else s.title[:27] + "..."
            options.append(f"{i}. {label}")

        options_text = "\n".join(options)

        # Concise format
        return f"{intro}\n\n{options_text}\n\nReply with a number."

    def _regex_scenario_detection(self, content: str) -> List[Dict]:
        """Regex-based scenario detection with SHORT titles."""
        scenarios = []
        seen_conditions = set()

        # Pattern 1: "Scenario X:" format
        for match in re.finditer(
            r"Scenario\s*(\d+)\s*[:\-]\s*([^\n]+)", content, re.IGNORECASE
        ):
            condition = match.group(2).strip()
            short_title = self._create_short_title(condition)

            if short_title.lower() not in seen_conditions:
                seen_conditions.add(short_title.lower())
                scenarios.append(
                    {
                        "id": str(len(scenarios) + 1),
                        "title": short_title,
                        "condition": condition[:100],
                        "description": match.group(0),
                    }
                )

        # Pattern 2: "If you are/have..." format
        for match in re.finditer(
            r"[Ii]f\s+(?:you\s+)?(are|have|want|need|the)\s+([^,.:]+)", content
        ):
            condition = f"{match.group(1)} {match.group(2)}".strip()
            short_title = self._create_short_title(condition)

            if short_title.lower() not in seen_conditions:
                seen_conditions.add(short_title.lower())
                scenarios.append(
                    {
                        "id": str(len(scenarios) + 1),
                        "title": short_title,
                        "condition": condition[:100],
                        "description": match.group(0),
                    }
                )

        # Pattern 3: Common issue patterns
        issue_patterns = [
            (
                r"already exists?\s+(?:in\s+)?(?:with\s+)?(?:the\s+)?(?:same\s+)?(\w+)",
                "Duplicate {}",
            ),
            (
                r"(?:license|establishment)\s+(?:has\s+)?(?:expired|ban)",
                "License/ban issue",
            ),
            (r"(?:invalid|incorrect)\s+(\w+)", "Invalid {}"),
            (r"(?:different|another)\s+sponsor", "Different sponsor"),
            (r"same\s+sponsor", "Same sponsor"),
        ]

        for pattern, title_template in issue_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                try:
                    title = (
                        title_template.format(match.group(1))
                        if "{}" in title_template
                        else title_template
                    )
                except:
                    title = title_template.replace("{}", "")

                title = self._create_short_title(title)
                if title.lower() not in seen_conditions:
                    seen_conditions.add(title.lower())
                    scenarios.append(
                        {
                            "id": str(len(scenarios) + 1),
                            "title": title,
                            "condition": match.group(0)[:100],
                            "description": match.group(0),
                        }
                    )

        return scenarios[:5]  # Max 5

    def _create_short_title(self, text: str) -> str:
        """Create a short 2-5 word title from text."""
        # Remove common filler words
        text = re.sub(
            r"\b(the|a|an|is|are|was|were|been|being|have|has|had|do|does|did|will|would|could|should|may|might|must|shall|can|need|dare|ought|used|to|of|in|for|on|with|at|by|from|as|into|through|during|before|after|above|below|between|under|again|further|then|once|here|there|when|where|why|how|all|each|every|both|few|more|most|other|some|such|no|nor|not|only|own|same|so|than|too|very|just|also|now|and|or|but|if|because|until|while|although|though|even|whether|that|which|who|whom|whose|this|these|those|what|whatever|whichever|whoever|whomever|however|whenever|wherever)\b",
            " ",
            text,
            flags=re.IGNORECASE,
        )

        # Clean up
        text = re.sub(r"\s+", " ", text).strip()

        # Take first 5 meaningful words
        words = text.split()[:5]

        if not words:
            return "Option"

        # Capitalize first letter of each word
        title = " ".join(w.capitalize() for w in words)

        # Ensure max 30 chars
        if len(title) > 30:
            title = title[:27] + "..."

        return title

    def _might_have_scenarios(self, content: str) -> bool:
        """Quick check for potential scenarios."""
        indicators = [
            r"\bscenario\s*\d",
            r"\boption\s*\d",
            r"\bif you (are|have)\b",
            r"\bdepending on\b",
        ]

        content_lower = content.lower()
        return sum(1 for p in indicators if re.search(p, content_lower)) >= 2

    def _llm_analysis(
        self, query: str, content: str, session_id: str = ""
    ) -> ScenarioAnalysisResult:
        """LLM-based scenario analysis with concise titles."""
        print(f"[ScenarioHandler] Running LLM analysis...")

        prompt = self.SCENARIO_DETECTION_PROMPT.format(
            query=query, search_results=content[:4000]
        )

        try:
            response = self.llm.invoke([SystemMessage(content=prompt)])
            result = self._parse_response(response.content)

            scenarios = []
            for i, s in enumerate(result.get("scenarios", [])):
                title = s.get("title", f"Option {i+1}")
                # Ensure title is short
                if len(title) > 30:
                    title = self._create_short_title(title)

                scenarios.append(
                    DetectedScenario(
                        id=s.get("id", str(i + 1)),
                        title=title,
                        condition=s.get("condition", "")[:100],
                        description=s.get("description", ""),
                        keywords=[],
                    )
                )

            has_multiple = len(scenarios) >= Config.MIN_SCENARIOS_FOR_CLARIFICATION

            print(f"[ScenarioHandler] LLM found {len(scenarios)} scenarios")

            return ScenarioAnalysisResult(
                has_multiple_scenarios=has_multiple,
                scenarios=scenarios,
                clarification_question=None,  # Will be generated separately
                direct_answer=None if has_multiple else content[:1000],
                confidence=0.8,
                reasoning=result.get("reasoning", "LLM analysis"),
            )

        except Exception as e:
            print(f"[ScenarioHandler] LLM error: {e}")
            return self._build_from_regex(query, content)

    def _build_from_regex(self, query: str, content: str) -> ScenarioAnalysisResult:
        """Build result from regex."""
        regex_scenarios = self._regex_scenario_detection(content)

        scenarios = [
            DetectedScenario(
                id=s.get("id", str(i + 1)),
                title=s.get("title", f"Option {i+1}"),
                condition=s.get("condition", ""),
                description=s.get("description", ""),
                keywords=[],
            )
            for i, s in enumerate(regex_scenarios)
        ]

        has_multiple = len(scenarios) >= Config.MIN_SCENARIOS_FOR_CLARIFICATION

        return ScenarioAnalysisResult(
            has_multiple_scenarios=has_multiple,
            scenarios=scenarios,
            clarification_question=None,
            direct_answer=None if has_multiple else content[:500],
            confidence=0.7,
            reasoning="Regex detection",
        )

    def _parse_response(self, response: str) -> Dict:
        """Parse LLM JSON response."""
        try:
            return json.loads(response.strip())
        except:
            pass

        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response)
        if match:
            try:
                return json.loads(match.group(1))
            except:
                pass

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

        # Keyword matching against titles
        response_words = set(response_clean.split())
        best_match = None
        best_score = 0

        for i, scenario in enumerate(scenarios):
            title_words = set(scenario.title.lower().split())
            condition_words = set(scenario.condition.lower().split())
            all_words = title_words | condition_words

            overlap = len(response_words & all_words)
            if overlap > best_score and overlap >= 1:
                best_score = overlap
                best_match = str(i + 1)

        return best_match


# Singleton
scenario_handler = ScenarioHandler()
