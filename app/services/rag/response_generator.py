"""
Response Generator - Concise clarifications and document-only responses.
"""

import json
import re
from typing import List, Dict, Optional
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

from app.core.config import settings


class StrictResponseGenerator:
    """
    Generates concise clarifications and document-only responses.
    """

    STRICT_ANSWER_PROMPT = """Answer ONLY from the provided documents. No external knowledge.

## DOCUMENTS:
{documents}

## QUESTION:
{question}

## RULES:
1. Use ONLY document content
2. Be concise and direct
3. No advice or tips not in documents
4. If info is incomplete, say so

ANSWER:"""

    CONCISE_CLARIFICATION_PROMPT = """Generate a SHORT clarification question (max 2 sentences intro).

USER QUESTION: {question}

SCENARIOS (use these exact titles):
{scenarios}

FORMAT:
- One short intro sentence
- Numbered list with ONLY the titles
- End with "Reply with a number."

EXAMPLE OUTPUT:
Which situation applies to you?

1. Same sponsor
2. Different sponsor
3. Duplicate passport
4. Expired license

Reply with a number.

YOUR OUTPUT (keep it this concise):"""

    SCENARIO_ANSWER_PROMPT = """Answer for the selected scenario ONLY. Use ONLY document content.

QUESTION: {original_question}
SELECTED: {selected_scenario}

DOCUMENTS:
{documents}

Provide the specific answer for this scenario only. Be helpful and clear.

ANSWER:"""

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

    def generate_direct_answer(
        self, question: str, documents: List[Dict], session_id: str = ""
    ) -> str:
        """Generate answer strictly from documents."""
        if not documents:
            return "I don't have information about that in my knowledge base."

        docs_text = self._format_documents(documents)

        prompt = self.STRICT_ANSWER_PROMPT.format(
            documents=docs_text, question=question
        )

        try:
            response = self.llm.invoke([SystemMessage(content=prompt)])
            answer = response.content.strip() if response.content else ""
            return self._clean_response(answer)
        except Exception as e:
            print(f"[ResponseGenerator] Error: {e}")
            return self._fallback_response(documents)

    def generate_clarification_question(
        self, question: str, scenarios: List[Dict], session_id: str = ""
    ) -> str:
        """Generate a CONCISE clarification question."""
        if not scenarios:
            return "Could you provide more details?"

        # Format scenarios with short titles only
        scenarios_text = "\n".join(
            [f"{i+1}. {self._get_short_title(s)}" for i, s in enumerate(scenarios[:5])]
        )

        prompt = self.CONCISE_CLARIFICATION_PROMPT.format(
            question=question, scenarios=scenarios_text
        )

        try:
            response = self.llm.invoke([SystemMessage(content=prompt)])
            result = response.content.strip() if response.content else ""

            # Validate it's concise enough
            if len(result) > 300:
                return self._fallback_clarification(scenarios)

            return result

        except Exception as e:
            print(f"[ResponseGenerator] Clarification error: {e}")
            return self._fallback_clarification(scenarios)

    def generate_scenario_answer(
        self,
        original_question: str,
        selected_scenario: Dict,
        documents: List[Dict],
        session_id: str = "",
    ) -> str:
        """Generate answer for a specific scenario."""
        docs_text = self._format_documents(documents)

        scenario_info = f"{selected_scenario.get('title', 'Selected')}: {selected_scenario.get('condition', '')}"

        prompt = self.SCENARIO_ANSWER_PROMPT.format(
            original_question=original_question,
            selected_scenario=scenario_info,
            documents=docs_text,
        )

        try:
            response = self.llm.invoke([SystemMessage(content=prompt)])
            answer = response.content.strip() if response.content else ""
            return self._clean_response(answer)
        except Exception as e:
            print(f"[ResponseGenerator] Scenario answer error: {e}")
            return self._fallback_response(documents)

    def _get_short_title(self, scenario: Dict) -> str:
        """Get short title from scenario."""
        title = scenario.get("title", "")

        # If title is short enough, use it
        if title and len(title) <= 25:
            return title

        # Otherwise, try condition
        condition = scenario.get("condition", "")
        if condition:
            # Take first 25 chars
            short = condition[:25]
            if len(condition) > 25:
                short = short.rsplit(" ", 1)[0] + "..."
            return short

        return f"Option {scenario.get('id', '?')}"

    def _format_documents(self, documents: List[Dict]) -> str:
        """Format documents for prompt."""
        formatted = []
        for i, doc in enumerate(documents[:5], 1):
            content = doc.get("content", "")[:800]
            formatted.append(f"[Doc {i}] {content}")
        return "\n\n".join(formatted)

    def _clean_response(self, response: str) -> str:
        """Remove any hallucination patterns."""
        patterns = [
            r"(?i)^(based on|according to).*?[:,]\s*",
            r"(?i)\b(generally|typically|usually|normally)\b",
            r"(?i)it'?s? (important|recommended|advisable) to\b",
            r"(?i)\bkeep in mind\b",
            r"(?i)\bpro tip\b",
            r"(?i)\bnote:\s*",
        ]

        cleaned = response
        for pattern in patterns:
            cleaned = re.sub(pattern, "", cleaned)

        return cleaned.strip()

    def _fallback_response(self, documents: List[Dict]) -> str:
        """Fallback to first document content."""
        if not documents:
            return "I don't have information about that."

        content = documents[0].get("content", "")
        return content[:500] if content else "Information not available."

    def _fallback_clarification(self, scenarios: List[Dict]) -> str:
        """Fallback concise clarification."""
        options = "\n".join(
            [f"{i+1}. {self._get_short_title(s)}" for i, s in enumerate(scenarios[:5])]
        )

        return f"Which applies to you?\n\n{options}\n\nReply with a number."


# Singleton
response_generator = StrictResponseGenerator()
