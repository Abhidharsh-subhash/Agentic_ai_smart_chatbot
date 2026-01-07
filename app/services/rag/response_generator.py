"""
Response Generator - Ensures responses are ONLY from documents.
No external knowledge allowed.
"""

import json
import re
from typing import List, Dict, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from app.core.config import settings


class StrictResponseGenerator:
    """
    Generates responses strictly from document content.
    No external knowledge or hallucination allowed.
    """

    STRICT_ANSWER_PROMPT = """You are a support assistant that answers ONLY from the provided documents.

## CRITICAL RULES - YOU MUST FOLLOW THESE:

1. **ONLY USE DOCUMENT CONTENT**: Your answer must come ONLY from the documents below. 
2. **NO EXTERNAL KNOWLEDGE**: Do NOT add any information not in the documents.
3. **NO ASSUMPTIONS**: Do NOT assume or infer anything beyond what's written.
4. **NO ADVICE**: Do NOT add tips, warnings, or suggestions not in the documents.
5. **QUOTE WHEN POSSIBLE**: Prefer to quote or closely paraphrase the documents.
6. **ADMIT LIMITATIONS**: If the documents don't fully answer, say "Based on the available documentation, I can only confirm that..." 

## DOCUMENTS:
{documents}

## USER QUESTION:
{question}

## ANSWER FORMAT:
- Start directly with the answer
- Do NOT say "Based on the documents..." or "According to..."
- Be concise and factual
- Only include what the documents explicitly state

YOUR ANSWER (document content only):"""

    SCENARIO_CLARIFICATION_PROMPT = """You are a support assistant. The user's question has multiple possible answers depending on their situation.

## CRITICAL RULES:
1. You MUST ask the user to clarify which scenario applies to them
2. List ONLY the scenarios found in the documents
3. Do NOT provide answers yet - just ask for clarification
4. Be friendly but concise

## USER QUESTION:
{question}

## DETECTED SCENARIOS FROM DOCUMENTS:
{scenarios}

## YOUR TASK:
Generate a friendly clarification question that:
1. Acknowledges their question briefly
2. Explains that the answer depends on their specific situation
3. Lists the scenarios as numbered options (1, 2, 3, etc.)
4. Asks them to reply with a number or describe their situation

CLARIFICATION QUESTION:"""

    SCENARIO_ANSWER_PROMPT = """You are a support assistant. The user selected a specific scenario. Answer ONLY for that scenario.

## CRITICAL RULES:
1. Answer ONLY for the selected scenario - ignore other scenarios
2. Use ONLY information from the documents
3. Do NOT add external knowledge or advice
4. Be specific to the user's selected situation

## ORIGINAL QUESTION:
{original_question}

## USER SELECTED SCENARIO:
{selected_scenario}

## RELEVANT DOCUMENTS:
{documents}

## YOUR TASK:
Provide the answer that applies ONLY to the selected scenario.
Use only document content. Be helpful and clear.

ANSWER FOR SELECTED SCENARIO:"""

    VALIDATION_PROMPT = """Check if this response contains only information from the source documents.

## SOURCE DOCUMENTS:
{documents}

## GENERATED RESPONSE:
{response}

## CHECK:
Does the response contain ANY information not found in the source documents?
Respond with JSON only:
{{"is_valid": true/false, "issues": ["list of external info found if any"], "cleaned_response": "response with external info removed"}}"""

    def __init__(self):
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=getattr(settings, "openai_model", "gpt-4o"),
                temperature=0.0,  # Deterministic responses
                openai_api_key=settings.openai_api_key,
            )
        return self._llm

    def generate_direct_answer(
        self, question: str, documents: List[Dict], session_id: str = ""
    ) -> str:
        """Generate answer strictly from documents."""
        if not documents:
            return "I don't have information about that in my knowledge base."

        # Combine document content
        docs_text = self._format_documents(documents)

        prompt = self.STRICT_ANSWER_PROMPT.format(
            documents=docs_text, question=question
        )

        try:
            response = self.llm.invoke([SystemMessage(content=prompt)])
            answer = response.content.strip() if response.content else ""

            # Validate and clean the response
            validated = self._validate_response(answer, documents)

            return validated

        except Exception as e:
            print(f"[ResponseGenerator] Error: {e}")
            # Fallback: return first document content
            return self._fallback_response(documents)

    def generate_clarification_question(
        self, question: str, scenarios: List[Dict], session_id: str = ""
    ) -> str:
        """Generate clarification question for multiple scenarios."""
        if not scenarios:
            return "Could you provide more details about your specific situation?"

        # Format scenarios
        scenarios_text = "\n".join(
            [
                f"{i+1}. {s.get('condition') or s.get('title', f'Scenario {i+1}')}: {s.get('description', '')[:100]}"
                for i, s in enumerate(scenarios[:5])
            ]
        )

        prompt = self.SCENARIO_CLARIFICATION_PROMPT.format(
            question=question, scenarios=scenarios_text
        )

        try:
            response = self.llm.invoke([SystemMessage(content=prompt)])
            return (
                response.content.strip()
                if response.content
                else self._fallback_clarification(scenarios)
            )
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
        """Generate answer for a specific selected scenario."""
        docs_text = self._format_documents(documents)

        scenario_text = f"{selected_scenario.get('title', 'Selected scenario')}: {selected_scenario.get('condition', '')} - {selected_scenario.get('description', '')}"

        prompt = self.SCENARIO_ANSWER_PROMPT.format(
            original_question=original_question,
            selected_scenario=scenario_text,
            documents=docs_text,
        )

        try:
            response = self.llm.invoke([SystemMessage(content=prompt)])
            answer = response.content.strip() if response.content else ""

            # Validate
            validated = self._validate_response(answer, documents)

            return validated

        except Exception as e:
            print(f"[ResponseGenerator] Scenario answer error: {e}")
            return self._fallback_response(documents)

    def _format_documents(self, documents: List[Dict]) -> str:
        """Format documents for prompt."""
        formatted = []
        for i, doc in enumerate(documents[:5], 1):
            content = doc.get("content", "")
            formatted.append(f"[Document {i}]\n{content}")
        return "\n\n---\n\n".join(formatted)

    def _validate_response(self, response: str, documents: List[Dict]) -> str:
        """Validate that response only contains document information."""
        # Quick validation - check for common hallucination patterns
        hallucination_patterns = [
            r"(?i)generally speaking",
            r"(?i)in most cases",
            r"(?i)it'?s? (important|recommended|advisable) to",
            r"(?i)you (should|might|may) (also )?consider",
            r"(?i)based on (my|general) knowledge",
            r"(?i)typically,?\s",
            r"(?i)usually,?\s",
            r"(?i)as a (best practice|general rule)",
            r"(?i)i would (recommend|suggest|advise)",
            r"(?i)keep in mind that",
            r"(?i)it'?s? worth noting",
            r"(?i)additionally,? you (may|might|should)",
            r"(?i)for (best|better) results",
            r"(?i)pro tip:",
            r"(?i)note:",
            r"(?i)important:",
            r"(?i)remember to",
            r"(?i)don'?t forget to",
            r"(?i)make sure to",
        ]

        cleaned = response
        for pattern in hallucination_patterns:
            # Find sentences containing these patterns
            sentences = re.split(r"(?<=[.!?])\s+", cleaned)
            filtered = []
            for sentence in sentences:
                if not re.search(pattern, sentence):
                    filtered.append(sentence)
            cleaned = " ".join(filtered)

        # If we removed too much, return original with disclaimer
        if len(cleaned) < len(response) * 0.5 and len(cleaned) < 50:
            return response

        return cleaned.strip() if cleaned.strip() else response

    def _fallback_response(self, documents: List[Dict]) -> str:
        """Fallback: return document content directly."""
        if not documents:
            return "I don't have information about that."

        content = documents[0].get("content", "")
        if len(content) > 500:
            content = content[:500] + "..."

        return content

    def _fallback_clarification(self, scenarios: List[Dict]) -> str:
        """Fallback clarification question."""
        options = "\n".join(
            [
                f"{i+1}. {s.get('condition') or s.get('title', f'Option {i+1}')}"
                for i, s in enumerate(scenarios[:5])
            ]
        )

        return (
            f"I found different solutions depending on your situation. "
            f"Which of these applies to you?\n\n{options}\n\n"
            f"Please reply with a number."
        )


# Singleton
response_generator = StrictResponseGenerator()
