# app/services/rag/tools.py
"""
Tools - matching standalone logic exactly.
"""
import json
from typing import List
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from app.services.embeddings import embedding_service
from app.core.config import settings
from .config import Config, SearchQuality


# Shared LLM for scenario detection
_llm = None


def get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=getattr(settings, "openai_model", "gpt-4o"),
            temperature=0,
            openai_api_key=settings.openai_api_key,
        )
    return _llm


class SearchResultAnalyzer:
    """Analyzes search results - matching standalone exactly."""

    @classmethod
    def analyze(cls, results: list, query: str) -> dict:
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
    """Detects scenarios - matching standalone exactly."""

    DETECTION_PROMPT = """Analyze the following search results and determine if they contain MULTIPLE DISTINCT scenarios that the user needs to choose between.

IMPORTANT RULES:
1. ONLY identify scenarios that are EXPLICITLY mentioned in the search results
2. DO NOT guess or infer scenarios that are not in the content
3. If the content describes ONE process/procedure, there is NO disambiguation needed
4. Multiple scenarios exist ONLY if the content explicitly mentions different cases like:
   - "For users with X, do this... For users with Y, do that..."
   - "Option A: ... Option B: ..."
   - "If you are an admin... If you are a regular user..."

USER QUERY: {query}

SEARCH RESULTS CONTENT:
{search_results}

RESPOND IN JSON FORMAT:
{{
    "has_multiple_scenarios": true/false,
    "scenarios": [
        {{
            "id": "scenario_1",
            "title": "<brief title from content>",
            "description": "<description from content>",
            "exact_quote": "<quote from content that describes this scenario>"
        }}
    ],
    "disambiguation_needed": true/false,
    "suggested_question": "<question ONLY if disambiguation_needed is true>",
    "reason": "<explanation>"
}}

CRITICAL: Set has_multiple_scenarios to FALSE unless the content EXPLICITLY contains multiple distinct procedures/options that require the user to choose."""

    @classmethod
    def detect(cls, query: str, documents: List[dict], llm: ChatOpenAI) -> dict:
        formatted_results = "\n\n---\n\n".join(
            [doc.get("content", "") for doc in documents[:5]]
        )

        prompt = cls.DETECTION_PROMPT.format(
            query=query, search_results=formatted_results
        )

        try:
            response = llm.invoke(
                [
                    SystemMessage(
                        content="You are a strict scenario detector. Only identify scenarios EXPLICITLY present in the content. Never speculate."
                    ),
                    HumanMessage(content=prompt),
                ]
            )

            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            result = json.loads(content.strip())

            # Validate - require exact quotes
            scenarios = result.get("scenarios", [])
            valid_scenarios = [s for s in scenarios if s.get("exact_quote")]

            if len(valid_scenarios) < Config.MIN_SCENARIOS_FOR_DISAMBIGUATION:
                result["has_multiple_scenarios"] = False
                result["disambiguation_needed"] = False

            result["scenarios"] = valid_scenarios
            return result

        except Exception as e:
            print(f"[ScenarioDetector] Error: {e}")
            return {
                "has_multiple_scenarios": False,
                "scenarios": [],
                "disambiguation_needed": False,
                "suggested_question": "",
                "reason": f"Error: {str(e)}",
            }


@tool
def search_and_analyze(query: str) -> str:
    """
    Search the document database and analyze if multiple scenarios exist.
    This is the primary tool - always call this first.
    """
    print(f"\n[search_and_analyze] Query: '{query}'")

    if not embedding_service.index_file.exists():
        return json.dumps(
            {
                "found_answer": False,
                "quality": SearchQuality.NOT_FOUND.value,
                "confidence": 0.0,
                "documents": [],
                "has_multiple_scenarios": False,
                "disambiguation_needed": False,
                "message": "No knowledge base available",
            }
        )

    try:
        results = embedding_service.search_with_scores(query, k=5)
        print(f"[search_and_analyze] Found {len(results)} raw results")
    except Exception as e:
        return json.dumps(
            {
                "found_answer": False,
                "error": str(e),
                "has_multiple_scenarios": False,
            }
        )

    if not results:
        return json.dumps(
            {
                "found_answer": False,
                "quality": SearchQuality.NOT_FOUND.value,
                "confidence": 0.0,
                "documents": [],
                "has_multiple_scenarios": False,
                "disambiguation_needed": False,
                "message": "No information found for this query in the knowledge base.",
            }
        )

    # Analyze search quality
    analysis = SearchResultAnalyzer.analyze(results, query)
    print(
        f"[search_and_analyze] Quality: {analysis['quality']}, should_respond: {analysis['should_respond']}"
    )

    if not analysis["should_respond"]:
        return json.dumps(
            {
                "found_answer": False,
                "quality": analysis["quality"],
                "confidence": float(analysis["confidence"]),
                "documents": [],
                "has_multiple_scenarios": False,
                "disambiguation_needed": False,
                "message": "No relevant information found for this query.",
                "reason": analysis["reason"],
            }
        )

    # Prepare documents
    documents = []
    for doc, score in results:
        if float(score) < Config.ACCEPTABLE_SCORE:
            documents.append(
                {
                    "content": doc.page_content,
                    "relevance": (
                        "high" if float(score) < Config.GOOD_SCORE else "medium"
                    ),
                    "score": float(score),
                }
            )

    if not documents:
        return json.dumps(
            {
                "found_answer": False,
                "quality": SearchQuality.NOT_FOUND.value,
                "has_multiple_scenarios": False,
                "disambiguation_needed": False,
                "message": "No relevant information found.",
            }
        )

    print(f"[search_and_analyze] {len(documents)} relevant documents")

    # Detect scenarios
    llm = get_llm()
    scenario_result = ScenarioDetector.detect(query, documents, llm)

    has_multiple = scenario_result.get("has_multiple_scenarios", False)
    disambiguation_needed = scenario_result.get("disambiguation_needed", False)

    print(
        f"[search_and_analyze] has_multiple={has_multiple}, disambiguation={disambiguation_needed}"
    )

    return json.dumps(
        {
            "found_answer": True,
            "should_respond": True,
            "quality": analysis["quality"],
            "confidence": float(analysis["confidence"]),
            "documents": documents,
            "count": len(documents),
            "has_multiple_scenarios": has_multiple,
            "disambiguation_needed": disambiguation_needed,
            "scenarios": scenario_result.get("scenarios", []),
            "disambiguation_question": (
                scenario_result.get("suggested_question", "")
                if disambiguation_needed
                else ""
            ),
        }
    )


@tool
def get_scenario_answer(
    search_results_json: str, selected_scenario: str, original_query: str
) -> str:
    """
    Get answer for a specific scenario after user selection.
    """
    print(
        f"\n[get_scenario_answer] Selected: '{selected_scenario}', Original: '{original_query}'"
    )

    if not embedding_service.index_file.exists():
        return json.dumps(
            {"found_answer": False, "message": "No knowledge base available"}
        )

    # Enhanced search
    enhanced_query = f"{original_query} {selected_scenario}"

    try:
        results = embedding_service.search_with_scores(enhanced_query, k=5)
    except Exception as e:
        return json.dumps({"found_answer": False, "error": str(e)})

    if not results:
        return json.dumps(
            {
                "found_answer": False,
                "message": f"No specific information found for '{selected_scenario}'.",
            }
        )

    # Filter for scenario relevance
    scenario_keywords = set(selected_scenario.lower().split())
    filtered_docs = []

    for doc, score in results:
        content_lower = doc.page_content.lower()
        keyword_matches = sum(
            1 for kw in scenario_keywords if kw in content_lower and len(kw) > 3
        )

        if keyword_matches > 0 or float(score) < Config.GOOD_SCORE:
            filtered_docs.append(
                {
                    "content": doc.page_content,
                    "relevance": "high" if keyword_matches >= 2 else "medium",
                    "score": float(score),
                }
            )

    filtered_docs.sort(key=lambda x: x["score"])

    return json.dumps(
        {
            "found_answer": len(filtered_docs) > 0,
            "selected_scenario": selected_scenario,
            "documents": filtered_docs[:3],
            "should_respond": len(filtered_docs) > 0,
        }
    )


@tool
def get_available_topics() -> str:
    """Get list of topics available in the knowledge base."""
    if not embedding_service.index_file.exists():
        return json.dumps({"topics": [], "message": "No knowledge base available"})

    try:
        stats = embedding_service.get_index_stats()
        doc_types = stats.get("documents_by_type", {})
        return json.dumps(
            {
                "topics": list(doc_types.keys()),
                "count": stats.get("unique_files", 0),
                "total_documents": stats.get("total_documents", 0),
            }
        )
    except Exception as e:
        return json.dumps({"topics": [], "error": str(e)})


tools = [search_and_analyze, get_scenario_answer, get_available_topics]
