# app/services/rag/tools.py
import json
from langchain_core.tools import tool
from app.services.embeddings import embedding_service
from .config import Config, SearchQuality
from .analyzers import SearchResultAnalyzer, ResponseScenarioAnalyzer


@tool
def search_documents(query: str, num_results: int = 10) -> str:
    """
    Search the document database for relevant information.
    Returns search results with quality analysis.

    IMPORTANT: The response will contain document content that you MUST use exactly as provided.
    Do NOT add any information that is not in the documents.
    """
    # Check if index exists
    if not embedding_service.index_file.exists():
        return json.dumps(
            {
                "found_answer": False,
                "quality": SearchQuality.NOT_FOUND.value,
                "confidence": 0.0,
                "documents": [],
                "message": "No knowledge base available",
                "should_respond": False,
            }
        )

    try:
        # Use embedding service's search with scores
        results = embedding_service.search_with_scores(query, k=num_results)
    except Exception as e:
        return json.dumps(
            {
                "found_answer": False,
                "error": str(e),
                "quality": SearchQuality.NOT_FOUND.value,
                "confidence": 0.0,
                "documents": [],
                "should_respond": False,
            }
        )

    if not results:
        return json.dumps(
            {
                "found_answer": False,
                "quality": SearchQuality.NOT_FOUND.value,
                "confidence": 0.0,
                "documents": [],
                "message": "No results found for this query",
                "should_respond": False,
            }
        )

    # Analyze results quality
    analysis = SearchResultAnalyzer.analyze(results, query)

    # Only include documents if we should respond
    documents = []
    if analysis["should_respond"]:
        for doc, score in results:
            if float(score) < Config.ACCEPTABLE_SCORE:
                documents.append(
                    {
                        "content": doc.page_content,
                        "score": float(score),
                        "relevance": (
                            "high" if float(score) < Config.GOOD_SCORE else "medium"
                        ),
                    }
                )

    # Add instruction for the LLM
    instruction = ""
    if documents:
        instruction = (
            "INSTRUCTION: Use ONLY the content from these documents to answer. "
            "Do NOT add any external information, explanations, or general knowledge. "
            "Report exactly what the documents say."
        )

    return json.dumps(
        {
            "found_answer": analysis["found_relevant_info"],
            "should_respond": analysis["should_respond"],
            "quality": analysis["quality"],
            "confidence": float(analysis["confidence"]),
            "best_score": float(analysis["best_score"]),
            "documents": documents,
            "count": len(documents),
            "reason": analysis["reason"],
            "instruction": instruction,
        }
    )


@tool
def get_available_topics() -> str:
    """Get list of topics available in the knowledge base."""
    if not embedding_service.index_file.exists():
        return json.dumps({"topics": [], "message": "No knowledge base available"})

    try:
        # Get stats from embedding service
        stats = embedding_service.get_index_stats()

        # Extract topics from document types
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


@tool
def analyze_response_for_scenarios(response: str, user_question: str) -> str:
    """
    Analyze an LLM response to detect if it contains multiple scenarios
    that require user clarification.

    Use this tool AFTER getting initial search results and before sending final response.
    If multiple scenarios are detected, the user should be asked to clarify their situation.

    Args:
        response: The generated response to analyze
        user_question: The user's original question

    Returns:
        JSON with analysis results including:
        - has_multiple_scenarios: bool
        - scenarios: list of detected scenarios
        - clarification_question: suggested question to ask user
    """
    try:
        analyzer = ResponseScenarioAnalyzer()
        analysis = analyzer.analyze_response(response, user_question)
        return json.dumps(analysis)
    except Exception as e:
        return json.dumps(
            {
                "has_multiple_scenarios": False,
                "error": str(e),
                "scenarios": [],
                "clarification_question": "",
            }
        )


@tool
def match_user_clarification_to_scenario(
    scenarios_json: str, user_clarification: str, original_question: str
) -> str:
    """
    Match user's clarification response to the most relevant scenario.

    Use this tool when user provides clarification after being asked about their specific situation.

    Args:
        scenarios_json: JSON string of available scenarios
        user_clarification: User's response describing their situation
        original_question: The original question that started this flow

    Returns:
        JSON with matched scenario and confidence score
    """
    try:
        scenarios = (
            json.loads(scenarios_json)
            if isinstance(scenarios_json, str)
            else scenarios_json
        )
        analyzer = ResponseScenarioAnalyzer()
        result = analyzer.match_user_to_scenario(
            scenarios, user_clarification, original_question
        )
        return json.dumps(result)
    except Exception as e:
        return json.dumps(
            {
                "matched_scenario_id": None,
                "confidence": 0.0,
                "error": str(e),
                "needs_more_info": True,
            }
        )


# Export tools list
tools = [search_documents, get_available_topics]

# Separate tools for response analysis (used internally, not by main agent)
analysis_tools = [analyze_response_for_scenarios, match_user_clarification_to_scenario]
