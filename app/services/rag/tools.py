import json
from langchain_core.tools import tool
from app.services.embeddings import embedding_service
from .config import Config, SearchQuality
from .analyzers import SearchResultAnalyzer, ScenarioDetector


@tool
def search_documents(query: str, num_results: int = 10) -> str:
    """
    Search the document database for relevant information.
    Returns search results with quality analysis and scenario detection.
    """
    if not embedding_service.index_file.exists():
        return json.dumps(
            {
                "found_answer": False,
                "quality": SearchQuality.NOT_FOUND.value,
                "confidence": 0.0,
                "documents": [],
                "message": "No knowledge base available",
                "should_respond": False,
                "has_multiple_scenarios": False,
                "scenarios": [],
            }
        )

    try:
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
                "message": "No results found for this query",
                "should_respond": False,
                "has_multiple_scenarios": False,
            }
        )

    # Analyze results quality
    analysis = SearchResultAnalyzer.analyze(results, query)

    # Build documents list
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
                        "metadata": doc.metadata,
                    }
                )

    # NEW: Detect scenarios in the documents
    scenario_detection = ScenarioDetector.detect_scenarios(documents, query)

    # Build response
    response = {
        "found_answer": analysis["found_relevant_info"],
        "should_respond": analysis["should_respond"],
        "quality": analysis["quality"],
        "confidence": float(analysis["confidence"]),
        "best_score": float(analysis["best_score"]),
        "documents": documents,
        "count": len(documents),
        "reason": analysis["reason"],
        # NEW: Scenario information
        "has_multiple_scenarios": scenario_detection["has_multiple_scenarios"],
        "scenarios": scenario_detection["scenarios"],
        "clarification_question": scenario_detection.get("clarification_question"),
        "scenario_confidence": scenario_detection.get("confidence", 0.0),
    }

    # Add instruction based on scenarios
    if scenario_detection["has_multiple_scenarios"]:
        response["instruction"] = (
            "MULTIPLE SCENARIOS DETECTED: Ask the user to clarify which situation applies to them "
            "before providing a specific answer. Use the clarification_question provided."
        )
    elif documents:
        response["instruction"] = (
            "Use ONLY the content from these documents to answer. "
            "Do NOT add any external information."
        )

    return json.dumps(response)


@tool
def get_scenario_specific_answer(scenario_id: str, documents: str) -> str:
    """
    Extract answer for a specific scenario from the documents.

    Args:
        scenario_id: The ID of the scenario selected by the user
        documents: JSON string of the original documents
    """
    try:
        docs = json.loads(documents) if isinstance(documents, str) else documents
    except:
        return json.dumps({"success": False, "error": "Could not parse documents"})

    # This would typically use an LLM to extract scenario-specific content
    # For now, return the documents for the main LLM to process
    return json.dumps(
        {
            "success": True,
            "scenario_id": scenario_id,
            "documents": docs,
            "instruction": f"Extract and provide ONLY the information relevant to scenario {scenario_id}.",
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


# Export tools list
tools = [search_documents, get_scenario_specific_answer, get_available_topics]
