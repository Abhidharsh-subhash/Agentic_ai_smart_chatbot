import json
from langchain_core.tools import tool
from app.services.embeddings import embedding_service
from .config import Config, SearchQuality
from .analyzers import SearchResultAnalyzer


@tool
def search_documents(query: str, num_results: int = 10) -> str:
    """
    Search the document database for relevant information.
    Returns search results with quality analysis.

    Note: Scenario detection is handled separately after this returns.
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
            }
        )

    if not results:
        return json.dumps(
            {
                "found_answer": False,
                "quality": SearchQuality.NOT_FOUND.value,
                "confidence": 0.0,
                "documents": [],
                "message": "No results found",
                "should_respond": False,
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

    response = {
        "found_answer": analysis["found_relevant_info"],
        "should_respond": analysis["should_respond"],
        "quality": analysis["quality"],
        "confidence": float(analysis["confidence"]),
        "best_score": float(analysis["best_score"]),
        "documents": documents,
        "count": len(documents),
        "reason": analysis["reason"],
        # Scenario detection will be done in validate_search node
        "needs_scenario_analysis": len(documents) > 0,
    }

    if documents:
        response["instruction"] = (
            "Use ONLY the content from these documents. "
            "Scenario analysis will determine if clarification is needed."
        )

    return json.dumps(response)


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
tools = [search_documents, get_available_topics]
