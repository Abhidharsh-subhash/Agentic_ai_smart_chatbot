from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from .state import AgentState
from .nodes import (
    analyze_input,
    think_and_plan,
    handle_greeting,
    handle_closing,
    handle_document_listing,
    ask_clarification,
    agent,
    tool_node,
    validate_search_results,
    handle_not_found,
    save_to_memory,
    should_continue,
    route_after_analysis,
    route_after_validation,
)


def create_rag_graph():
    """Create and compile the RAG agent graph with CoT and Memory."""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("analyze_input", analyze_input)
    workflow.add_node("think_and_plan", think_and_plan)  # NEW: CoT node
    workflow.add_node("handle_greeting", handle_greeting)
    workflow.add_node("handle_closing", handle_closing)
    workflow.add_node("handle_document_listing", handle_document_listing)
    workflow.add_node("ask_clarification", ask_clarification)
    workflow.add_node("agent", agent)
    workflow.add_node("tools", tool_node)
    workflow.add_node("validate_search", validate_search_results)
    workflow.add_node("handle_not_found", handle_not_found)
    workflow.add_node("save_memory", save_to_memory)  # NEW: Memory node

    # Entry point
    workflow.add_edge(START, "analyze_input")

    # Route after input analysis
    workflow.add_conditional_edges(
        "analyze_input",
        route_after_analysis,
        {
            "handle_greeting": "handle_greeting",
            "handle_closing": "handle_closing",
            "handle_document_listing": "handle_document_listing",
            "ask_clarification": "ask_clarification",
            "think_and_plan": "think_and_plan",  # Go to CoT
        },
    )

    # After thinking, go to agent
    workflow.add_edge("think_and_plan", "agent")

    # Terminal nodes go to END
    workflow.add_edge("handle_greeting", END)
    workflow.add_edge("handle_closing", END)
    workflow.add_edge("handle_document_listing", END)
    workflow.add_edge("ask_clarification", END)
    workflow.add_edge("handle_not_found", END)
    workflow.add_edge("save_memory", END)  # Memory save then END

    # Agent -> Tools or Save Memory
    workflow.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", "save_memory": "save_memory"}
    )

    # Tools -> Validate Search Results
    workflow.add_edge("tools", "validate_search")

    # Validate -> Handle Not Found or Continue to Agent
    workflow.add_conditional_edges(
        "validate_search",
        route_after_validation,
        {"handle_not_found": "handle_not_found", "agent": "agent"},
    )

    # Compile with memory
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)
