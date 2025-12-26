from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from .state import AgentState
from .nodes import (
    analyze_input,
    handle_greeting,
    handle_closing,
    ask_clarification,
    agent,
    tool_node,
    validate_search_results,
    handle_not_found,
    should_continue,
    route_after_analysis,
    route_after_validation,
)


def create_rag_graph():
    """Create and compile the RAG agent graph."""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("analyze_input", analyze_input)
    workflow.add_node("handle_greeting", handle_greeting)
    workflow.add_node("handle_closing", handle_closing)
    workflow.add_node("ask_clarification", ask_clarification)
    workflow.add_node("agent", agent)
    workflow.add_node("tools", tool_node)
    workflow.add_node("validate_search", validate_search_results)
    workflow.add_node("handle_not_found", handle_not_found)

    # Entry point
    workflow.add_edge(START, "analyze_input")

    # Route after input analysis
    workflow.add_conditional_edges(
        "analyze_input",
        route_after_analysis,
        {
            "handle_greeting": "handle_greeting",
            "handle_closing": "handle_closing",
            "ask_clarification": "ask_clarification",
            "agent": "agent",
        },
    )

    # Terminal nodes
    workflow.add_edge("handle_greeting", END)
    workflow.add_edge("handle_closing", END)
    workflow.add_edge("ask_clarification", END)
    workflow.add_edge("handle_not_found", END)

    # Agent -> Tools or End
    workflow.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", "end": END}
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
