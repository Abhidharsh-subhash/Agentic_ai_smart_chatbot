# app/services/rag/graph.py
"""
Graph definition - EXACT copy from standalone agentic_ai_logic.py
"""
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
    validate_and_route,
    handle_not_found,
    present_scenarios,
    should_continue,
    route_after_analysis,
    route_after_validation,
)


def create_agent():
    """Create and compile the agent graph - matches standalone exactly."""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("analyze_input", analyze_input)
    workflow.add_node("handle_greeting", handle_greeting)
    workflow.add_node("handle_closing", handle_closing)
    workflow.add_node("ask_clarification", ask_clarification)
    workflow.add_node("agent", agent)
    workflow.add_node("tools", tool_node)
    workflow.add_node("validate_and_route", validate_and_route)
    workflow.add_node("handle_not_found", handle_not_found)
    workflow.add_node("present_scenarios", present_scenarios)

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
    workflow.add_edge("present_scenarios", END)

    # Agent -> tools or end
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END},
    )

    # Tools -> Validate
    workflow.add_edge("tools", "validate_and_route")

    # Validate -> route appropriately
    workflow.add_conditional_edges(
        "validate_and_route",
        route_after_validation,
        {
            "handle_not_found": "handle_not_found",
            "present_scenarios": "present_scenarios",
            "agent": "agent",
        },
    )

    # CRITICAL: MemorySaver maintains conversation across invocations
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)
