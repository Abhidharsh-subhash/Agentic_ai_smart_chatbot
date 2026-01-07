from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from .state import AgentState
from .nodes import (
    analyze_input,
    think_and_plan,
    handle_greeting,
    handle_closing,
    agent,
    tool_node,
    validate_search_results,
    handle_not_found,
    save_to_memory,
    handle_scenario_selection,
    ask_scenario_clarification,
    handle_awaiting_scenario,
    generate_direct_answer,
    should_continue,
    route_after_analysis,
    route_after_validation,
)


def create_rag_graph():
    """Create the RAG agent graph with consistent scenario handling."""
    workflow = StateGraph(AgentState)

    # Nodes
    workflow.add_node("analyze_input", analyze_input)
    workflow.add_node("think_and_plan", think_and_plan)
    workflow.add_node("agent", agent)
    workflow.add_node("tools", tool_node)
    workflow.add_node("validate_search", validate_search_results)
    workflow.add_node("save_memory", save_to_memory)

    workflow.add_node("handle_greeting", handle_greeting)
    workflow.add_node("handle_closing", handle_closing)
    workflow.add_node("handle_not_found", handle_not_found)
    workflow.add_node("generate_direct_answer", generate_direct_answer)

    workflow.add_node("handle_scenario_selection", handle_scenario_selection)
    workflow.add_node("ask_scenario_clarification", ask_scenario_clarification)
    workflow.add_node("handle_awaiting_scenario", handle_awaiting_scenario)

    # Entry
    workflow.add_edge(START, "analyze_input")

    # After analysis
    workflow.add_conditional_edges(
        "analyze_input",
        route_after_analysis,
        {
            "handle_greeting": "handle_greeting",
            "handle_closing": "handle_closing",
            "handle_scenario_selection": "handle_scenario_selection",
            "handle_awaiting_scenario": "handle_awaiting_scenario",
            "think_and_plan": "think_and_plan",
        },
    )

    # Think -> Agent
    workflow.add_edge("think_and_plan", "agent")

    # Terminal nodes
    workflow.add_edge("handle_greeting", END)
    workflow.add_edge("handle_closing", END)
    workflow.add_edge("handle_not_found", END)
    workflow.add_edge("handle_awaiting_scenario", END)

    # Scenario selection -> save -> end
    workflow.add_edge("handle_scenario_selection", "save_memory")
    workflow.add_edge("save_memory", END)

    # Direct answer -> save -> end
    workflow.add_edge("generate_direct_answer", "save_memory")

    # Clarification -> end (wait for user)
    workflow.add_edge("ask_scenario_clarification", END)

    # Agent -> tools or save
    workflow.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", "save_memory": "save_memory"}
    )

    # Tools -> validate
    workflow.add_edge("tools", "validate_search")

    # Validate -> route
    workflow.add_conditional_edges(
        "validate_search",
        route_after_validation,
        {
            "handle_not_found": "handle_not_found",
            "ask_scenario_clarification": "ask_scenario_clarification",
            "generate_direct_answer": "generate_direct_answer",
        },
    )

    # Compile
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)
