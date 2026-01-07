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
    """Create and compile the RAG agent graph with dynamic scenario handling."""
    workflow = StateGraph(AgentState)

    # Core nodes
    workflow.add_node("analyze_input", analyze_input)
    workflow.add_node("think_and_plan", think_and_plan)
    workflow.add_node("agent", agent)
    workflow.add_node("tools", tool_node)
    workflow.add_node("validate_search", validate_search_results)
    workflow.add_node("save_memory", save_to_memory)

    # Response nodes
    workflow.add_node("handle_greeting", handle_greeting)
    workflow.add_node("handle_closing", handle_closing)
    workflow.add_node("handle_not_found", handle_not_found)
    workflow.add_node("generate_direct_answer", generate_direct_answer)

    # Scenario nodes
    workflow.add_node("handle_scenario_selection", handle_scenario_selection)
    workflow.add_node("ask_scenario_clarification", ask_scenario_clarification)
    workflow.add_node("handle_awaiting_scenario", handle_awaiting_scenario)

    # Entry point
    workflow.add_edge(START, "analyze_input")

    # Route after input analysis
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

    # After thinking, go to agent
    workflow.add_edge("think_and_plan", "agent")

    # Terminal nodes
    workflow.add_edge("handle_greeting", END)
    workflow.add_edge("handle_closing", END)
    workflow.add_edge("handle_not_found", END)
    workflow.add_edge("handle_awaiting_scenario", END)

    # After scenario selection, save and end
    workflow.add_edge("handle_scenario_selection", "save_memory")
    workflow.add_edge("save_memory", END)

    # Direct answer after generation
    workflow.add_edge("generate_direct_answer", "save_memory")

    # Scenario clarification ends (waits for user input)
    workflow.add_edge("ask_scenario_clarification", END)

    # Agent -> Tools or Save Memory
    workflow.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", "save_memory": "save_memory"}
    )

    # Tools -> Validate
    workflow.add_edge("tools", "validate_search")

    # Validate -> Route to appropriate handler
    workflow.add_conditional_edges(
        "validate_search",
        route_after_validation,
        {
            "handle_not_found": "handle_not_found",
            "ask_scenario_clarification": "ask_scenario_clarification",
            "generate_direct_answer": "generate_direct_answer",
        },
    )

    # Compile with memory
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)
