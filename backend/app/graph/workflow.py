
from langgraph.graph import StateGraph, END
from backend.app.graph.state import AgentState
from backend.app.graph.nodes import (
    query_understanding_node,
    code_generation_node,
    execution_node,
    reasoning_node
)

def create_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("query_analysis", query_understanding_node)
    workflow.add_node("code_generation", code_generation_node)
    workflow.add_node("execution", execution_node)
    workflow.add_node("reasoning", reasoning_node)
    
    workflow.set_entry_point("query_analysis")
    
    workflow.add_edge("query_analysis", "code_generation")
    workflow.add_edge("code_generation", "execution")
    workflow.add_edge("execution", "reasoning")
    workflow.add_edge("reasoning", END)
    
    return workflow.compile()

app_graph = create_graph()
