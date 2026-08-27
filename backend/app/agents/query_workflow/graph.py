"""
LangGraph StateGraph definition.
"""
from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.base import BaseCheckpointSaver
from app.agents.query_workflow.state import QueryWorkflowState
from app.agents.query_workflow.nodes import QueryNodes
from app.config.logging import get_logger

logger = get_logger(__name__)


def build_query_graph(nodes: QueryNodes, checkpointer: BaseCheckpointSaver | None = None) -> StateGraph:
    """
    Build and compile the Phase 3 LangGraph state machine.
    """
    workflow = StateGraph(QueryWorkflowState)

    # Add Nodes
    workflow.add_node("policy", nodes.evaluate_policy)
    workflow.add_node("cache", nodes.check_cache)
    workflow.add_node("retrieve", nodes.retrieve_evidence)
    workflow.add_node("refine", nodes.refine_query)
    workflow.add_node("generate", nodes.generate_response)
    workflow.add_node("verify", nodes.verify_citations)

    # Entry point
    workflow.set_entry_point("policy")

    # Routing
    def route_after_policy(state: QueryWorkflowState) -> Literal["cache", "__end__"]:
        if state.get("error_message"):
            return END
        return "cache"

    def route_after_cache(state: QueryWorkflowState) -> Literal["retrieve", "__end__"]:
        if state.get("cache_hit"):
            return END
        return "retrieve"

    def route_after_retrieve(state: QueryWorkflowState) -> Literal["generate", "refine"]:
        conf = state.get("confidence_level", "LOW")
        retries = state.get("retries", 0)
        
        if conf == "LOW" and retries < 1:
            return "refine"
        return "generate"

    def route_after_verify(state: QueryWorkflowState) -> Literal["__end__", "retrieve"]:
        status = state.get("verification_status", "UNSUPPORTED")
        retries = state.get("retries", 0)
        
        if status == "PARTIALLY_SUPPORTED" and retries < 2:
            # We can trigger a refinement loop
            return "retrieve"
            
        return END

    # Edges
    workflow.add_conditional_edges("policy", route_after_policy)
    workflow.add_conditional_edges("cache", route_after_cache)
    workflow.add_conditional_edges("retrieve", route_after_retrieve)
    
    # Refinement just loops back to retrieval
    workflow.add_edge("refine", "retrieve")
    
    workflow.add_edge("generate", "verify")
    workflow.add_conditional_edges("verify", route_after_verify)

    return workflow.compile(checkpointer=checkpointer)
