"""
AutoML Debate LangGraph orchestration.

Flow (edges):
    START → prepare → eda → memory_retrieve → [model_rf, model_xgb, model_lr] (parallel)
    → evaluate → debate → judge → END

``memory_retrieve`` queries Chroma for similar past dataset patterns (characteristics + best model + metrics)
and fills ``memory_context`` for model proposals. Parallel model agents use ``Send``; results merge via
``Annotated`` reducers. ``evaluate`` waits for all three model nodes (barrier edge).
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from app.agents.nodes import (
    node_debate_agent,
    node_eda_agent,
    node_evaluation_agent,
    node_judge_agent,
    node_memory_retrieve,
    node_model_agent_lr,
    node_model_agent_rf,
    node_model_agent_xgb,
    node_prepare_dataset,
)
from app.state import DebateGraphState

MODEL_WORKER_NODES = ("model_rf", "model_xgb", "model_lr")

# Ordered LangGraph nodes for UI (prepare → … → judge).
PIPELINE_NODE_ORDER = (
    "prepare",
    "eda",
    "memory_retrieve",
    *MODEL_WORKER_NODES,
    "evaluate",
    "debate",
    "judge",
)


def route_after_prepare(state: DebateGraphState) -> str:
    """Skip pipeline when dataset preparation failed."""
    if state.get("error"):
        return "end"
    return "eda"


def route_parallel_model_agents(state: DebateGraphState) -> Any:
    """
    Fan-out: dispatch all model agents with the same graph state (post-EDA).
    On error, short-circuit to END so downstream nodes see ``error`` and no-op.
    """
    if state.get("error"):
        return END
    return [
        Send("model_rf", state),
        Send("model_xgb", state),
        Send("model_lr", state),
    ]


def build_debate_graph() -> Any:
    """
    Build the compiled StateGraph: EDA → parallel model training/eval → evaluation → debate → judge.

    State keys that merge across parallel workers: ``model_runs``, ``model_proposals`` (dict merge),
    ``metrics`` (dict merge after evaluate), ``reasoning_logs`` (list concat).
    """
    g = StateGraph(DebateGraphState)

    g.add_node("prepare", node_prepare_dataset)
    g.add_node("eda", node_eda_agent)
    g.add_node("memory_retrieve", node_memory_retrieve)
    g.add_node("model_rf", node_model_agent_rf)
    g.add_node("model_xgb", node_model_agent_xgb)
    g.add_node("model_lr", node_model_agent_lr)
    g.add_node("evaluate", node_evaluation_agent)
    g.add_node("debate", node_debate_agent)
    g.add_node("judge", node_judge_agent)

    g.add_edge(START, "prepare")
    g.add_conditional_edges("prepare", route_after_prepare, {"eda": "eda", "end": END})

    g.add_edge("eda", "memory_retrieve")
    g.add_conditional_edges("memory_retrieve", route_parallel_model_agents)

    # Join: wait for all three model workers before evaluation (barrier).
    g.add_edge(list(MODEL_WORKER_NODES), "evaluate")

    g.add_edge("evaluate", "debate")
    g.add_edge("debate", "judge")
    g.add_edge("judge", END)

    return g.compile()
