"""Specialist nodes used by the single master supervisor graph."""

from app.agent.multi_agent.specialists.general import GeneralLLMNode
from app.agent.multi_agent.specialists.rag import RAGSubGraphNode
from app.agent.multi_agent.specialists.sql import SQLSubGraphNode
from app.agent.multi_agent.specialists.web import WebAgentNode

__all__ = [
    "GeneralLLMNode",
    "RAGSubGraphNode",
    "SQLSubGraphNode",
    "WebAgentNode",
]
