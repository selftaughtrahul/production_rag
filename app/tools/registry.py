import logging
from typing import Dict, List, Optional
from sqlalchemy.engine import Engine

from app.services.retriever.hybrid import HybridRetriever
from app.tools.base import BaseAgentTool
from app.tools.doc_search import DocumentSearchTool
from app.tools.web_search import WebSearchTool
from app.tools.sql_search import SQLQueryTool, SQLSchemaTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Central registry storing and provisioning agent tools."""

    def __init__(self):
        self._tools: Dict[str, BaseAgentTool] = {}

    def register_tool(self, tool: BaseAgentTool) -> None:
        if not isinstance(tool, BaseAgentTool):
            raise ValueError(f"Tool must inherit from BaseAgentTool, got {type(tool)}")
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: '{tool.name}'")

    def get_tool(self, name: str) -> Optional[BaseAgentTool]:
        return self._tools.get(name)

    def get_all_tools(self) -> List[BaseAgentTool]:
        return list(self._tools.values())

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def list_tool_names(self) -> List[str]:
        return list(self._tools.keys())


def build_default_tool_registry(
    retriever: HybridRetriever,
    db_engine: Engine,
) -> ToolRegistry:
    """Factory creating and populating the standard ToolRegistry."""
    registry = ToolRegistry()

    # 1. Document Search Tool (Hybrid RAG retrieval)
    doc_tool = DocumentSearchTool(retriever=retriever)
    registry.register_tool(doc_tool)

    # 2. Live Web Search Tool (Tavily + DuckDuckGo fallback)
    web_tool = WebSearchTool()
    registry.register_tool(web_tool)

    # 3. SQL Query & Schema Inspection Tools
    sql_query_tool = SQLQueryTool(db_engine=db_engine)
    sql_schema_tool = SQLSchemaTool(db_engine=db_engine)
    registry.register_tool(sql_query_tool)
    registry.register_tool(sql_schema_tool)

    return registry