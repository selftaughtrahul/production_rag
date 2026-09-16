import logging
from typing import Dict, List, Optional

from database.order_store import OrderRepository
from app.services.retriever.hybrid import HybridRetriever
from app.tools.base import BaseAgentTool
from app.tools.doc_search import DocumentSearchTool, ListUserDocumentsTool
from app.tools.order_tools import (
    CreateOrderTool,
    GetOrderTool,
    ListOrdersTool,
    OrderAnalyticsTool,
    UpdateOrderTool,
)
from app.tools.utility_tools import CalculatorTool, CurrentDateTimeTool
from app.tools.web_read import WebPageReadTool
from app.tools.web_search import WebSearchTool

logger = logging.getLogger(__name__)

AGENT_TOOL_NAMES: dict[str, tuple[str, ...]] = {
    "rag_agent": ("document_search", "list_user_documents"),
    "sql_agent": (
        "get_order",
        "list_orders",
        "order_analytics",
        "create_order",
        "update_order",
    ),
    "web_agent": ("web_search", "read_web_page"),
    "general_agent": ("calculate", "current_datetime"),
}


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

    def get_tools_for_agent(self, agent_name: str) -> list[BaseAgentTool]:
        """Return only tools explicitly assigned to a specialist."""
        return [
            self._tools[name]
            for name in AGENT_TOOL_NAMES.get(agent_name, ())
            if name in self._tools
        ]

    def get_tool_for_agent(
        self,
        agent_name: str,
        tool_name: str,
    ) -> Optional[BaseAgentTool]:
        """Return a tool only when the specialist is explicitly allowed to use it."""
        if tool_name not in AGENT_TOOL_NAMES.get(agent_name, ()):
            logger.warning(
                "Agent '%s' attempted unassigned tool '%s'",
                agent_name,
                tool_name,
            )
            return None
        return self._tools.get(tool_name)


def build_default_tool_registry(
    retriever: HybridRetriever,
    order_repository: OrderRepository,
) -> ToolRegistry:
    """Factory creating and populating the standard ToolRegistry."""
    registry = ToolRegistry()

    # 1. Document Search Tool (Hybrid RAG retrieval)
    doc_tool = DocumentSearchTool(retriever=retriever)
    registry.register_tool(doc_tool)

    # 2. Live Web Search Tool (Tavily + DuckDuckGo fallback)
    web_tool = WebSearchTool()
    registry.register_tool(web_tool)

    registry.register_tool(ListUserDocumentsTool())
    registry.register_tool(WebPageReadTool())
    registry.register_tool(CalculatorTool())
    registry.register_tool(CurrentDateTimeTool())
    registry.register_tool(GetOrderTool(repository=order_repository))
    registry.register_tool(ListOrdersTool(repository=order_repository))
    registry.register_tool(OrderAnalyticsTool(repository=order_repository))
    registry.register_tool(CreateOrderTool(repository=order_repository))
    registry.register_tool(UpdateOrderTool(repository=order_repository))

    return registry