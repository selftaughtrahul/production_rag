from typing import Any, Dict, Optional, Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Standardized output wrapper returned by all agent tools."""

    success: bool
    data: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_str(self) -> str:
        """Helper to format output for LLM consumption."""
        if not self.success:
            return f"Tool Execution Failed: {self.error}"
        if isinstance(self.data, str):
            return self.data
        return str(self.data)


class BaseAgentTool(BaseTool):
    """Base tool class enforcing typed input schema and structured result format."""

    args_schema: Optional[Type[BaseModel]] = None
    return_direct: bool = False

    def _format_error(self, message: str) -> ToolResult:
        return ToolResult(success=False, data=None, error=message)

    def _format_success(
        self, data: Any, metadata: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        return ToolResult(
            success=True, data=data, metadata=metadata or {}
        )
