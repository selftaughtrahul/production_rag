import os
import logging
import asyncio
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel, Field
from duckduckgo_search import DDGS
from tavily import AsyncTavilyClient

from app.tools.base import BaseAgentTool, ToolResult

logger = logging.getLogger(__name__)


class WebSearchInput(BaseModel):
    query: str = Field(
        ..., description="Search query string for real-time web search (e.g. current news, recent events)."
    )
    max_results: int = Field(
        default=5, ge=1, le=10, description="Maximum number of search results to retrieve."
    )


class WebSearchTool(BaseAgentTool):
    name: str = "web_search"
    description: str = (
        "Searches the public internet for real-time information, recent events, breaking news, "
        "or facts not available in the internal document database."
    )
    args_schema: Type[BaseModel] = WebSearchInput

    def __init__(self, tavily_api_key: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        api_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
        self._tavily_client = AsyncTavilyClient(api_key=api_key) if api_key else None

    def _run(self, query: str, max_results: int = 5) -> str:
        """Synchronous runner calling DuckDuckGo."""
        def ddg_search():
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=max_results))

        try:
            results = ddg_search()
            if not results:
                return self._format_success(
                    f"No results found on the web for query: '{query}'", metadata={"query": query, "count": 0}
                ).to_str()

            formatted = [f"### Web Search Results for '{query}':\n"]
            for idx, item in enumerate(results, 1):
                title = item.get("title", "No Title")
                url = item.get("href", "")
                snippet = item.get("body", "No Description")
                formatted.append(f"{idx}. [{title}]({url})\n   {snippet}\n")
            return self._format_success("\n".join(formatted), metadata={"provider": "duckduckgo"}).to_str()
        except Exception as e:
            logger.error(f"Web search error: {str(e)}")
            return self._format_error(f"Web search failed: {str(e)}").to_str()

    async def _arun(self, query: str, max_results: int = 5) -> str:
        """Asynchronous execution checking Tavily first, falling back to DuckDuckGo."""
        if self._tavily_client:
            try:
                response = await self._tavily_client.search(
                    query=query,
                    search_depth="basic",
                    max_results=max_results,
                )
                results = response.get("results", [])
                if results:
                    formatted = [f"### Web Search Results (Tavily) for '{query}':\n"]
                    for idx, item in enumerate(results, 1):
                        title = item.get("title", "No Title")
                        url = item.get("url", "")
                        content = item.get("content", "No Content Snippet")
                        formatted.append(f"{idx}. [{title}]({url})\n   {content}\n")
                    return self._format_success("\n".join(formatted), metadata={"provider": "tavily"}).to_str()
            except Exception as e:
                logger.warning(f"Tavily search failed, falling back to DuckDuckGo: {str(e)}")

        return await asyncio.to_thread(self._run, query=query, max_results=max_results)
