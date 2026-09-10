"""First-pass supervisor routing. Return None to fall back to the LLM.

Edit the hint tuples when a new query type should skip the router call.
"""

GREETING_HINTS = ("hi", "hello", "hey", "thanks", "thank you", "good morning", "good evening")
RAG_HINTS = (
    "document",
    "pdf",
    "uploaded",
    "policy",
    "knowledge base",
    "from the file",
    "in the docs",
)
SQL_HINTS = ("sql", "table", "database", "how many rows", "schema", "select ")
WEB_HINTS = (
    "latest news",
    "today's",
    "current price",
    "stock",
    "share price",
    "ticker",
    "live price",
    "breaking",
    "search the web",
    "on the internet",
)


def _has_hint(text: str, hints: tuple[str, ...]) -> bool:
    return any(hint in text for hint in hints)


def next_agent(query: str, already_ran: list[str]) -> str | None:
    """
    Return rag_agent / sql_agent / web_agent / general_agent / FINISH,
    or None when the LLM supervisor should decide.
    """
    if already_ran:
        return "FINISH"

    text = (query or "").lower().strip()
    if not text:
        return "general_agent"

    hits = []
    if _has_hint(text, RAG_HINTS):
        hits.append("rag_agent")
    if _has_hint(text, SQL_HINTS):
        hits.append("sql_agent")
    if _has_hint(text, WEB_HINTS):
        hits.append("web_agent")
    if len(hits) > 1:
        return None
    if len(hits) == 1:
        return hits[0]
    return "general_agent"
