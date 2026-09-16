import re
import logging
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel, Field
from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine
from app.tools.base import BaseAgentTool, ToolResult

logger = logging.getLogger(__name__)

DENIED_TABLES = {
    "users",
    "user_memories",
    "checkpoints",
    "checkpoint_writes",
    "checkpoint_blobs",
    "writes",
    "sqlite_master",
    "sqlite_sequence",
    "checkpoint_migrations",
    "pg_catalog",
    "information_schema",
}
_TABLE_REF = re.compile(
    r'\b(?:FROM|JOIN|INTO|UPDATE|TABLE)\s+["\'`]?(\w+)',
    re.IGNORECASE,
)
_BLOCKED_OPS = [
    r"\bINSERT\b", r"\bUPDATE\b", r"\bDELETE\b", r"\bDROP\b",
    r"\bALTER\b", r"\bTRUNCATE\b", r"\bCREATE\b", r"\bGRANT\b", r"\bREVOKE\b",
    r"\bATTACH\b", r"\bPRAGMA\b", r"\bVACUUM\b", r"\bREINDEX\b",
]


def _strip_sql_comments(sql_statement: str) -> str:
    return re.sub(r"--.*?\n|/\*.*?\*/", "", sql_statement, flags=re.DOTALL).strip()


def _referenced_tables(sql_statement: str) -> set[str]:
    return {match.group(1).lower() for match in _TABLE_REF.finditer(sql_statement)}


class SQLQueryInput(BaseModel):
    query: str = Field(
        ..., description="Read-only SQL SELECT query to execute against the database."
    )


class SQLSchemaInspectorInput(BaseModel):
    include_tables: Optional[List[str]] = Field(
        default=None, description="Optional list of specific table names to inspect."
    )


class SQLQueryTool(BaseAgentTool):
    name: str = "sql_db_query"
    description: str = (
        "Executes a read-only SQL query against the database and returns structured tabular results. "
        "ALWAYS use the 'sql_db_schema' tool first to inspect available table structures before generating queries."
    )
    args_schema: Type[BaseModel] = SQLQueryInput
    db_engine: Engine

    class Config:
        arbitrary_types_allowed = True

    def _is_read_only(self, sql_statement: str) -> bool:
        """Strict validation to block destructive/write queries."""
        cleaned = _strip_sql_comments(sql_statement).rstrip(";").strip()
        if ";" in cleaned:
            return False
        for kw in _BLOCKED_OPS:
            if re.search(kw, cleaned, re.IGNORECASE):
                return False
        return cleaned.upper().startswith("SELECT") or cleaned.upper().startswith("WITH")

    def _run(self, query: str) -> str:
        if not self._is_read_only(query):
            return self._format_error("Security Error: Only read-only SELECT queries are allowed.").to_str()
        if _referenced_tables(query) & DENIED_TABLES:
            return self._format_error("Security Error: Query targets a restricted table.").to_str()

        try:
            with self.db_engine.connect() as conn:
                result = conn.execute(text(query))
                keys = list(result.keys())
                rows = result.fetchmany(100)  # Row limit cap for safety

                if not rows:
                    return self._format_success("Query returned 0 rows.", metadata={"rows_count": 0}).to_str()

                formatted_results = [dict(zip(keys, row)) for row in rows]
                return self._format_success(
                    formatted_results, metadata={"rows_returned": len(rows)}
                ).to_str()

        except Exception as e:
            logger.error(f"SQL execution error: {str(e)}", exc_info=True)
            return self._format_error(f"Database execution error: {str(e)}").to_str()

    async def _arun(self, query: str) -> str:
        return self._run(query=query)


class SQLSchemaTool(BaseAgentTool):
    name: str = "sql_db_schema"
    description: str = (
        "Inspects database schema details including tables, columns, primary keys, and data types. "
        "Use this tool to understand the database structure before executing queries."
    )
    args_schema: Type[BaseModel] = SQLSchemaInspectorInput
    db_engine: Engine

    class Config:
        arbitrary_types_allowed = True

    def _run(self, include_tables: Optional[List[str]] = None) -> str:
        try:
            inspector = inspect(self.db_engine)
            all_tables = [
                table for table in inspector.get_table_names()
                if table.lower() not in DENIED_TABLES
            ]
            requested = include_tables if include_tables else all_tables
            target_tables = [table for table in requested if table.lower() not in DENIED_TABLES]

            schema_info = []
            for table in target_tables:
                if table not in all_tables:
                    continue
                columns = inspector.get_columns(table)
                pk = inspector.get_pk_constraint(table)
                col_descriptions = [
                    f"  - {col['name']} ({col['type']})" + (" [PRIMARY KEY]" if col['name'] in pk.get('constrained_columns', []) else "")
                    for col in columns
                ]
                schema_info.append(f"Table: {table}\n" + "\n".join(col_descriptions))

            output = "\n\n".join(schema_info) if schema_info else "No table metadata found."
            return self._format_success(output, metadata={"tables": target_tables}).to_str()

        except Exception as e:
            logger.error(f"SQL Schema inspection error: {str(e)}")
            return self._format_error(f"Schema inspection failed: {str(e)}").to_str()

    async def _arun(self, include_tables: Optional[List[str]] = None) -> str:
        return self._run(include_tables=include_tables)
