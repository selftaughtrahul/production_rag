from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator

from anthropic import (
    Anthropic,
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncAnthropic,
    AuthenticationError,
    PermissionDeniedError,
    RateLimitError,
)
from dotenv import load_dotenv
from langsmith import traceable
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from app.core.exceptions import LLMUnavailableError

logger = logging.getLogger(__name__)

# Provider wording that means "the account cannot spend right now".
_QUOTA_MARKERS = ("usage limit", "credit balance", "billing", "quota")
_LLM_TIMEOUT_SECONDS = 60.0
_LLM_RETRY_ATTEMPTS = 3
_RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504, 529})


def _provider_message(exc: APIStatusError) -> str:
    """Pull the human-readable message out of an Anthropic error body."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message.strip():
                return message.strip()
    return str(exc)


def _as_unavailable(exc: Exception) -> LLMUnavailableError | None:
    """Map provider quota/auth/connectivity failures to a user-safe error."""
    if isinstance(exc, (AuthenticationError, PermissionDeniedError)):
        return LLMUnavailableError(
            "The AI provider rejected the configured API key. Check ANTHROPIC_API_KEY."
        )
    if isinstance(exc, RateLimitError):
        return LLMUnavailableError(_provider_message(exc))
    if isinstance(exc, APIConnectionError):
        return LLMUnavailableError(
            "The AI provider is unreachable right now. Please try again shortly."
        )
    if isinstance(exc, APIStatusError):
        message = _provider_message(exc)
        if any(marker in message.lower() for marker in _QUOTA_MARKERS):
            return LLMUnavailableError(message)
    return None


def _is_retryable(exc: BaseException) -> bool:
    """Retry transient provider blips; do not retry auth or billing failures."""
    if isinstance(exc, (AuthenticationError, PermissionDeniedError)):
        return False
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)):
        return True
    if isinstance(exc, APIStatusError):
        message = _provider_message(exc).lower()
        if any(marker in message for marker in _QUOTA_MARKERS):
            return False
        return exc.status_code in _RETRYABLE_STATUS
    return False


def _log_usage(response: object) -> None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    logger.info(
        "Claude usage model=%s input_tokens=%s output_tokens=%s",
        getattr(response, "model", ""),
        getattr(usage, "input_tokens", None),
        getattr(usage, "output_tokens", None),
    )


def _reraise_unavailable(exc: Exception) -> None:
    unavailable = _as_unavailable(exc)
    if unavailable is not None:
        raise unavailable from exc
    raise exc


@retry(
    reraise=True,
    stop=stop_after_attempt(_LLM_RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception(_is_retryable),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def _messages_create_with_retry(client: Anthropic, **kwargs):
    return client.messages.create(timeout=_LLM_TIMEOUT_SECONDS, **kwargs)


@retry(
    reraise=True,
    stop=stop_after_attempt(_LLM_RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception(_is_retryable),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
async def _amessages_create_with_retry(client: AsyncAnthropic, **kwargs):
    return await client.messages.create(timeout=_LLM_TIMEOUT_SECONDS, **kwargs)


class ClaudeService:
    """Service responsible for interacting with Claude."""

    def __init__(self, model: str | None = None) -> None:
        # Load environment variables.
        load_dotenv(override=False)

        self.model = model or os.getenv(
            "ANTHROPIC_MODEL",
            "claude-sonnet-4-5",
        )

        self.api_key = os.getenv(
            "ANTHROPIC_API_KEY",
            "",
        ).strip()

        timeout = float(os.getenv("ANTHROPIC_TIMEOUT_SECONDS", str(_LLM_TIMEOUT_SECONDS)))

        # Synchronous client — used by generate(), generate_text(), rewrite_query()
        self.client = (
            Anthropic(api_key=self.api_key, timeout=timeout) if self.api_key else None
        )

        # Async client — used by generate_stream() for token-by-token streaming
        self.async_client = (
            AsyncAnthropic(api_key=self.api_key, timeout=timeout) if self.api_key else None
        )

    def _check_client(self) -> None:
        """Raise a clear error when ANTHROPIC_API_KEY is missing."""
        if self.client is None:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not configured. "
                "Add it to the project .env file or set it "
                "in the process environment, then restart the API."
            )

    def generate_text(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
        chat_history: list[dict[str, str]] | None = None,
    ) -> str:
        """
        Generic Claude text generation.

        Used by RAG operations such as:
        - Query rewriting
        - Document grading
        - Query expansion
        - Other LLM tasks
        """

        self._check_client()
        
        messages = []
        if chat_history:
            messages.extend(chat_history)
        messages.append({
            "role": "user",
            "content": prompt,
        })

        try:
            response = _messages_create_with_retry(
                self.client,
                model=self.model,
                max_tokens=max_tokens,
                system=system_prompt or "You are a helpful assistant.",
                messages=messages,
            )
        except Exception as e:
            _reraise_unavailable(e)

        _log_usage(response)
        return response.content[0].text.strip()

    @traceable(run_type="llm", name="Claude RAG Generate")
    def generate(
        self,
        question: str,
        context: str,
        chat_history: list[dict[str, str]] | None = None,
        long_term_memories: list[str] | None = None,
    ) -> str:
        """
        Generate the final RAG answer using retrieved context.
        """

        self._check_client()

        system_prompt = """
You are an expert, direct, and concise AI assistant integrated with a RAG pipeline.

Guidelines:
- Your primary task is to answer the user's question using the provided Context documents.
- If the user asks a conversational question or asks something based on the chat history (like "what is my name?"), answer naturally using the chat history and long-term memories.
- Use long-term memories as persistent facts about the user. Prefer them over chat history when they conflict with older turns.
- Provide a single, cohesive, well-structured answer.
- Do NOT generate multiple responses, alternative versions, simulated dialogue, or section separators.
- Do NOT start your response with filler phrases like "Based on the provided context:", "According to the documents:".
- If the user asks a factual question that requires documents, but the context is empty and it's not in the chat history or memories, respond with: "I don't have enough information in the provided documents to answer this question."
- Do not make up facts or extrapolate beyond what is stated.
"""

        memories_text = "\n".join(f"- {item}" for item in (long_term_memories or [])) or "(none)"

        user_prompt = f"""Long-term memories about the user:
{memories_text}

Context:
{context}

Question:
{question}"""

        messages = []
        if chat_history:
            messages.extend(chat_history)
        messages.append({
            "role": "user",
            "content": user_prompt,
        })

        try:
            response = _messages_create_with_retry(
                self.client,
                model=self.model,
                max_tokens=1024,
                system=system_prompt,
                messages=messages,
            )
        except Exception as e:
            _reraise_unavailable(e)

        _log_usage(response)
        return response.content[0].text.strip()

    # ──────────────────────────────────────────────────────────────────
    # Streaming variant — yields one text token at a time
    # ──────────────────────────────────────────────────────────────────

    async def generate_stream(
        self,
        question: str,
        context: str,
        chat_history: list[dict[str, str]] | None = None,
        long_term_memories: list[str] | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream the RAG answer token by token.

        Usage (in an async context)::

            async for token in claude.generate_stream(question, context):
                print(token, end="", flush=True)

        Yields:
            str — each text token as it arrives from the Anthropic API.
        """
        self._check_client()

        # ── Build system prompt ────────────────────────────────────────
        system_prompt = """\
You are an expert, direct, and concise AI assistant integrated with a RAG pipeline.

Guidelines:
- Answer the user's question using the provided Context documents.
- If the user asks a conversational question (e.g. "what is my name?"), \
answer naturally using the chat history and long-term memories.
- Use long-term memories as persistent facts about the user.
- Provide a single, cohesive, well-structured answer.
- Do NOT generate multiple responses or alternative versions.
- Do NOT start with filler like "Based on the provided context:".
- If context is empty and the answer is not in chat history or memories, \
respond with: "I don't have enough information in the provided documents to answer this question."
- Do not make up facts or extrapolate beyond what is stated.
"""

        # ── Build user turn ────────────────────────────────────────────
        memories_text = (
            "\n".join(f"- {item}" for item in (long_term_memories or []))
            or "(none)"
        )

        user_turn = f"""Long-term memories about the user:
{memories_text}

Context:
{context}

Question:
{question}"""

        # ── Assemble message list (history + current turn) ─────────────
        messages: list[dict[str, str]] = []
        if chat_history:
            messages.extend(chat_history)
        messages.append({"role": "user", "content": user_turn})

        # ── Stream from Anthropic ──────────────────────────────────────
        try:
            async with self.async_client.messages.stream(
                model=self.model,
                max_tokens=1024,
                system=system_prompt,
                messages=messages,
                timeout=_LLM_TIMEOUT_SECONDS,
            ) as stream:
                async for text_token in stream.text_stream:
                    yield text_token
        except Exception as exc:
            _reraise_unavailable(exc)

    @traceable(run_type="llm", name="Claude Rewrite Query")
    def rewrite_query(
        self,
        question: str,
        chat_history: list[dict[str, str]] | None = None,
    ) -> str:
        """
        Rewrite a user question to improve semantic retrieval.
        """

        prompt = f"""
Given a user query, output a concise and direct search query optimized for semantic and keyword retrieval over technical documentation.

Original question:
{question}

Guidelines:
- Keep the rewritten query concise (under 8-12 words).
- Focus on key concepts and technical synonyms.
- Do NOT turn it into a long introductory sentence.
- Return ONLY the search query string.
"""

        rewritten_question = self.generate_text(
            prompt=prompt,
            max_tokens=32,
            chat_history=chat_history,
        )

        return rewritten_question

    @traceable(name="ChatAnthropic", run_type="llm")
    async def agenerate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
    ) -> str:
        """Asynchronously generate a text response from Claude."""
        self._check_client()
        messages = [{"role": "user", "content": prompt}]
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        try:
            response = await self.async_client.messages.create(**kwargs)
        except Exception as exc:
            unavailable = _as_unavailable(exc)
            if unavailable is not None:
                raise unavailable from exc
            raise
        return response.content[0].text.strip()

    async def agenerate_structured(
        self,
        prompt: str,
        system_prompt: str,
        response_model: type,
        max_tokens: int = 1024,
    ):
        """Asynchronously generate structured output conforming to a Pydantic model."""
        import json
        import re

        schema_json = json.dumps(response_model.model_json_schema(), indent=2)
        full_system = f"{system_prompt}\n\nIMPORTANT: Output strictly valid JSON conforming to this JSON Schema:\n{schema_json}\nReturn ONLY the JSON object, with no markdown code blocks or surrounding commentary."

        raw_response = await self.agenerate(
            prompt=prompt,
            system_prompt=full_system,
            max_tokens=max_tokens,
        )

        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_response.strip(), flags=re.DOTALL)
        return response_model.model_validate_json(cleaned)
