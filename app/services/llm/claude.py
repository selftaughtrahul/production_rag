from __future__ import annotations

import os
from collections.abc import AsyncIterator

from anthropic import Anthropic, AsyncAnthropic
from dotenv import load_dotenv
from langsmith import traceable


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

        # Synchronous client — used by generate(), generate_text(), rewrite_query()
        self.client = Anthropic(api_key=self.api_key) if self.api_key else None

        # Async client — used by generate_stream() for token-by-token streaming
        self.async_client = AsyncAnthropic(api_key=self.api_key) if self.api_key else None

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
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system_prompt or "You are a helpful assistant.",
                messages=messages,
            )
        except Exception as e:
            if "billing" in str(e).lower():
                return "I'm having trouble accessing my knowledge base right now. Please check my account details or try again later."
            raise

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
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=system_prompt,
                messages=messages,
            )
        except Exception as e:
            if "billing" in str(e).lower():
                return "I'm having trouble accessing my knowledge base right now. Please check my account details or try again later."
            raise

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
        async with self.async_client.messages.stream(
            model=self.model,
            max_tokens=1024,
            system=system_prompt,
            messages=messages,
        ) as stream:
            async for text_token in stream.text_stream:
                yield text_token

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

        response = await self.async_client.messages.create(**kwargs)
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
