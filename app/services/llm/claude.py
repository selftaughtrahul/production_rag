from __future__ import annotations

import os

from anthropic import Anthropic
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

        self.client = Anthropic(api_key=self.api_key) if self.api_key else None

    def _check_client(self) -> None:
        """Ensure Anthropic client is configured."""

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

        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt or "You are a helpful assistant.",
            messages=messages,
        )

        return response.content[0].text.strip()

    @traceable(run_type="llm", name="Claude RAG Generate")
    def generate(
        self,
        question: str,
        context: str,
        chat_history: list[dict[str, str]] | None = None,
    ) -> str:
        """
        Generate the final RAG answer using retrieved context.
        """

        self._check_client()

        system_prompt = """
You are an expert, direct, and concise RAG assistant.

Your task is to answer the user's question using ONLY the facts provided in the Context.

Guidelines:
- Provide a single, cohesive, well-structured answer.
- Do NOT generate multiple responses, alternative versions, simulated dialogue, or section separators (e.g., '=====' or '-----').
- Do NOT start your response with filler phrases like "Based on the provided context:", "According to the documents:", or "Okay, let me break this down". Start directly with the answer.
- If the answer cannot be determined from the context, respond with: "I don't have enough information in the provided documents to answer this question."
- Do not make up facts or extrapolate beyond what is stated.
"""

        user_prompt = f"""Context:
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

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system_prompt,
            messages=messages,
        )

        return response.content[0].text.strip()

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
            max_tokens=256,
            chat_history=chat_history,
        )

        print(
            "Original question:",
            question,
        )

        print(
            "Rewritten question:",
            rewritten_question,
        )

        return rewritten_question
