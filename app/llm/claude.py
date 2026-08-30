from __future__ import annotations

import os

from anthropic import Anthropic
from dotenv import load_dotenv


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

        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt or "You are a helpful assistant.",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        return response.content[0].text.strip()

    def generate(
        self,
        question: str,
        context: str,
    ) -> str:
        """
        Generate the final RAG answer using retrieved context.
        """

        self._check_client()

        system_prompt = """
You are a helpful RAG assistant.

Answer the user's question using ONLY the provided context.

Rules:
- Do not invent information.
- If the answer is not present in the context, say that you don't know.
- Keep the answer clear and concise.
"""

        user_prompt = f"""
Context:
----------------
{context}
----------------

Question:
{question}

Answer:
"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ],
        )

        return response.content[0].text.strip()

    def rewrite_query(
        self,
        question: str,
    ) -> str:
        """
        Rewrite a user question to improve semantic retrieval.
        """

        prompt = f"""
Rewrite the following user question so that it is
better suited for semantic search over a document
knowledge base.

Original question:
{question}

Rules:
- Preserve the original meaning.
- Make the question specific and clear.
- Add useful terminology when appropriate.
- Do not answer the question.
- Return ONLY the rewritten question.
"""

        rewritten_question = self.generate_text(
            prompt=prompt,
            max_tokens=256,
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
