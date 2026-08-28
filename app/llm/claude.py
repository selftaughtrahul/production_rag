from __future__ import annotations

import os

from anthropic import Anthropic
from dotenv import load_dotenv


class ClaudeService:
    """Service responsible for generating answers using Claude."""

    def __init__(self, model: str | None = None) -> None:
        # Supports direct use of this service as well as construction through the
        # application dependency module.
        load_dotenv(override=False)
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        self.client = Anthropic(api_key=self.api_key) if self.api_key else None

    def generate(self, question: str, context: str) -> str:
        if self.client is None:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not configured. Add it to the project .env "
                "file or set it in the process environment, then restart the API."
            )

        system_prompt = """
You are a helpful RAG assistant.

Answer the user's question using ONLY the provided context.

Rules:
- Do not invent information.
- If the answer is not present in the context, say that you don't have enough information.
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
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text
