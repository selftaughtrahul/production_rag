"""


"""
import json
import re
from app.memory.models import MemoryDecision, UserMemory
from app.prompts.memory_prompts import MEMORY_PROMPT
from app.services.llm.claude import ClaudeService


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


class MemoryExtractor:
    def __init__(self, llm: ClaudeService) -> None:
        self.llm = llm

    def decide(self, user_id: str, conversation: str, existing_memories: list[UserMemory]) -> MemoryDecision:
        existing_text = (
            "\n".join(
                f"- id={item.id} type={item.memory_type} importance={item.importance}: {item.memory}"
                for item in existing_memories
            )
            or "(none)"
        )

        prompt = f"""USER ID:
{user_id}

EXISTING MEMORIES:
{existing_text}

NEW CONVERSATION:
{conversation}

Return a JSON object with these keys:
- action: ADD, UPDATE, or IGNORE
- memory: concise fact about the user, or null
- memory_id: existing memory id when action is UPDATE, otherwise null
- memory_type: personal, professional, technical, preference, project, goal, or general
- importance: number between 0 and 1
- reason: short explanation

Return ONLY JSON.
"""

        raw = self.llm.generate_text(prompt=prompt,system_prompt=MEMORY_PROMPT,max_tokens=512)

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            match = _JSON_BLOCK.search(raw)
            if not match:
                return MemoryDecision(action="IGNORE", reason="unparseable extractor output")
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                return MemoryDecision(action="IGNORE", reason="invalid extractor json")

        try:
            return MemoryDecision.model_validate(payload)
        except Exception:
            return MemoryDecision(action="IGNORE", reason="invalid extractor payload")
