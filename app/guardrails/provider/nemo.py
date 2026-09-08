from app.guardrails.base import BaseInputGuardrail,GuardrailResult
from nemoguardrails import LLMRails, RailsConfig

class NeMoProvider:
    def __init__(self, config_path: str):
        config = RailsConfig.from_path(config_path)
        self.rails = LLMRails(config)

    async def check_prompt_injection(self, query: str) -> dict:
        response = await self.rails.generate_async(
            messages=[{"role": "user", "content": query}]
        )
        info = response.get("output_data", {}) if isinstance(response, dict) else getattr(response, "info", {})
        content = response.get("content", "") if isinstance(response, dict) else getattr(response, "content", "")
        
        is_blocked = "I cannot answer this" in content or bool(info.get("triggered_rail"))
        return {"passed": not is_blocked, "content": content}