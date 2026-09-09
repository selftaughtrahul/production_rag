import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    from nemoguardrails import LLMRails, RailsConfig
    NEMO_AVAILABLE = True
except ImportError:
    LLMRails = None
    RailsConfig = None
    NEMO_AVAILABLE = False
    logger.warning("nemoguardrails is not installed. NeMoProvider will operate in mock/fallback mode.")


class NeMoProvider:
    def __init__(self, config_path: Optional[str] = None):
        self.rails = None
        if NEMO_AVAILABLE and config_path:
            try:
                config = RailsConfig.from_path(config_path)
                self.rails = LLMRails(config)
            except Exception as e:
                logger.error(f"Failed to load NeMo configuration: {e}")

    async def check_prompt_injection(self, query: str) -> Dict[str, Any]:
        if not self.rails:
            return {"passed": True, "content": query}

        try:
            response = await self.rails.generate_async(
                messages=[{"role": "user", "content": query}]
            )
            info = response.get("output_data", {}) if isinstance(response, dict) else getattr(response, "info", {})
            content = response.get("content", "") if isinstance(response, dict) else getattr(response, "content", "")

            is_blocked = "I cannot answer this" in content or bool(info.get("triggered_rail"))
            return {"passed": not is_blocked, "content": content}
        except Exception as e:
            logger.error(f"NeMo rail check failed: {e}")
            return {"passed": True, "content": query}