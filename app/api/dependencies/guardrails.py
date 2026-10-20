import logging
from functools import lru_cache
from typing import Any

from app.core.config import Settings
from app.guardrails.factory import build_output_guardrails
from app.guardrails.output.grounding import TokenOverlapGroundingEvaluator
from app.guardrails.output_service import OutputGuardrailService

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_nemo_provider() -> Any:
    """Load NeMo only when enabled and the config is usable."""
    settings = Settings.from_environment()
    if not settings.enable_nemo:
        return None
    from app.guardrails.provider.nemo import NeMoProvider

    provider = NeMoProvider(config_path=settings.nemo_config_path)
    return provider if provider.is_ready else None


@lru_cache(maxsize=1)
def get_presidio_provider() -> Any:
    """Load Presidio for PII checks. None if spaCy is missing."""
    try:
        from app.guardrails.provider.presidio import PresidioProvider

        return PresidioProvider()
    except Exception as exc:
        logger.warning("Presidio provider not available: %s", exc)
        return None


def build_wired_output_guardrails() -> OutputGuardrailService:
    """Output rails with real providers, not an empty factory call."""
    return build_output_guardrails(
        presidio_provider=get_presidio_provider(),
        grounding_evaluator=TokenOverlapGroundingEvaluator(),
    )
