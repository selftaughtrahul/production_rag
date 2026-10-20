import logging
from functools import lru_cache
from pathlib import Path

from app.core.config import Settings
from app.services.llm.claude import ClaudeService
from app.services.llm.laya import LayaService

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_llm() -> ClaudeService:
    """Return shared ClaudeService instance."""
    return ClaudeService()


@lru_cache(maxsize=1)
def get_laya() -> LayaService | None:
    """Load local Laya for supervisor routing when the checkpoint is on disk."""
    settings = Settings.from_environment()
    model_path = Path(settings.local_model_path(settings.laya_model))
    if not model_path.is_dir():
        logger.warning(
            "Laya checkpoint not found at %s; supervisor routing uses Claude",
            model_path,
        )
        return None
    try:
        return LayaService(model=settings.laya_model)
    except Exception as exc:
        logger.warning("Laya failed to load (%s); supervisor routing uses Claude", exc)
        return None
