from functools import lru_cache

from app.core.config import Settings
from app.services.ingestion.chunker import ChunkerService, LangChainRecursiveStrategy
from app.services.ingestion.data_cleaning import DataCleaningLibrary
from app.services.ingestion.document_loader import ParserSelector
from app.services.ingestion.pipeline import IngestionPipeline

from .components import build_components


@lru_cache(maxsize=8)
def get_document_parser(mode: str, ocr: bool = False) -> ParserSelector:
    return ParserSelector(mode=mode, ocr=ocr)


def build_ingestion_pipeline(source: str) -> IngestionPipeline:
    settings = Settings.from_environment()
    components = build_components()
    strategy = LangChainRecursiveStrategy(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return IngestionPipeline(
        loader=get_document_parser(settings.document_parser, settings.document_ocr),
        cleaner=DataCleaningLibrary(),
        chunker=ChunkerService(strategy=strategy),
        embedder=components.embedder,
        vector_store=components.vector_store,
    )
