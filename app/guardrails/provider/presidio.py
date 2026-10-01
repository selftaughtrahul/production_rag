import logging

logger = logging.getLogger(__name__)

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
except Exception as exc:
    AnalyzerEngine = None
    AnonymizerEngine = None
    logger.warning("Presidio is unavailable and will be skipped: %s", exc)


# Dates and clock times show up in ordinary questions. They are not identity data.
IGNORED_PII_ENTITIES = frozenset({"DATE_TIME"})


class PresidioProvider:
    def __init__(self, analyzer=None, anonymizer=None):
        if analyzer is None or anonymizer is None:
            if AnalyzerEngine is None or AnonymizerEngine is None:
                raise RuntimeError("Presidio/spaCy could not be loaded on this machine.")
            analyzer = analyzer or AnalyzerEngine()
            anonymizer = anonymizer or AnonymizerEngine()
        self.analyzer = analyzer
        self.anonymizer = anonymizer

    def analyze_and_anonymize(self, text: str, entities: list[str] | None = None) -> dict:
        results = [
            item
            for item in self.analyzer.analyze(text=text, entities=entities, language="en")
            if item.entity_type not in IGNORED_PII_ENTITIES
        ]
        if not results:
            return {"passed": True, "anonymized_text": text, "entities": []}

        anonymized = self.anonymizer.anonymize(text=text, analyzer_results=results)
        return {
            "passed": False,
            "anonymized_text": anonymized.text,
            "entities": [res.entity_type for res in results],
        }
