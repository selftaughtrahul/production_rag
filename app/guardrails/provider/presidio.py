import logging

logger = logging.getLogger(__name__)

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
except Exception as exc:
    AnalyzerEngine = None
    AnonymizerEngine = None
    logger.warning("Presidio is unavailable and will be skipped: %s", exc)


class PresidioProvider:
    def __init__(self, analyzer=None, anonymizer=None):
        if AnalyzerEngine is None or AnonymizerEngine is None:
            raise RuntimeError("Presidio/spaCy could not be loaded on this machine.")
        self.analyzer = analyzer or AnalyzerEngine()
        self.anonymizer = anonymizer or AnonymizerEngine()

    def analyze_and_anonymize(self, text: str, entities: list[str] = None) -> dict:
        results = self.analyzer.analyze(text=text, entities=entities, language="en")
        if not results:
            return {"passed": True, "anonymized_text": text, "entities": []}

        anonymized = self.anonymizer.anonymize(text=text, analyzer_results=results)
        return {
            "passed": False,
            "anonymized_text": anonymized.text,
            "entities": [res.entity_type for res in results],
        }
