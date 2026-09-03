from ftfy import fix_text
import unicodedata
import re


class TextNormalizer:
    """
    Production-oriented text normalization service.

    Responsible for making cleaned text consistent
    without removing meaningful semantic information.
    """

    def __init__(self,*,lowercase: bool = False,normalize_unicode: bool = True,normalize_whitespace: bool = True,normalize_quotes: bool = True, normalize_dashes: bool = True,):
        self.lowercase = lowercase
        self.normalize_unicode = normalize_unicode
        self.normalize_whitespace = normalize_whitespace
        self.normalize_quotes = normalize_quotes
        self.normalize_dashes = normalize_dashes

    def normalize(self, text: str) -> str:

        if not isinstance(text, str):
            raise TypeError("text must be a string")

        if not text.strip():
            return ""

        # Fix broken encoding
        text = fix_text(text)

        # Unicode normalization
        if self.normalize_unicode:
            text = unicodedata.normalize("NFKC", text)

        # Normalize quotes
        if self.normalize_quotes:
            text = self._normalize_quotes(text)

        # Normalize dashes
        if self.normalize_dashes:
            text = self._normalize_dashes(text)

        # Normalize whitespace
        if self.normalize_whitespace:
            text = self._normalize_whitespace(text)

        # Lowercase only when explicitly requested
        if self.lowercase:
            text = text.lower()

        return text.strip()

    @staticmethod
    def _normalize_quotes(text: str) -> str:
        replacements = {
            "“": '"',
            "”": '"',
            "„": '"',
            "‘": "'",
            "’": "'",
            "‚": "'",
        }

        return text.translate(
            str.maketrans(replacements)
        )

    @staticmethod
    def _normalize_dashes(text: str) -> str:
        replacements = {
            "–": "-",
            "—": "-",
            "−": "-",
        }

        return text.translate(
            str.maketrans(replacements)
        )

    @staticmethod
    def _normalize_whitespace(text: str) -> str:

        # Spaces and tabs
        text = re.sub(r"[ \t]+", " ", text)

        # Excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()
