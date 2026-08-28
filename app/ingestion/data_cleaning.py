from __future__ import annotations
import html
import re
import unicodedata
from dataclasses import dataclass
from cleantext import clean
from ftfy import fix_text


class DataCleaningCustom:
    """
    Production-oriented text cleaning pipeline for NLP and RAG.

    The cleaner is intentionally conservative:
    - Removes HTML noise
    - Normalizes Unicode
    - Removes URLs/emails when required
    - Removes emojis
    - Normalizes whitespace
    - Removes unwanted control characters
    - Optionally removes special characters
    - Preserves meaningful text for embeddings/retrieval
    """

    def __init__(
        self,
        text: str,
        *,
        remove_urls: bool = True,
        remove_emails: bool = True,
        remove_special_characters: bool = False,
        lowercase: bool = False,
    ):
        self.text = text
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_special_characters = remove_special_characters
        self.lowercase = lowercase

    @staticmethod
    def _normalize_unicode(text: str) -> str:
        """
        Normalize Unicode characters.

        Example:
            café / café -> normalized representation
        """
        return unicodedata.normalize("NFKC", text)

    @staticmethod
    def _remove_html(text: str) -> str:
        """
        Remove HTML tags and decode HTML entities.
        """

        text = html.unescape(text)

        # Remove script and style blocks
        text = re.sub(
            r"<(script|style).*?>.*?</\1>",
            " ",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        # Remove remaining HTML tags
        text = re.sub(r"<[^>]+>", " ", text)

        return text

    @staticmethod
    def _remove_urls(text: str) -> str:
        """
        Remove HTTP/HTTPS URLs.
        """

        url_pattern = re.compile(
            r"https?://\S+|www\.\S+",
            flags=re.IGNORECASE,
        )

        return url_pattern.sub(" ", text)

    @staticmethod
    def _remove_emails(text: str) -> str:
        """
        Remove email addresses.
        """

        email_pattern = re.compile(
            r"\b[A-Za-z0-9._%+-]+@"
            r"[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
        )

        return email_pattern.sub(" ", text)

    @staticmethod
    def _remove_emojis(text: str) -> str:
        """
        Remove emojis and common Unicode symbols.
        """

        emoji_pattern = re.compile(
            "["
            "\U0001F300-\U0001F5FF"
            "\U0001F600-\U0001F64F"
            "\U0001F680-\U0001F6FF"
            "\U0001F700-\U0001F77F"
            "\U0001F780-\U0001F7FF"
            "\U0001F800-\U0001F8FF"
            "\U0001F900-\U0001F9FF"
            "\U0001FA00-\U0001FAFF"
            "\U00002700-\U000027BF"
            "\U00002600-\U000026FF"
            "]+",
            flags=re.UNICODE,
        )

        return emoji_pattern.sub(" ", text)

    @staticmethod
    def _remove_control_characters(text: str) -> str:
        """
        Remove invisible/control characters while preserving
        normal whitespace.
        """

        return "".join(
            char
            for char in text
            if unicodedata.category(char) != "Cc"
            or char in "\n\t"
        )

    @staticmethod
    def _remove_special_characters(text: str) -> str:
        """
        Remove most special characters.

        Keeps:
        - letters
        - numbers
        - whitespace
        """

        return re.sub(
            r"[^\w\s]",
            "",
            text,
            flags=re.UNICODE,
        )

    @staticmethod
    def _normalize_spaces(text: str) -> str:
        """
        Normalize multiple spaces and line breaks.
        """

        text = re.sub(r"[ \t]+", " ", text)

        text = re.sub(
            r"\n\s*\n+",
            "\n\n",
            text,
        )

        return text.strip()
    
    @staticmethod
    def _lowercase(text: str) -> str:
        """
        Convert text to lowercase.
        """

        return text.lower()

    @staticmethod
    def _normalize_repeated_characters(text: str) -> str:
        """
        Reduce excessive repeated characters.

        Example:
            heyyyyyyyy -> heyy
            !!!!!!!!!! -> !!
        """

        text = re.sub(r"(.)\1{3,}", r"\1\1", text)

        return text

    @staticmethod
    def _final_cleanup(text: str) -> str:
        """
        Final cleanup after all transformations.
        """

        text = re.sub(r"[ \t]+", " ", text)

        return text.strip()

    def clean_data(self) -> str:
        """
        Execute the complete cleaning pipeline.
        """

        text = self.text

        if not isinstance(text, str):
            raise TypeError(
                "text must be a string"
            )

        # Unicode
        text = self._normalize_unicode(text)

        # HTML
        text = self._remove_html(text)

        # URLs
        if self.remove_urls:
            text = self._remove_urls(text)

        if self.remove_emails:
            text = self._remove_emails(text)

        text = self._remove_emojis(text)
        text = self._remove_control_characters(text)
        text = self._normalize_repeated_characters(text)

        # Special characters
        if self.remove_special_characters:
            text = self._remove_special_characters(text)

        if self.lowercase:
            text = self._lowercase(text)

        text = self._normalize_spaces(text)
        text = self._final_cleanup(text)

        return text


@dataclass(slots=True)
class DataCleaningLibrary:
    """
    Production-oriented text cleaning pipeline for NLP and RAG.

    Uses:
        - ftfy       -> fixes broken Unicode / mojibake
        - clean-text -> removes URLs, emails, emojis, HTML, etc.

    Designed to preserve meaningful semantic text for:
        - Embeddings
        - RAG
        - Semantic search
        - NLP pipelines
    """

    remove_urls: bool = True
    remove_emails: bool = True
    remove_phone_numbers: bool = False
    remove_numbers: bool = False
    remove_punctuation: bool = False
    remove_currency_symbols: bool = False
    remove_emoji: bool = True
    lowercase: bool = False
    replace_with_url: str = ""
    replace_with_email: str = ""
    replace_with_punct: str = ""

    def clean(self, text: str) -> str:
        """Alias for clean_data."""
        return self.clean_data(text)

    def clean_data(self, text: str) -> str:
        """
        Clean and normalize input text.

        Args:
            text: Raw input text.

        Returns:
            Cleaned text.

        Raises:
            TypeError: If text is not a string.
        """

        if not isinstance(text, str):
            raise TypeError("text must be a string")

        if not text.strip():
            return ""

        # ---------------------------------------------------------
        # 1. Fix broken Unicode / mojibake
        # ---------------------------------------------------------
        text = fix_text(text)

        # ---------------------------------------------------------
        # 2. Clean text using clean-text
        # ---------------------------------------------------------
        text = clean(
            text,
            fix_unicode=True,
            to_ascii=False,

            lower=self.lowercase,

            no_line_breaks=False,

            no_urls=self.remove_urls,
            replace_with_url=self.replace_with_url,

            no_emails=self.remove_emails,
            replace_with_email=self.replace_with_email,

            no_phone_numbers=self.remove_phone_numbers,

            no_numbers=self.remove_numbers,

            no_punct=self.remove_punctuation,
            replace_with_punct=self.replace_with_punct,

            no_emoji=self.remove_emoji,

            no_currency_symbols=self.remove_currency_symbols,

            lang="en",
        )

        # ---------------------------------------------------------
        # 3. Normalize whitespace
        # ---------------------------------------------------------
        text = self._normalize_whitespace(text)

        return text

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """
        Normalize excessive whitespace while preserving
        meaningful line breaks.
        """

        lines = []

        for line in text.splitlines():
            line = " ".join(line.split())

            if line:
                lines.append(line)

        return "\n".join(lines).strip()