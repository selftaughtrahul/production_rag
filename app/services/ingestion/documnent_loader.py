from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llama_index.core import SimpleDirectoryReader


@dataclass(slots=True)
class Document:
    """
    Standard representation of a loaded document.
    """

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentLoaderCustom:
    """
    Base document loader.

    Supports multiple document formats while exposing a single
    load() method to the rest of the application.
    """

    SUPPORTED_EXTENSIONS = {".txt",".pdf",".md",".csv",".json"}

    def __init__(self, source: str | Path):
        self.source = Path(source)
        self.file_extension = self.source.suffix.lower()

    def load(self, source: str | Path | None = None) -> Document:
        """
        Load document and return normalized Document object.
        """
        if source is not None:
            self.source = Path(source)
            self.file_extension = self.source.suffix.lower()

        self.validate_source()

        reader = self._get_reader()

        text = reader()

        return Document(
            text=text,
            metadata=self.get_metadata(),
        )

    def validate_source(self) -> None:
        """
        Validate that source exists and is a file.
        """

        if not self.source.exists():
            raise FileNotFoundError(f"Document does not exist: {self.source}")

        if not self.source.is_file():
            raise ValueError(f"Source is not a file: {self.source}")

        if self.file_extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {self.file_extension}. "
                f"Supported types: {sorted(self.SUPPORTED_EXTENSIONS)}"
            )

    def _get_reader(self):
        """
        Return appropriate reader based on file extension.
        """

        readers = {
            ".txt": self.read_text_file,
            ".md": self.read_text_file,
            ".pdf": self.read_pdf_file,
            ".csv": self.read_csv_file,
            ".json": self.read_json_file,
        }

        try:
            return readers[self.file_extension]
        except KeyError:
            raise ValueError(f"No reader available for: {self.file_extension}")

    def get_metadata(self) -> dict[str, Any]:
        """
        Common metadata for every document.
        """

        return {
            "source": str(self.source),
            "filename": self.source.name,
            "extension": self.file_extension,
        }

    def read_text_file(self) -> str:
        """
        Read TXT / Markdown files.
        """

        return self.source.read_text(encoding="utf-8")

    def read_pdf_file(self) -> str:
        """
        Extract text from PDF.
        """

        import fitz

        text_parts: list[str] = []

        with fitz.open(self.source) as pdf:

            for page in pdf:
                page_text = page.get_text("text")

                if page_text:
                    text_parts.append(page_text)

        return "\n".join(text_parts)

    def read_csv_file(self) -> str:
        """
        Read CSV and convert it into text.

        For RAG, this can later be replaced with a structured
        CSV-to-document strategy.
        """

        import csv

        rows: list[str] = []

        with self.source.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:
                rows.append(" | ".join(f"{key}: {value}" for key, value in row.items()))

        return "\n".join(rows)

    def read_json_file(self) -> str:
        """
        Read JSON and convert it to text.
        """

        import json

        data = json.loads(self.source.read_text(encoding="utf-8"))

        return json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        )


class DocumentLoaderLibrary:
    """
    Application-level document loader.

    Delegates actual document parsing to LlamaIndex
    while exposing our own stable Document interface.
    """

    SUPPORTED_EXTENSIONS = {".txt",".pdf",".md",".csv",".json",".docx",".pptx",".xlsx",}

    def __init__(self, source: str | Path):
        self.source = Path(source)
        self.file_extension = self.source.suffix.lower()

    def load(self, source: str | Path | None = None) -> Document:
        """
        Load a document using the underlying parser library.
        """
        if source is not None:
            self.source = Path(source)
            self.file_extension = self.source.suffix.lower()

        self.validate_source()

        documents = SimpleDirectoryReader(input_files=[self.source]).load_data()

        text = "\n\n".join(document.text for document in documents if document.text)

        metadata = self.get_metadata()

        return Document(
            text=text,
            metadata=metadata,
        )

    def validate_source(self) -> None:
        """
        Validate document source.
        """

        if not self.source.exists():
            raise FileNotFoundError(f"Document does not exist: {self.source}")

        if not self.source.is_file():
            raise ValueError(f"Source is not a file: {self.source}")

        if self.file_extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {self.file_extension}. "
                f"Supported types: "
                f"{sorted(self.SUPPORTED_EXTENSIONS)}"
            )

    def get_metadata(self) -> dict[str, Any]:
        """
        Common application metadata.
        """

        return {
            "source": str(self.source),
            "filename": self.source.name,
            "extension": self.file_extension,
        }
