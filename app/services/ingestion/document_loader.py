from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from html.parser import HTMLParser
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any

from llama_index.core import SimpleDirectoryReader

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ParseReport:
    """Parser facts recorded for metrics; never contains document text."""

    parser: str
    parser_version: str | None = None
    document_type: str = ""
    page_count: int | None = None
    element_count: int = 0
    table_count: int = 0
    ocr_used: bool = False
    failed_pages: list[int] = field(default_factory=list)
    processing_seconds: float = 0.0
    fallback_from: str | None = None


@dataclass(slots=True)
class Document:
    """
    Standard representation of a loaded document.
    """

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    report: ParseReport | None = None


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


# ─────────────────────────────────────────────────────────────
# Canonical document model
# ─────────────────────────────────────────────────────────────

ELEMENT_TYPES = frozenset(
    {
        "heading",
        "paragraph",
        "list_item",
        "table",
        "caption",
        "footnote",
        "header",
        "footer",
        "image",
    }
)

# Repeated running headers/footers add noise to every chunk; keep them on the
# canonical document for provenance but leave them out of the flat text.
FLATTEN_EXCLUDED_TYPES = frozenset({"header", "footer"})


@dataclass(slots=True)
class TableData:
    """Structured table grid preserved beside its searchable text."""

    table_id: str
    page_number: int | None = None
    caption: str | None = None
    columns: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    header_rows: int = 1

    def to_markdown(self) -> str:
        """Render a GitHub-style Markdown table; cell labels stay searchable."""
        width = max(
            [len(self.columns)] + [len(row) for row in self.rows] or [0]
        )
        if width == 0:
            return ""

        def pad(cells: list[str]) -> list[str]:
            cleaned = [_markdown_cell(cell) for cell in cells]
            return cleaned + [""] * (width - len(cleaned))

        header = pad(self.columns) if self.columns else [""] * width
        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join("---" for _ in range(width)) + " |",
        ]
        lines.extend("| " + " | ".join(pad(row)) + " |" for row in self.rows)
        return "\n".join(lines)


def _markdown_cell(value: Any) -> str:
    text = " ".join(str(value if value is not None else "").split())
    return text.replace("|", "\\|")


@dataclass(slots=True)
class DocumentElement:
    """One layout unit (paragraph, heading, table, ...) in reading order."""

    type: str
    text: str
    page_number: int | None = None
    section: str | None = None
    order: int = 0
    bbox: tuple[float, float, float, float] | None = None
    confidence: float | None = None
    ocr: bool = False
    table: TableData | None = None

    def __post_init__(self) -> None:
        if self.type not in ELEMENT_TYPES:
            raise ValueError(f"Unknown element type: {self.type!r}")


@dataclass(slots=True)
class CanonicalDocument:
    """Parser-neutral document. Downstream stages see only its flattened text."""

    filename: str
    source: str
    extension: str
    document_type: str
    parser: str
    parser_version: str | None = None
    page_count: int | None = None
    ocr_used: bool = False
    failed_pages: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    elements: list[DocumentElement] = field(default_factory=list)

    @property
    def table_count(self) -> int:
        return sum(1 for element in self.elements if element.type == "table")


def _table_text(element: DocumentElement) -> str:
    """Caption line plus Markdown grid so BM25 can match cell labels."""
    table = element.table
    if table is None:
        return element.text
    markdown = table.to_markdown()
    caption = (table.caption or "").strip()
    parts = [part for part in (caption, markdown or element.text) if part]
    return "\n".join(parts)


def flatten_to_document(
    canonical: CanonicalDocument,
    *,
    processing_seconds: float = 0.0,
    fallback_from: str | None = None,
) -> Document:
    """
    Collapse a CanonicalDocument into the flat ``Document`` the ingestion
    pipeline already consumes. Element texts are joined with a blank line in
    reading order; element text is not stripped so the existing LlamaIndex
    adapter produces the same string as ``DocumentLoaderLibrary.load``.
    """
    parts: list[str] = []
    for element in sorted(canonical.elements, key=lambda item: item.order):
        if element.type in FLATTEN_EXCLUDED_TYPES:
            continue
        text = _table_text(element) if element.type == "table" else element.text
        if text:
            parts.append(text)

    metadata: dict[str, Any] = {
        key: value
        for key, value in canonical.metadata.items()
        if isinstance(value, (str, int, float, bool))
    }
    metadata.update(
        {
            "source": canonical.source,
            "filename": canonical.filename,
            "extension": canonical.extension,
            "parser": canonical.parser,
            "ocr_used": canonical.ocr_used,
        }
    )
    if canonical.page_count is not None:
        metadata["page_count"] = canonical.page_count

    report = ParseReport(
        parser=canonical.parser,
        parser_version=canonical.parser_version,
        document_type=canonical.document_type,
        page_count=canonical.page_count,
        element_count=len(canonical.elements),
        table_count=canonical.table_count,
        ocr_used=canonical.ocr_used,
        failed_pages=list(canonical.failed_pages),
        processing_seconds=processing_seconds,
        fallback_from=fallback_from,
    )
    return Document(text="\n\n".join(parts), metadata=metadata, report=report)


def _document_type(extension: str) -> str:
    return extension.lstrip(".").lower() or "unknown"


def _package_version(name: str) -> str | None:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return None


# ─────────────────────────────────────────────────────────────
# Parser interface + adapters
# ─────────────────────────────────────────────────────────────


class DocumentParser(ABC):
    """Produce a CanonicalDocument from a file on disk."""

    name: str = "base"
    supported_extensions: frozenset[str] = frozenset()

    def supports(self, extension: str) -> bool:
        return extension.lower() in self.supported_extensions

    @abstractmethod
    def parse(self, source: Path) -> CanonicalDocument:
        raise NotImplementedError

    def load(self, source: str | Path) -> Document:
        """Parse and flatten; this is what the ingestion pipeline calls."""
        path = Path(source)
        started = time.perf_counter()
        canonical = self.parse(path)
        return flatten_to_document(
            canonical, processing_seconds=time.perf_counter() - started
        )


class ExistingLlamaIndexParser(DocumentParser):
    """
    The current production parser (LlamaIndex SimpleDirectoryReader) behind
    the DocumentParser interface. One element per LlamaIndex document, so the
    flattened text equals ``DocumentLoaderLibrary.load`` output.
    """

    name = "existing"
    supported_extensions = frozenset(DocumentLoaderLibrary.SUPPORTED_EXTENSIONS)

    def parse(self, source: Path) -> CanonicalDocument:
        library = DocumentLoaderLibrary(source)
        library.validate_source()

        documents = SimpleDirectoryReader(input_files=[library.source]).load_data()

        elements: list[DocumentElement] = []
        for order, item in enumerate(documents):
            page = _coerce_int(
                (item.metadata or {}).get("page_label")
                or (item.metadata or {}).get("page_number")
            )
            elements.append(
                DocumentElement(
                    type="paragraph",
                    text=item.text or "",
                    page_number=page,
                    order=order,
                )
            )

        page_count = len(documents) if library.file_extension == ".pdf" else None
        return CanonicalDocument(
            filename=library.source.name,
            source=str(library.source),
            extension=library.file_extension,
            document_type=_document_type(library.file_extension),
            parser=self.name,
            parser_version=_package_version("llama-index-core"),
            page_count=page_count,
            metadata=library.get_metadata(),
            elements=elements,
        )


def _coerce_int(value: Any) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


_DOCLING_LABELS: dict[str, str] = {
    "title": "heading",
    "section_header": "heading",
    "text": "paragraph",
    "paragraph": "paragraph",
    "list_item": "list_item",
    "table": "table",
    "caption": "caption",
    "footnote": "footnote",
    "page_header": "header",
    "page_footer": "footer",
    "picture": "image",
    "formula": "paragraph",
    "code": "paragraph",
    "checkbox_selected": "list_item",
    "checkbox_unselected": "list_item",
}


class DoclingParser(DocumentParser):
    """
    Layout-aware parsing via Docling. OCR stays off so digital PDFs are not
    re-read through an OCR engine; scanned-page OCR is a later change.
    """

    name = "docling"
    supported_extensions = frozenset({".pdf", ".docx", ".pptx"})

    def __init__(self) -> None:
        self._converter: Any = None

    def _get_converter(self) -> Any:
        if self._converter is not None:
            return self._converter
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        pdf_options = PdfPipelineOptions(do_ocr=False, do_table_structure=True)
        self._converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)}
        )
        return self._converter

    def parse(self, source: Path) -> CanonicalDocument:
        library = DocumentLoaderLibrary(source)
        library.validate_source()
        if not self.supports(library.file_extension):
            raise ValueError(f"Docling parser does not handle {library.file_extension}")

        result = self._get_converter().convert(str(library.source))
        doc = result.document
        if doc is None:
            raise RuntimeError(f"Docling returned no document for {library.source.name}")

        elements: list[DocumentElement] = []
        current_section: str | None = None
        table_index = 0

        for order, (item, _level) in enumerate(doc.iterate_items()):
            label = str(getattr(getattr(item, "label", None), "value", getattr(item, "label", "")))
            element_type = _DOCLING_LABELS.get(label, "paragraph")
            page_number, bbox = _docling_provenance(item)
            text = getattr(item, "text", "") or ""

            table: TableData | None = None
            if element_type == "table":
                table_index += 1
                table = _docling_table(item, doc, table_index, page_number)
                text = text or table.to_markdown()
            elif element_type == "image":
                text = _docling_caption(item, doc)
            elif element_type == "heading" and text.strip():
                current_section = text.strip()

            elements.append(
                DocumentElement(
                    type=element_type,
                    text=text,
                    page_number=page_number,
                    section=current_section if element_type != "heading" else None,
                    order=order,
                    bbox=bbox,
                    table=table,
                )
            )

        failed_pages = _docling_failed_pages(result)
        return CanonicalDocument(
            filename=library.source.name,
            source=str(library.source),
            extension=library.file_extension,
            document_type=_document_type(library.file_extension),
            parser=self.name,
            parser_version=_package_version("docling"),
            page_count=_docling_page_count(doc),
            ocr_used=False,
            failed_pages=failed_pages,
            metadata=library.get_metadata(),
            elements=elements,
        )


def _docling_provenance(item: Any) -> tuple[int | None, tuple[float, float, float, float] | None]:
    prov = getattr(item, "prov", None) or []
    if not prov:
        return None, None
    first = prov[0]
    page = _coerce_int(getattr(first, "page_no", None))
    box = getattr(first, "bbox", None)
    bbox = None
    if box is not None:
        try:
            bbox = (float(box.l), float(box.t), float(box.r), float(box.b))
        except (AttributeError, TypeError, ValueError):
            bbox = None
    return page, bbox


def _docling_caption(item: Any, doc: Any) -> str:
    getter = getattr(item, "caption_text", None)
    if getter is None:
        return ""
    try:
        return str(getter(doc) or "")
    except Exception:  # noqa: BLE001 - caption lookup must not fail the parse
        return ""


def _docling_table(item: Any, doc: Any, index: int, page_number: int | None) -> TableData:
    """Pull the cell grid from Docling's TableData; fall back to its Markdown export."""
    columns: list[str] = []
    rows: list[list[str]] = []
    header_rows = 0

    grid = getattr(getattr(item, "data", None), "grid", None) or []
    for row in grid:
        cells = [str(getattr(cell, "text", "") or "") for cell in row]
        is_header = bool(cells) and all(
            bool(getattr(cell, "column_header", False)) for cell in row
        )
        if is_header and not rows:
            header_rows += 1
            columns = cells if not columns else [
                f"{left} {right}".strip() for left, right in zip(columns, cells)
            ]
        else:
            rows.append(cells)

    if not columns and rows:
        columns, rows = rows[0], rows[1:]
        header_rows = 1

    return TableData(
        table_id=f"table_{index}",
        page_number=page_number,
        caption=_docling_caption(item, doc) or None,
        columns=columns,
        rows=rows,
        header_rows=header_rows,
    )


def _docling_page_count(doc: Any) -> int | None:
    counter = getattr(doc, "num_pages", None)
    if callable(counter):
        try:
            return int(counter())
        except Exception:  # noqa: BLE001 - optional metadata
            return None
    pages = getattr(doc, "pages", None)
    return len(pages) if isinstance(pages, dict) else None


def _docling_failed_pages(result: Any) -> list[int]:
    failed: list[int] = []
    for error in getattr(result, "errors", None) or []:
        page = _coerce_int(getattr(error, "page_no", None))
        if page is not None:
            failed.append(page)
    return failed


_UNSTRUCTURED_CATEGORIES: dict[str, str] = {
    "Title": "heading",
    "Header": "header",
    "Footer": "footer",
    "NarrativeText": "paragraph",
    "Text": "paragraph",
    "UncategorizedText": "paragraph",
    "ListItem": "list_item",
    "Table": "table",
    "FigureCaption": "caption",
    "Footnote": "footnote",
    "Image": "image",
    "Formula": "paragraph",
    "Address": "paragraph",
    "EmailAddress": "paragraph",
    "CodeSnippet": "paragraph",
    "PageBreak": "paragraph",
}


class UnstructuredParser(DocumentParser):
    """
    Unstructured ``partition`` with the ``fast`` strategy: no layout model,
    no Tesseract. Tables arrive as HTML when the partitioner exposes them.
    """

    name = "unstructured"
    supported_extensions = frozenset({".pdf", ".docx", ".pptx"})

    def parse(self, source: Path) -> CanonicalDocument:
        library = DocumentLoaderLibrary(source)
        library.validate_source()
        if not self.supports(library.file_extension):
            raise ValueError(
                f"Unstructured parser does not handle {library.file_extension}"
            )

        from unstructured.partition.auto import partition

        raw_elements = partition(filename=str(library.source), strategy="fast")

        elements: list[DocumentElement] = []
        current_section: str | None = None
        table_index = 0
        max_page: int | None = None

        for order, raw in enumerate(raw_elements):
            category = str(getattr(raw, "category", "") or type(raw).__name__)
            element_type = _UNSTRUCTURED_CATEGORIES.get(category, "paragraph")
            meta = getattr(raw, "metadata", None)
            page_number = _coerce_int(getattr(meta, "page_number", None))
            if page_number is not None:
                max_page = page_number if max_page is None else max(max_page, page_number)
            text = str(getattr(raw, "text", "") or "")

            table: TableData | None = None
            if element_type == "table":
                table_index += 1
                html_text = getattr(meta, "text_as_html", None)
                columns, rows = _parse_html_table(html_text) if html_text else ([], [])
                table = TableData(
                    table_id=f"table_{table_index}",
                    page_number=page_number,
                    columns=columns,
                    rows=rows,
                    header_rows=1 if columns else 0,
                )
                if not columns and not rows:
                    # No grid available; keep the partitioner's text as the cell soup.
                    table.rows = [[text]] if text else []
                    table.header_rows = 0
            elif element_type == "heading" and text.strip():
                current_section = text.strip()

            elements.append(
                DocumentElement(
                    type=element_type,
                    text=text,
                    page_number=page_number,
                    section=current_section if element_type != "heading" else None,
                    order=order,
                    bbox=_unstructured_bbox(meta),
                    table=table,
                )
            )

        return CanonicalDocument(
            filename=library.source.name,
            source=str(library.source),
            extension=library.file_extension,
            document_type=_document_type(library.file_extension),
            parser=self.name,
            parser_version=_package_version("unstructured"),
            page_count=max_page,
            ocr_used=False,
            metadata=library.get_metadata(),
            elements=elements,
        )


def _unstructured_bbox(meta: Any) -> tuple[float, float, float, float] | None:
    coordinates = getattr(meta, "coordinates", None)
    points = getattr(coordinates, "points", None)
    if not points:
        return None
    try:
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
    except (TypeError, ValueError, IndexError):
        return None
    return (min(xs), min(ys), max(xs), max(ys))


class _HtmlTableParser(HTMLParser):
    """Minimal <table> reader: header row from <th>, body rows from <td>."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self.header: list[str] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._row_is_header = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self._row = []
            self._row_is_header = False
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = []
            if tag == "th":
                self._row_is_header = True

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._row is not None and self._cell is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row_is_header and not self.header and not self.rows:
                self.header = self._row
            else:
                self.rows.append(self._row)
            self._row = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


def _parse_html_table(html_text: str) -> tuple[list[str], list[list[str]]]:
    parser = _HtmlTableParser()
    try:
        parser.feed(str(html_text))
        parser.close()
    except Exception:  # noqa: BLE001 - malformed HTML must not fail ingestion
        return [], []
    return parser.header, parser.rows


# ─────────────────────────────────────────────────────────────
# Selection + fallback
# ─────────────────────────────────────────────────────────────

# Formats where a layout parser adds value. Everything else stays on the
# existing LlamaIndex loader regardless of DOCUMENT_PARSER.
STRUCTURED_EXTENSIONS = frozenset({".pdf", ".docx", ".pptx"})


class ParserSelector:
    """
    Choose the parser from DOCUMENT_PARSER and fall back to the existing
    loader on any failure. Exposes ``load()`` so IngestionPipeline sees the
    same interface as DocumentLoaderLibrary.
    """

    def __init__(self, mode: str = "existing") -> None:
        from app.core.config import DOCUMENT_PARSER_CHOICES

        if mode not in DOCUMENT_PARSER_CHOICES:
            raise ValueError(f"Unknown DOCUMENT_PARSER mode: {mode!r}")
        self.mode = mode
        self._existing = ExistingLlamaIndexParser()
        self._parsers: dict[str, DocumentParser] = {}

    def _parser(self, name: str) -> DocumentParser:
        if name not in self._parsers:
            if name == "docling":
                self._parsers[name] = DoclingParser()
            elif name == "unstructured":
                self._parsers[name] = UnstructuredParser()
            else:
                raise ValueError(f"No parser registered for {name!r}")
        return self._parsers[name]

    def candidates(self, extension: str) -> list[str]:
        """Ordered parser names to try before the existing loader."""
        if self.mode == "existing" or extension.lower() not in STRUCTURED_EXTENSIONS:
            return []
        if self.mode == "auto":
            return ["docling", "unstructured"]
        return [self.mode]

    def load(self, source: str | Path) -> Document:
        path = Path(source)
        extension = path.suffix.lower()
        failed: str | None = None

        for name in self.candidates(extension):
            started = time.perf_counter()
            try:
                canonical = self._parser(name).parse(path)
            except ImportError as exc:
                logger.warning(
                    "Parser %s is not installed (%s); trying next parser for %s",
                    name, exc, path.name,
                )
                failed = failed or name
                continue
            except Exception as exc:  # noqa: BLE001 - any parser failure falls back
                logger.warning(
                    "Parser %s failed on %s (%s: %s); trying next parser",
                    name, path.name, type(exc).__name__, exc,
                )
                failed = failed or name
                continue

            if not any(element.text for element in canonical.elements):
                logger.warning(
                    "Parser %s returned no text for %s; trying next parser",
                    name, path.name,
                )
                failed = failed or name
                continue

            return flatten_to_document(
                canonical,
                processing_seconds=time.perf_counter() - started,
                fallback_from=None,
            )

        started = time.perf_counter()
        document = self._existing.load(path)
        if document.report is not None:
            document.report.processing_seconds = time.perf_counter() - started
            document.report.fallback_from = failed
        return document
