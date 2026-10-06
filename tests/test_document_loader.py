"""Parser adapter tests: existing-loader regression, flatten, flag, fallback."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

from app.services.ingestion.document_loader import (
    CanonicalDocument,
    DocumentElement,
    DocumentLoaderLibrary,
    DocumentParser,
    ExistingLlamaIndexParser,
    ParseReport,
    ParserSelector,
    TableData,
    flatten_to_document,
)


@pytest.fixture
def text_file(tmp_path: Path) -> Path:
    path = tmp_path / "notes.txt"
    path.write_text("Return window is 30 days.\n\nRefunds go to the original card.", encoding="utf-8")
    return path


@pytest.fixture
def pdf_file(tmp_path: Path) -> Path:
    """Two-page PDF built with pypdf; text drawn with a bare content stream."""
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    for label in ("Page one body text.", "Page two body text."):
        page = writer.add_blank_page(width=300, height=300)
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        font_ref = writer._add_object(font)
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})}
        )
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 20 150 Td ({label}) Tj ET".encode("latin-1"))
        page[NameObject("/Contents")] = writer._add_object(stream)
    path = tmp_path / "two_pages.pdf"
    with path.open("wb") as handle:
        writer.write(handle)
    return path


# ─────────────────────────────────────────────────────────────
# Existing loader regression
# ─────────────────────────────────────────────────────────────


def test_existing_parser_matches_library_loader_for_text(text_file: Path) -> None:
    library = DocumentLoaderLibrary(text_file).load()
    adapted = ExistingLlamaIndexParser().load(text_file)

    assert adapted.text == library.text
    for key in ("source", "filename", "extension"):
        assert adapted.metadata[key] == library.metadata[key]
    assert adapted.metadata["parser"] == "existing"
    assert adapted.metadata["ocr_used"] is False
    assert isinstance(adapted.report, ParseReport)
    assert adapted.report.parser == "existing"
    assert adapted.report.document_type == "txt"


def test_existing_parser_matches_library_loader_for_pdf(pdf_file: Path) -> None:
    library = DocumentLoaderLibrary(pdf_file).load()
    adapted = ExistingLlamaIndexParser().load(pdf_file)

    assert adapted.text == library.text
    assert "Page one body text." in adapted.text
    assert "Page two body text." in adapted.text
    assert adapted.report is not None
    assert adapted.report.page_count == 2
    assert adapted.metadata["page_count"] == 2


def test_library_loader_metadata_keys_unchanged(text_file: Path) -> None:
    document = DocumentLoaderLibrary(text_file).load()
    assert set(document.metadata) == {"source", "filename", "extension"}
    assert document.report is None


# ─────────────────────────────────────────────────────────────
# Canonical flatten
# ─────────────────────────────────────────────────────────────


def _canonical(elements: list[DocumentElement]) -> CanonicalDocument:
    return CanonicalDocument(
        filename="report.pdf",
        source="documents/u1/report.pdf",
        extension=".pdf",
        document_type="pdf",
        parser="docling",
        parser_version="2.0",
        page_count=3,
        elements=elements,
    )


def test_flatten_keeps_table_cells_and_headers_searchable() -> None:
    table = TableData(
        table_id="table_1",
        page_number=2,
        caption="Quarterly results",
        columns=["Metric", "Q4 2024"],
        rows=[["Adjusted EBITDA", "1,234"], ["Revenue", "5,678"]],
    )
    canonical = _canonical(
        [
            DocumentElement(type="heading", text="Financial Summary", order=0, page_number=1),
            DocumentElement(type="paragraph", text="Results improved.", order=1, page_number=1),
            DocumentElement(type="table", text="", order=2, page_number=2, table=table),
        ]
    )

    document = flatten_to_document(canonical)

    assert document.text.startswith("Financial Summary\n\nResults improved.\n\n")
    assert "Quarterly results" in document.text
    assert "| Metric | Q4 2024 |" in document.text
    assert "| Adjusted EBITDA | 1,234 |" in document.text
    assert "| Revenue | 5,678 |" in document.text
    assert document.report is not None
    assert document.report.table_count == 1
    assert document.report.element_count == 3
    assert document.metadata["page_count"] == 3
    assert document.metadata["parser"] == "docling"


def test_flatten_respects_reading_order_and_skips_running_headers() -> None:
    canonical = _canonical(
        [
            DocumentElement(type="paragraph", text="second", order=2),
            DocumentElement(type="header", text="ACME Corp - Confidential", order=0),
            DocumentElement(type="paragraph", text="first", order=1),
            DocumentElement(type="footer", text="Page 1 of 9", order=3),
        ]
    )

    document = flatten_to_document(canonical)

    assert document.text == "first\n\nsecond"


def test_flatten_only_copies_scalar_metadata() -> None:
    canonical = _canonical([DocumentElement(type="paragraph", text="x", order=0)])
    canonical.metadata = {"title": "Report", "nested": {"a": 1}, "tags": ["a"]}

    document = flatten_to_document(canonical)

    assert document.metadata["title"] == "Report"
    assert "nested" not in document.metadata
    assert "tags" not in document.metadata


def test_table_markdown_escapes_pipes_and_pads_short_rows() -> None:
    table = TableData(table_id="t", columns=["A", "B", "C"], rows=[["x|y"]])
    rendered = table.to_markdown()
    assert rendered.splitlines()[0] == "| A | B | C |"
    assert rendered.splitlines()[2] == "| x\\|y |  |  |"


def test_unknown_element_type_is_rejected() -> None:
    with pytest.raises(ValueError):
        DocumentElement(type="banner", text="x")


# ─────────────────────────────────────────────────────────────
# DOCUMENT_PARSER flag
# ─────────────────────────────────────────────────────────────


def _reload_settings(monkeypatch: pytest.MonkeyPatch, value: str | None):
    if value is None:
        monkeypatch.delenv("DOCUMENT_PARSER", raising=False)
    else:
        monkeypatch.setenv("DOCUMENT_PARSER", value)
    import app.core.config as config

    return importlib.reload(config).Settings.from_environment()


def test_settings_default_parser_is_existing(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _reload_settings(monkeypatch, None).document_parser == "existing"


@pytest.mark.parametrize("value", ["docling", "Unstructured", " auto "])
def test_settings_accepts_known_parsers(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    assert _reload_settings(monkeypatch, value).document_parser == value.strip().lower()


def test_settings_rejects_unknown_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="DOCUMENT_PARSER"):
        _reload_settings(monkeypatch, "pymupdf")


def test_selector_existing_mode_never_imports_docling(text_file: Path, pdf_file: Path) -> None:
    sys.modules.pop("docling", None)
    selector = ParserSelector(mode="existing")

    assert selector.candidates(".pdf") == []
    document = selector.load(pdf_file)

    assert document.report is not None
    assert document.report.parser == "existing"
    assert document.report.fallback_from is None
    assert "docling" not in sys.modules
    assert selector.load(text_file).text == DocumentLoaderLibrary(text_file).load().text


def test_selector_candidate_order() -> None:
    assert ParserSelector(mode="auto").candidates(".pdf") == ["docling", "unstructured"]
    assert ParserSelector(mode="docling").candidates(".docx") == ["docling"]
    assert ParserSelector(mode="unstructured").candidates(".pptx") == ["unstructured"]
    # Plain-text formats always stay on the existing loader.
    assert ParserSelector(mode="auto").candidates(".txt") == []
    assert ParserSelector(mode="docling").candidates(".xlsx") == []


def test_selector_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError):
        ParserSelector(mode="pymupdf")


# ─────────────────────────────────────────────────────────────
# Fallback
# ─────────────────────────────────────────────────────────────


class _BrokenParser(DocumentParser):
    name = "broken"
    supported_extensions = frozenset({".pdf"})

    def parse(self, source: Path) -> CanonicalDocument:
        raise RuntimeError("layout model exploded")


class _MissingDependencyParser(DocumentParser):
    name = "missing"
    supported_extensions = frozenset({".pdf"})

    def parse(self, source: Path) -> CanonicalDocument:
        raise ImportError("No module named 'docling'")


class _EmptyParser(DocumentParser):
    name = "empty"
    supported_extensions = frozenset({".pdf"})

    def parse(self, source: Path) -> CanonicalDocument:
        return CanonicalDocument(
            filename=source.name,
            source=str(source),
            extension=".pdf",
            document_type="pdf",
            parser=self.name,
            elements=[DocumentElement(type="image", text="", order=0)],
        )


@pytest.mark.parametrize(
    "parser_cls",
    [_BrokenParser, _MissingDependencyParser, _EmptyParser],
)
def test_selector_falls_back_to_existing_loader(pdf_file: Path, parser_cls) -> None:
    selector = ParserSelector(mode="docling")
    selector._parsers["docling"] = parser_cls()

    document = selector.load(pdf_file)

    assert document.text == DocumentLoaderLibrary(pdf_file).load().text
    assert document.report is not None
    assert document.report.parser == "existing"
    assert document.report.fallback_from == "docling"


def test_selector_uses_structured_parser_when_it_succeeds(pdf_file: Path) -> None:
    class _GoodParser(DocumentParser):
        name = "docling"
        supported_extensions = frozenset({".pdf"})

        def parse(self, source: Path) -> CanonicalDocument:
            return CanonicalDocument(
                filename=source.name,
                source=str(source),
                extension=".pdf",
                document_type="pdf",
                parser=self.name,
                page_count=2,
                elements=[DocumentElement(type="paragraph", text="layout text", order=0)],
            )

    selector = ParserSelector(mode="docling")
    selector._parsers["docling"] = _GoodParser()

    document = selector.load(pdf_file)

    assert document.text == "layout text"
    assert document.report is not None
    assert document.report.parser == "docling"
    assert document.report.fallback_from is None
    assert document.report.processing_seconds >= 0.0


# ─────────────────────────────────────────────────────────────
# Optional live smoke tests (skipped when the library is absent)
# ─────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    importlib.util.find_spec("docling") is None or os.getenv("RUN_DOCLING_SMOKE") != "1",
    reason="docling not installed or RUN_DOCLING_SMOKE != 1",
)
def test_docling_smoke(pdf_file: Path) -> None:
    from app.services.ingestion.document_loader import DoclingParser

    canonical = DoclingParser().parse(pdf_file)
    assert canonical.parser == "docling"
    assert canonical.page_count == 2
    assert any("Page one body text" in element.text for element in canonical.elements)


@pytest.mark.skipif(
    importlib.util.find_spec("unstructured") is None,
    reason="unstructured not installed",
)
def test_unstructured_smoke_or_clean_fallback(pdf_file: Path) -> None:
    """With extras missing, partition raises and the selector must fall back."""
    document = ParserSelector(mode="unstructured").load(pdf_file)
    assert "Page one body text." in document.text
    assert document.report is not None
    assert document.report.parser in {"unstructured", "existing"}
