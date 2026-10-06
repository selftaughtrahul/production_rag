"""Upload checks, image labels, ingest retries, and the dead-letter row."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from app.services.ingestion.document_loader import (
    UploadRejected,
    extract_embedded_images,
    scan_saved_file,
    validate_upload_bytes,
)
from app.tasks.tasks import is_retryable_ingest_error


def test_validate_accepts_pdf_magic() -> None:
    assert validate_upload_bytes("notes.pdf", b"%PDF-1.7\n", max_bytes=100) == ".pdf"


def test_validate_rejects_mismatched_extension() -> None:
    with pytest.raises(UploadRejected, match="does not match"):
        validate_upload_bytes("notes.txt", b"%PDF-1.7", max_bytes=100)


def test_validate_rejects_oversize_file() -> None:
    with pytest.raises(UploadRejected, match="byte limit"):
        validate_upload_bytes("notes.txt", b"hello", max_bytes=3)


def test_scan_skips_when_disabled(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("hello", encoding="utf-8")
    scan_saved_file(path, enabled=False, command="clamscan")


def test_scan_rejects_infected_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("hello", encoding="utf-8")

    class _Result:
        returncode = 1

    monkeypatch.setattr(
        "app.services.ingestion.document_loader.subprocess.run",
        lambda *args, **kwargs: _Result(),
    )
    with pytest.raises(UploadRejected, match="malware scan") as caught:
        scan_saved_file(path, enabled=True, command="clamscan")
    assert caught.value.status_code == 400


def test_scan_fails_closed_when_scanner_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("hello", encoding="utf-8")

    def _missing(*args, **kwargs):
        raise FileNotFoundError("clamscan")

    monkeypatch.setattr("app.services.ingestion.document_loader.subprocess.run", _missing)
    with pytest.raises(UploadRejected, match="unavailable") as caught:
        scan_saved_file(path, enabled=True, command="clamscan")
    assert caught.value.status_code == 503


def test_extract_office_images(tmp_path: Path) -> None:
    source = tmp_path / "brief.docx"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("word/media/chart.png", b"\x89PNG\r\n")
    labels = extract_embedded_images(source, tmp_path / "brief_images")
    assert len(labels) == 1
    assert "chart.png" in labels[0]
    stored = list((tmp_path / "brief_images").iterdir())
    assert len(stored) == 1
    assert stored[0].read_bytes().startswith(b"\x89PNG")


def test_retryable_errors() -> None:
    assert is_retryable_ingest_error(ConnectionError("broker down"))
    assert is_retryable_ingest_error(TimeoutError("slow"))
    assert not is_retryable_ingest_error(ValueError("bad pdf"))
    assert not is_retryable_ingest_error(FileNotFoundError("missing"))


def test_dead_letter_row_is_written() -> None:
    from database.models import IngestionDeadLetter
    from database.sqlite import SessionLocal, init_db
    from app.tasks.tasks import record_ingestion_failure

    init_db()
    document_id = "dead-letter-test-doc"
    record_ingestion_failure(
        document_id=document_id,
        user_id="user-1",
        payload={"filename": "notes.pdf"},
        exc=ConnectionError("qdrant down"),
        retries=3,
    )
    session = SessionLocal()
    try:
        row = session.query(IngestionDeadLetter).filter_by(document_id=document_id).one()
        assert row.error_type == "ConnectionError"
        assert row.retries == 3
        assert "notes.pdf" in row.payload
        session.delete(row)
        session.commit()
    finally:
        session.close()
