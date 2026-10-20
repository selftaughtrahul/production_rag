"""Document metadata prepared during ingestion."""

import pytest

from app.models.schemas import normalize_metadata
from app.services.ingestion.pipeline import prepare_document_metadata


def test_normalize_metadata_object_and_pairs():
    assert normalize_metadata({"Department": "Finance", "user_id": "nope"}) == {
        "department": "Finance",
    }
    assert normalize_metadata("region: emea, owner=ada") == {
        "region": "emea",
        "owner": "ada",
    }
    assert normalize_metadata("") == {}
    assert normalize_metadata(None) == {}


def test_normalize_metadata_rejects_a_list():
    with pytest.raises(ValueError):
        normalize_metadata(["region:emea"])


def test_prepare_document_metadata_keeps_file_facts():
    prepared = prepare_document_metadata(
        {"title": "Report", "source": "/tmp/upload.pdf", "tags": ["finance"], "page_count": 3},
        filename="report.pdf",
        content_type="application/pdf",
        file_size=1200,
        image_count=0,
    )
    assert prepared["filename"] == "report.pdf"
    assert prepared["title"] == "Report"
    assert prepared["page_count"] == "3"
    assert prepared["content_type"] == "application/pdf"
    assert prepared["file_size"] == "1200"
    assert "source" not in prepared
    assert "tags" not in prepared
