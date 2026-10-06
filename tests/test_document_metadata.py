"""Custom metadata chips."""

import pytest

from app.models.schemas import normalize_metadata


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
