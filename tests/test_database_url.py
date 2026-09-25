"""DATABASE_URL must be a file-based SQLite URL."""

import pytest

from database.sqlite import sqlite_file_path


def test_sqlite_file_path_from_url() -> None:
    assert sqlite_file_path("sqlite:///data/app.db") == "data/app.db"


@pytest.mark.parametrize("url", ["postgresql://rag:rag@localhost:5432/rag", "sqlite://"])
def test_sqlite_file_path_rejects_non_file_urls(url: str) -> None:
    with pytest.raises(ValueError):
        sqlite_file_path(url)
