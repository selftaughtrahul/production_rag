"""DATABASE_URL conversion for SQLAlchemy vs LangGraph."""

from database.sqlite import checkpoint_conninfo, sqlalchemy_database_url


def test_sqlalchemy_url_uses_psycopg_driver() -> None:
    assert (
        sqlalchemy_database_url("postgresql://rag:rag@localhost:5432/rag")
        == "postgresql+psycopg://rag:rag@localhost:5432/rag"
    )


def test_checkpoint_conninfo_is_libpq() -> None:
    assert (
        checkpoint_conninfo("postgresql+psycopg://rag:rag@db:5432/rag")
        == "postgresql://rag:rag@db:5432/rag"
    )
