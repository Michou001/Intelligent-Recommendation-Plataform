"""Layer 5 - Persistence & Storage: segregated repository adapters.

Section 5.2 puts SQL databases behind the profile and catalog repositories
and document storage behind the high-volume interaction history and the
embedding vectors. Both are reached only through the segregated interfaces of
``persistence/interfaces.py``, never through a concrete engine driver.

This module holds what the two SQL adapters share: the SQLAlchemy metadata,
the table definitions, and one engine per URL. The default URL points at a
local SQLite file, so the system runs with no database server; pointing it at
PostgreSQL is the configuration change described in Table 7, and the adapters
below are unchanged by it.
"""

from __future__ import annotations

import threading

from sqlalchemy import (
    Column,
    Engine,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
)

metadata = MetaData()

#: Profiles and explicit preferences (FR-01).
user_profiles = Table(
    "user_profiles",
    metadata,
    Column("user_id", Integer, primary_key=True),
    Column("preferred_categories", Text, nullable=False, default=""),
    Column("attributes", Text, nullable=False, default="{}"),
)

#: The transactional catalog.
catalog_items = Table(
    "catalog_items",
    metadata,
    Column("item_id", Integer, primary_key=True),
    Column("title", String(512), nullable=False),
    Column("categories", Text, nullable=False, default=""),
)

_engines: dict[str, Engine] = {}
_lock = threading.Lock()


def get_engine(url: str, *, create_tables: bool = True) -> Engine:
    """Return the engine for a URL, creating it and the schema once.

    Engines hold a connection pool, so one per URL is shared by both SQL
    adapters instead of each opening its own.
    """
    with _lock:
        engine = _engines.get(url)
        if engine is None:
            connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
            engine = create_engine(url, future=True, connect_args=connect_args)
            _engines[url] = engine
            if create_tables:
                _ensure_parent_directory(url)
                metadata.create_all(engine)
        return engine


def _ensure_parent_directory(url: str) -> None:
    """Create the directory of a SQLite file before the engine touches it."""
    if not url.startswith("sqlite:///"):
        return
    from pathlib import Path

    path = Path(url.removeprefix("sqlite:///"))
    path.parent.mkdir(parents=True, exist_ok=True)


def dispose_engines() -> None:
    """Close every pooled engine, used between tests."""
    with _lock:
        for engine in _engines.values():
            engine.dispose()
        _engines.clear()


__all__ = [
    "catalog_items",
    "dispose_engines",
    "get_engine",
    "metadata",
    "user_profiles",
]
