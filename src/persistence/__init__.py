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
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    inspect,
    text,
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

#: Cuentas de acceso de la aplicación web; user_id es el mismo de user_profiles.
cuentas = Table(
    "cuentas",
    metadata,
    Column("user_id", Integer, primary_key=True),
    Column("usuario", String(32), nullable=False, unique=True),
    Column("hash_contrasena", String(256), nullable=False),
    # Columnas agregadas después de crear la tabla: ver _migrar_cuentas.
    Column("correo", String(254), nullable=True),
    Column("totp_secreto", String(64), nullable=True),
    Column("totp_activo", Integer, nullable=False, default=0),
    Column("codigos_respaldo", Text, nullable=False, default=""),
    # El correo se confirma con un código que llega a ese correo.
    Column("correo_verificado", Integer, nullable=False, default=0),
    Column("codigo_correo", String(64), nullable=True),  # hash SHA-256 del código
    Column("codigo_correo_expira", Integer, nullable=True),  # segundos desde 1970
)
Index("ix_cuentas_correo", cuentas.c.correo, unique=True)

#: Columna -> definición SQL, para agregarla a una tabla ``cuentas`` ya existente.
_COLUMNAS_NUEVAS_CUENTAS = {
    "correo": "VARCHAR(254)",
    "totp_secreto": "VARCHAR(64)",
    "totp_activo": "INTEGER NOT NULL DEFAULT 0",
    "codigos_respaldo": "TEXT NOT NULL DEFAULT ''",
    "correo_verificado": "INTEGER NOT NULL DEFAULT 0",
    "codigo_correo": "VARCHAR(64)",
    "codigo_correo_expira": "INTEGER",
}

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
                _migrar_cuentas(engine)
        return engine


def _migrar_cuentas(engine: Engine) -> None:
    """Agrega a una base existente las columnas nuevas de ``cuentas``.

    ``create_all`` crea las tablas que faltan pero no modifica las que ya
    existen, así que una base creada antes del correo y del TOTP se completa
    aquí. Las cuentas viejas quedan sin correo (NULL), lo que el índice único
    permite.
    """
    existentes = {columna["name"] for columna in inspect(engine).get_columns("cuentas")}
    with engine.begin() as conexion:
        for nombre, definicion in _COLUMNAS_NUEVAS_CUENTAS.items():
            if nombre not in existentes:
                conexion.execute(text(f"ALTER TABLE cuentas ADD COLUMN {nombre} {definicion}"))
        conexion.execute(
            text("CREATE UNIQUE INDEX IF NOT EXISTS ix_cuentas_correo ON cuentas (correo)")
        )


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
