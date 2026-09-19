"""SqlCatalogRepository - relational adapter for the item catalog (Section 3.2).

Implements ``ICatalogRepository``. The controller uses it to turn a ranking of
identifiers into a response carrying titles and categories; it never asks it
for anything related to ranking, which belongs to the AI layer.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sqlalchemy import Engine, func, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from config.settings import Settings, load_settings
from persistence import catalog_items, get_engine
from persistence.interfaces import CatalogItem, ICatalogRepository, ItemId

CATEGORY_SEPARATOR = "|"


class SqlCatalogRepository(ICatalogRepository):
    """Relational storage of the content catalog."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> SqlCatalogRepository:
        settings = settings or load_settings()
        return cls(get_engine(settings.persistence.sql_url))

    def get_item(self, item_id: ItemId) -> CatalogItem | None:
        statement = select(catalog_items).where(catalog_items.c.item_id == int(item_id))
        with self._engine.connect() as connection:
            row = connection.execute(statement).mappings().first()
        return None if row is None else _to_item(dict(row))

    def get_items(self, item_ids: Sequence[ItemId]) -> list[CatalogItem]:
        """Fetch a whole ranking in one round trip, preserving the given order."""
        if not item_ids:
            return []
        wanted = [int(i) for i in item_ids]
        statement = select(catalog_items).where(catalog_items.c.item_id.in_(wanted))
        with self._engine.connect() as connection:
            rows = {
                int(row["item_id"]): _to_item(dict(row))
                for row in connection.execute(statement).mappings()
            }
        return [rows[i] for i in wanted if i in rows]

    def list_by_category(self, category: str, limit: int = 100) -> list[CatalogItem]:
        pattern = f"%{category}%"
        statement = (
            select(catalog_items).where(catalog_items.c.categories.like(pattern)).limit(limit)
        )
        with self._engine.connect() as connection:
            return [_to_item(dict(row)) for row in connection.execute(statement).mappings()]

    def count(self) -> int:
        with self._engine.connect() as connection:
            return int(
                connection.execute(select(func.count()).select_from(catalog_items)).scalar_one()
            )

    def save(self, item: CatalogItem) -> None:
        self.save_many([item])

    def save_many(self, items: Sequence[CatalogItem]) -> int:
        """Insert or update a batch, so seeding a full catalog is one transaction."""
        if not items:
            return 0
        values = [
            {
                "item_id": int(item.item_id),
                "title": item.title,
                "categories": CATEGORY_SEPARATOR.join(item.categories),
            }
            for item in items
        ]
        with self._engine.begin() as connection:
            if connection.dialect.name == "sqlite":
                statement = sqlite_insert(catalog_items)
                statement = statement.on_conflict_do_update(
                    index_elements=[catalog_items.c.item_id],
                    set_={
                        "title": statement.excluded.title,
                        "categories": statement.excluded.categories,
                    },
                )
                connection.execute(statement, values)
            else:
                for row in values:
                    existing = connection.execute(
                        select(func.count())
                        .select_from(catalog_items)
                        .where(catalog_items.c.item_id == row["item_id"])
                    ).scalar_one()
                    if existing:
                        connection.execute(
                            catalog_items.update()
                            .where(catalog_items.c.item_id == row["item_id"])
                            .values(**row)
                        )
                    else:
                        connection.execute(catalog_items.insert().values(**row))
        return len(values)


def _to_item(row: dict[str, Any]) -> CatalogItem:
    raw = str(row.get("categories") or "")
    return CatalogItem(
        item_id=int(row["item_id"]),
        title=str(row.get("title") or ""),
        categories=tuple(c for c in raw.split(CATEGORY_SEPARATOR) if c),
    )


__all__ = ["SqlCatalogRepository"]
