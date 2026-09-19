"""Seed the persistence layer from the materialised feature store.

    python -m persistence --seed

The tree of Section 9.2 declares no seeding module, but Table 10 requires the
profile repository to be able to answer ``UserNotFoundError``, and the
controller enriches a ranking with catalog titles. Both need the stores to
hold data. This entry point loads them from the artifacts the offline
pipeline already produced, so the databases are never populated from the
request path.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from config.settings import load_settings
from persistence.interfaces import CatalogItem, UserProfile
from persistence.nosql_interaction_history_repository import NoSqlInteractionHistoryRepository
from persistence.sql_catalog_repository import SqlCatalogRepository
from persistence.sql_user_profile_repository import SqlUserProfileRepository
from persistence.vector_embedding_repository import VectorEmbeddingRepository


def seed(*, batch_size: int = 1000) -> dict[str, Any]:
    """Load profiles, catalog, and embeddings from the feature store."""
    import pandas as pd

    settings = load_settings()
    store = settings.feature_store.version_dir

    users = pd.read_parquet(
        store / "user_features.parquet", columns=["user_id", "preferred_category"]
    )
    profiles = SqlUserProfileRepository.from_settings(settings)
    for row in users.to_dict(orient="records"):
        category = row.get("preferred_category")
        user_categories: tuple[str, ...] = (
            () if category is None or pd.isna(category) else (str(category),)
        )
        profiles.save(
            UserProfile(user_id=int(row["user_id"]), preferred_categories=user_categories)
        )

    items = pd.read_parquet(
        store / "item_features.parquet", columns=["item_id", "title", "categories"]
    )
    catalog = SqlCatalogRepository.from_settings(settings)
    written = 0
    batch: list[CatalogItem] = []
    for row in items.to_dict(orient="records"):
        raw_categories = row.get("categories")
        categories = tuple(str(c) for c in raw_categories) if raw_categories is not None else ()
        batch.append(
            CatalogItem(
                item_id=int(row["item_id"]),
                title=str(row.get("title") or f"item {row['item_id']}"),
                categories=categories,
            )
        )
        if len(batch) >= batch_size:
            written += catalog.save_many(batch)
            batch = []
    written += catalog.save_many(batch)

    embeddings = VectorEmbeddingRepository.from_feature_store(settings)
    archive = embeddings.flush()

    history = NoSqlInteractionHistoryRepository.from_settings(settings)

    return {
        "sql_url": settings.persistence.sql_url,
        "profiles": profiles.count(),
        "catalog_items": catalog.count(),
        "catalog_written": written,
        "embeddings": embeddings.count(),
        "embedding_archive": str(archive) if archive else None,
        "history_backend": history.backend,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m persistence",
        description="Seed the persistence layer from the materialised feature store.",
    )
    parser.add_argument(
        "--seed", action="store_true", help="Populate profiles, catalog and embeddings"
    )
    args = parser.parse_args(argv)

    if not args.seed:
        parser.error("nothing to do: pass --seed")

    print(json.dumps(seed(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
