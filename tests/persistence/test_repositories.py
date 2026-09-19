"""The four segregated repositories (Section 3.2, Table 6).

The correction recorded in Table 6 was to split one generic contract into
four. These tests hold that split in place: each interface is checked against
its own adapter, and the suite asserts that no universal repository was
reintroduced.
"""

from __future__ import annotations

import inspect
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from config.settings import Settings
from persistence import interfaces
from persistence.interfaces import (
    CatalogItem,
    ICatalogRepository,
    IEmbeddingRepository,
    IInteractionHistoryRepository,
    InteractionRecord,
    IUserProfileRepository,
    UserProfile,
)
from persistence.nosql_interaction_history_repository import (
    LocalDocumentBackend,
    NoSqlInteractionHistoryRepository,
)
from persistence.sql_catalog_repository import SqlCatalogRepository
from persistence.sql_user_profile_repository import SqlUserProfileRepository
from persistence.vector_embedding_repository import VectorEmbeddingRepository

# --------------------------------------------------------------------------
# Interface segregation
# --------------------------------------------------------------------------


def test_the_contract_is_split_into_four_interfaces() -> None:
    for interface in (
        IUserProfileRepository,
        ICatalogRepository,
        IInteractionHistoryRepository,
        IEmbeddingRepository,
    ):
        assert inspect.isabstract(interface)
        assert interface.__abstractmethods__


def test_no_universal_repository_was_reintroduced() -> None:
    """A common save/find_by_id ancestor is what the correction removed."""
    assert not hasattr(interfaces, "IDataRepository")
    for interface in (
        IUserProfileRepository,
        ICatalogRepository,
        IInteractionHistoryRepository,
        IEmbeddingRepository,
    ):
        bases = [b for b in interface.__bases__ if b.__name__ != "ABC"]
        assert not bases, f"{interface.__name__} inherits from {bases}"


def test_similarity_search_belongs_to_the_embedding_repository() -> None:
    """The concrete example Table 6 gives for the split."""
    assert "similarity_search" in IEmbeddingRepository.__abstractmethods__
    assert not hasattr(IUserProfileRepository, "similarity_search")


def test_each_interface_exposes_only_its_own_methods() -> None:
    profile = set(IUserProfileRepository.__abstractmethods__)
    history = set(IInteractionHistoryRepository.__abstractmethods__)
    assert not profile & history


# --------------------------------------------------------------------------
# SqlUserProfileRepository
# --------------------------------------------------------------------------


@pytest.fixture
def profiles(settings: Settings) -> SqlUserProfileRepository:
    return SqlUserProfileRepository.from_settings(settings)


def test_a_profile_round_trips(profiles: SqlUserProfileRepository) -> None:
    profiles.save(UserProfile(user_id=1, preferred_categories=("Drama", "Comedy")))
    stored = profiles.get_by_id(1)
    assert stored is not None
    assert stored.preferred_categories == ("Drama", "Comedy")


def test_an_absent_profile_is_none(profiles: SqlUserProfileRepository) -> None:
    assert profiles.get_by_id(4242) is None
    assert profiles.exists(4242) is False


def test_saving_twice_updates_instead_of_duplicating(profiles: SqlUserProfileRepository) -> None:
    profiles.save(UserProfile(user_id=1, preferred_categories=("Drama",)))
    profiles.save(UserProfile(user_id=1, preferred_categories=("Action",)))
    assert profiles.count() == 1
    stored = profiles.get_by_id(1)
    assert stored is not None and stored.preferred_categories == ("Action",)


def test_attributes_round_trip(profiles: SqlUserProfileRepository) -> None:
    profiles.save(UserProfile(user_id=2, attributes={"country": "MX"}))
    stored = profiles.get_by_id(2)
    assert stored is not None and stored.attributes == {"country": "MX"}


# --------------------------------------------------------------------------
# SqlCatalogRepository
# --------------------------------------------------------------------------


@pytest.fixture
def catalog(settings: Settings) -> SqlCatalogRepository:
    repository = SqlCatalogRepository.from_settings(settings)
    repository.save_many(
        [
            CatalogItem(item_id=101, title="First", categories=("Drama",)),
            CatalogItem(item_id=102, title="Second", categories=("Comedy", "Drama")),
            CatalogItem(item_id=103, title="Third", categories=("Action",)),
        ]
    )
    return repository


def test_an_item_round_trips(catalog: SqlCatalogRepository) -> None:
    item = catalog.get_item(102)
    assert item is not None
    assert item.title == "Second"
    assert item.categories == ("Comedy", "Drama")


def test_a_batch_is_fetched_in_the_requested_order(catalog: SqlCatalogRepository) -> None:
    items = catalog.get_items([103, 101])
    assert [item.item_id for item in items] == [103, 101]


def test_unknown_identifiers_are_skipped_not_fatal(catalog: SqlCatalogRepository) -> None:
    items = catalog.get_items([101, 9999])
    assert [item.item_id for item in items] == [101]


def test_an_empty_batch_is_an_empty_result(catalog: SqlCatalogRepository) -> None:
    assert catalog.get_items([]) == []


def test_items_can_be_listed_by_category(catalog: SqlCatalogRepository) -> None:
    titles = {item.title for item in catalog.list_by_category("Drama")}
    assert titles == {"First", "Second"}


def test_count_reflects_the_batch(catalog: SqlCatalogRepository) -> None:
    assert catalog.count() == 3


# --------------------------------------------------------------------------
# NoSqlInteractionHistoryRepository
# --------------------------------------------------------------------------


@pytest.fixture
def history(tmp_path: Path) -> NoSqlInteractionHistoryRepository:
    return NoSqlInteractionHistoryRepository(LocalDocumentBackend(tmp_path / "history.jsonl"))


def _record(event_id: str = "e1", user_id: int = 7) -> InteractionRecord:
    return InteractionRecord(
        event_id=event_id,
        user_id=user_id,
        event_type="recommendation_served",
        occurred_at=datetime.now(UTC),
        recommended_items=(101, 102),
        model_version="v1",
    )


def test_a_record_round_trips(history: NoSqlInteractionHistoryRepository) -> None:
    assert history.append(_record()) is True
    stored = history.list_by_user(7)
    assert len(stored) == 1
    assert stored[0].recommended_items == (101, 102)


def test_the_write_is_idempotent(history: NoSqlInteractionHistoryRepository) -> None:
    """Section 9.9: a redelivery must not duplicate the history."""
    record = _record()
    assert history.append(record) is True
    assert history.append(record) is False
    assert history.count_by_user(7) == 1


def test_events_are_recognised_by_identifier(history: NoSqlInteractionHistoryRepository) -> None:
    history.append(_record("known"))
    assert history.exists_event("known") is True
    assert history.exists_event("unknown") is False


def test_the_history_survives_a_restart(tmp_path: Path) -> None:
    path = tmp_path / "history.jsonl"
    NoSqlInteractionHistoryRepository(LocalDocumentBackend(path)).append(_record())
    reopened = NoSqlInteractionHistoryRepository(LocalDocumentBackend(path))
    assert reopened.exists_event("e1") is True
    assert reopened.count_by_user(7) == 1


def test_records_are_returned_newest_first(history: NoSqlInteractionHistoryRepository) -> None:
    from datetime import timedelta

    now = datetime.now(UTC)
    for index in range(3):
        history.append(
            InteractionRecord(
                event_id=f"e{index}",
                user_id=7,
                event_type="click",
                occurred_at=now + timedelta(seconds=index),
            )
        )
    stored = history.list_by_user(7)
    assert [r.event_id for r in stored] == ["e2", "e1", "e0"]


def test_the_limit_is_honoured(history: NoSqlInteractionHistoryRepository) -> None:
    for index in range(5):
        history.append(_record(f"e{index}"))
    assert len(history.list_by_user(7, limit=2)) == 2


# --------------------------------------------------------------------------
# VectorEmbeddingRepository
# --------------------------------------------------------------------------


@pytest.fixture
def embeddings(tmp_path: Path) -> VectorEmbeddingRepository:
    repository = VectorEmbeddingRepository(tmp_path / "vectors.npz")
    repository.replace_all(
        np.array([10, 11, 12], dtype=np.int64),
        np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]], dtype=np.float32),
    )
    return repository


def test_an_embedding_round_trips(embeddings: VectorEmbeddingRepository) -> None:
    vector = embeddings.get_embedding(10)
    assert vector is not None
    assert np.allclose(vector, [1.0, 0.0])
    assert embeddings.get_embedding(999) is None


def test_similarity_search_ranks_by_cosine(embeddings: VectorEmbeddingRepository) -> None:
    result = embeddings.similarity_search(np.array([1.0, 0.0], dtype=np.float32), k=3)
    assert [item for item, _score in result] == [10, 12, 11]
    assert result[0][1] == pytest.approx(1.0, abs=1e-6)


def test_similarity_search_is_bounded_by_the_store(embeddings: VectorEmbeddingRepository) -> None:
    assert len(embeddings.similarity_search(np.array([1.0, 0.0], dtype=np.float32), k=99)) == 3


def test_an_upsert_replaces_in_place(embeddings: VectorEmbeddingRepository) -> None:
    embeddings.upsert_embedding(10, np.array([0.0, 1.0], dtype=np.float32))
    assert embeddings.count() == 3
    vector = embeddings.get_embedding(10)
    assert vector is not None and np.allclose(vector, [0.0, 1.0])


def test_an_upsert_can_add_a_new_item(embeddings: VectorEmbeddingRepository) -> None:
    embeddings.upsert_embedding(13, np.array([0.5, 0.5], dtype=np.float32))
    assert embeddings.count() == 4


def test_a_dimension_mismatch_is_rejected(embeddings: VectorEmbeddingRepository) -> None:
    with pytest.raises(ValueError, match="dimension"):
        embeddings.upsert_embedding(14, np.array([1.0, 2.0, 3.0], dtype=np.float32))
    with pytest.raises(ValueError, match="dimension"):
        embeddings.similarity_search(np.array([1.0, 2.0, 3.0], dtype=np.float32))


def test_the_store_survives_a_restart(
    tmp_path: Path, embeddings: VectorEmbeddingRepository
) -> None:
    path = embeddings.flush()
    assert path is not None
    reopened = VectorEmbeddingRepository(path)
    assert reopened.count() == 3


def test_an_empty_store_searches_to_nothing(tmp_path: Path) -> None:
    assert VectorEmbeddingRepository(tmp_path / "empty.npz").similarity_search(np.zeros(3)) == []
