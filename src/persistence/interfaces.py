"""Segregated persistence contracts (Section 3.2, Table 11).

Milestone 1 exposed a single ``IDataRepository`` with ``save`` and
``findById``. The feedback recorded in Table 6 identified that contract as too
generic, and the correction was to split it by responsibility. This module
therefore declares four independent interfaces, each exposing only the queries
its consumer actually needs:

* ``IUserProfileRepository``   - profiles and explicit preferences (FR-01)
* ``ICatalogRepository``       - the item catalog
* ``IInteractionHistoryRepository`` - interaction and recommendation history (FR-06)
* ``IEmbeddingRepository``     - embedding vectors and similarity search

No universal base repository is declared on purpose: re-introducing a common
``save``/``find_by_id`` ancestor would restore exactly the generic contract
the Milestone 1 correction removed, and would force every client to inherit
operations it does not use (Interface Segregation Principle).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

UserId = int
ItemId = int


@dataclass(frozen=True)
class UserProfile:
    """User profile and explicit preferences (FR-01)."""

    user_id: UserId
    preferred_categories: tuple[str, ...] = ()
    attributes: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class CatalogItem:
    """One entry of the content catalog."""

    item_id: ItemId
    title: str
    categories: tuple[str, ...] = ()


@dataclass(frozen=True)
class InteractionRecord:
    """One persisted interaction and the recommendation shown with it (FR-06).

    ``event_id`` is what makes the write idempotent: a redelivery after a
    worker failure carries the same identifier and must not duplicate the
    record (Section 9.9).
    """

    event_id: str
    user_id: UserId
    event_type: str
    occurred_at: datetime
    item_id: ItemId | None = None
    recommended_items: tuple[ItemId, ...] = ()
    model_version: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)


class IUserProfileRepository(ABC):
    """Profiles and explicit preferences (FR-01, NFR-03)."""

    @abstractmethod
    def get_by_id(self, user_id: UserId) -> UserProfile | None:
        """Return the profile, or None when the user does not exist."""

    @abstractmethod
    def exists(self, user_id: UserId) -> bool:
        """Whether a profile exists, without materialising it."""

    @abstractmethod
    def save(self, profile: UserProfile) -> None:
        """Create or update a profile."""


class ICatalogRepository(ABC):
    """The item catalog, read to enrich a ranking with titles and categories."""

    @abstractmethod
    def get_item(self, item_id: ItemId) -> CatalogItem | None:
        """Return one catalog item, or None when it does not exist."""

    @abstractmethod
    def get_items(self, item_ids: Sequence[ItemId]) -> list[CatalogItem]:
        """Return the requested items, in one round trip, skipping unknown ids."""

    @abstractmethod
    def list_by_category(self, category: str, limit: int = 100) -> list[CatalogItem]:
        """Return items belonging to a content category."""

    @abstractmethod
    def count(self) -> int:
        """Number of items in the catalog."""

    @abstractmethod
    def save(self, item: CatalogItem) -> None:
        """Create or update a catalog item."""


class IInteractionHistoryRepository(ABC):
    """High-volume interaction and recommendation history (FR-06).

    Written exclusively by ``FeedbackHistoryService`` on the asynchronous
    path, never by the request thread.
    """

    @abstractmethod
    def append(self, record: InteractionRecord) -> bool:
        """Persist one record idempotently.

        Returns:
            True when the record was written, False when ``event_id`` was
            already present and the redelivery was ignored.
        """

    @abstractmethod
    def list_by_user(self, user_id: UserId, limit: int = 100) -> list[InteractionRecord]:
        """Return the most recent records of a user, newest first."""

    @abstractmethod
    def count_by_user(self, user_id: UserId) -> int:
        """Number of records stored for a user."""

    @abstractmethod
    def exists_event(self, event_id: str) -> bool:
        """Whether an event identifier was already consumed."""


class IEmbeddingRepository(ABC):
    """Embedding vectors and similarity search.

    Similarity search belongs here and not to the profile repository, which
    is the concrete example the Milestone 1 correction gives for the split.
    """

    @abstractmethod
    def get_embedding(self, item_id: ItemId) -> np.ndarray | None:
        """Return the embedding of an item, or None when it is not stored."""

    @abstractmethod
    def upsert_embedding(self, item_id: ItemId, vector: np.ndarray) -> None:
        """Create or replace the embedding of an item."""

    @abstractmethod
    def similarity_search(self, vector: np.ndarray, k: int = 10) -> list[tuple[ItemId, float]]:
        """Return the K stored embeddings closest to ``vector`` by cosine similarity."""

    @abstractmethod
    def count(self) -> int:
        """Number of stored embeddings."""


__all__ = [
    "CatalogItem",
    "ICatalogRepository",
    "IEmbeddingRepository",
    "IInteractionHistoryRepository",
    "IUserProfileRepository",
    "InteractionRecord",
    "ItemId",
    "UserId",
    "UserProfile",
]
