"""NoSqlInteractionHistoryRepository - document adapter for FR-06 (Section 3.2).

Implements ``IInteractionHistoryRepository`` over a document store: MongoDB
when a URL is configured (Table 7), and an append-only local file store with
the same behaviour when no database server is available.

``append`` is idempotent by event identifier. That is not a detail: Section
9.9 requires the background worker to be idempotent with respect to the
events it consumes, so that a redelivery after a failure does not duplicate
history records.
"""

from __future__ import annotations

import json
import threading
from abc import ABC, abstractmethod
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from config.settings import Settings, load_settings
from persistence.interfaces import IInteractionHistoryRepository, InteractionRecord, UserId

#: Name of the collection, and of the local file, holding the history.
COLLECTION_NAME = "interaction_history"


class DocumentBackend(ABC):
    """Minimal document contract the repository needs."""

    @abstractmethod
    def insert_if_absent(self, key: str, document: Mapping[str, Any]) -> bool:
        """Insert a document unless its key is already present."""

    @abstractmethod
    def find_by_user(self, user_id: int, limit: int) -> list[Mapping[str, Any]]:
        """Return the newest documents of a user."""

    @abstractmethod
    def count_by_user(self, user_id: int) -> int:
        """Number of documents stored for a user."""

    @abstractmethod
    def has_key(self, key: str) -> bool:
        """Whether an event identifier was already stored."""

    @abstractmethod
    def describe(self) -> str:
        """Backend name, reported by /health."""


class LocalDocumentBackend(DocumentBackend):
    """Append-only JSON Lines store with an in-memory key index.

    Chosen when no MongoDB endpoint is configured. It preserves the two
    properties the design depends on - durability across restarts and
    idempotency by event identifier - without requiring infrastructure.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._keys: set[str] = set()
        self._load_keys()

    def _load_keys(self) -> None:
        if not self._path.exists():
            return
        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    self._keys.add(str(json.loads(line)["event_id"]))
                except (ValueError, KeyError):
                    continue

    def insert_if_absent(self, key: str, document: Mapping[str, Any]) -> bool:
        with self._lock:
            if key in self._keys:
                return False
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(dict(document), default=str) + "\n")
            self._keys.add(key)
            return True

    def _iter_documents(self) -> list[Mapping[str, Any]]:
        if not self._path.exists():
            return []
        documents: list[Mapping[str, Any]] = []
        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    try:
                        documents.append(json.loads(line))
                    except ValueError:
                        continue
        return documents

    def find_by_user(self, user_id: int, limit: int) -> list[Mapping[str, Any]]:
        matching = [d for d in self._iter_documents() if int(d.get("user_id", -1)) == user_id]
        matching.sort(key=lambda d: str(d.get("occurred_at", "")), reverse=True)
        return matching[:limit]

    def count_by_user(self, user_id: int) -> int:
        return sum(1 for d in self._iter_documents() if int(d.get("user_id", -1)) == user_id)

    def has_key(self, key: str) -> bool:
        with self._lock:
            return key in self._keys

    def describe(self) -> str:
        return f"local-file:{self._path.name}"


class MongoDocumentBackend(DocumentBackend):
    """MongoDB backend of Table 7."""

    def __init__(self, collection: Any) -> None:
        self._collection = collection
        self._collection.create_index("event_id", unique=True)
        self._collection.create_index("user_id")

    @classmethod
    def connect(cls, url: str, database: str) -> MongoDocumentBackend:
        """Open a MongoDB client. Raises when the package or the server is absent."""
        from pymongo import MongoClient

        client: Any = MongoClient(url, serverSelectionTimeoutMS=500)
        client.admin.command("ping")
        return cls(client[database][COLLECTION_NAME])

    def insert_if_absent(self, key: str, document: Mapping[str, Any]) -> bool:
        from pymongo.errors import DuplicateKeyError

        try:
            self._collection.insert_one(dict(document))
            return True
        except DuplicateKeyError:
            return False

    def find_by_user(self, user_id: int, limit: int) -> list[Mapping[str, Any]]:
        cursor = (
            self._collection.find({"user_id": user_id}, {"_id": 0})
            .sort("occurred_at", -1)
            .limit(limit)
        )
        return list(cursor)

    def count_by_user(self, user_id: int) -> int:
        return int(self._collection.count_documents({"user_id": user_id}))

    def has_key(self, key: str) -> bool:
        return self._collection.find_one({"event_id": key}, {"_id": 1}) is not None

    def describe(self) -> str:
        return "mongodb"


class NoSqlInteractionHistoryRepository(IInteractionHistoryRepository):
    """Document storage of the interaction and recommendation history (FR-06)."""

    def __init__(self, backend: DocumentBackend) -> None:
        self._backend = backend

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> NoSqlInteractionHistoryRepository:
        """Use MongoDB when configured and reachable, otherwise the local store."""
        settings = settings or load_settings()
        if settings.persistence.mongo_url:
            try:
                return cls(
                    MongoDocumentBackend.connect(
                        settings.persistence.mongo_url, settings.persistence.mongo_database
                    )
                )
            except Exception:  # noqa: BLE001 - absent infrastructure is not an error here
                pass
        path = settings.persistence.document_store_root / f"{COLLECTION_NAME}.jsonl"
        return cls(LocalDocumentBackend(path))

    @property
    def backend(self) -> str:
        """Name of the backend in use, reported by /health."""
        return self._backend.describe()

    def append(self, record: InteractionRecord) -> bool:
        return self._backend.insert_if_absent(record.event_id, _to_document(record))

    def list_by_user(self, user_id: UserId, limit: int = 100) -> list[InteractionRecord]:
        return [_to_record(d) for d in self._backend.find_by_user(int(user_id), limit)]

    def count_by_user(self, user_id: UserId) -> int:
        return self._backend.count_by_user(int(user_id))

    def exists_event(self, event_id: str) -> bool:
        return self._backend.has_key(event_id)


def _to_document(record: InteractionRecord) -> dict[str, Any]:
    return {
        "event_id": record.event_id,
        "user_id": int(record.user_id),
        "event_type": record.event_type,
        "occurred_at": record.occurred_at.isoformat(),
        "item_id": None if record.item_id is None else int(record.item_id),
        "recommended_items": [int(i) for i in record.recommended_items],
        "model_version": record.model_version,
        "metadata": dict(record.metadata),
    }


def _to_record(document: Mapping[str, Any]) -> InteractionRecord:
    raw_timestamp = str(document.get("occurred_at") or "")
    try:
        occurred_at = datetime.fromisoformat(raw_timestamp)
    except ValueError:
        occurred_at = datetime.now(UTC)
    return InteractionRecord(
        event_id=str(document.get("event_id", "")),
        user_id=int(document.get("user_id", 0)),
        event_type=str(document.get("event_type", "unknown")),
        occurred_at=occurred_at,
        item_id=document.get("item_id"),
        recommended_items=tuple(int(i) for i in document.get("recommended_items", [])),
        model_version=document.get("model_version"),
        metadata={str(k): str(v) for k, v in dict(document.get("metadata", {})).items()},
    )


__all__ = [
    "DocumentBackend",
    "LocalDocumentBackend",
    "MongoDocumentBackend",
    "NoSqlInteractionHistoryRepository",
]
