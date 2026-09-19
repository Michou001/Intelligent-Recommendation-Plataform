"""VectorEmbeddingRepository - adapter for embedding vectors (Section 3.2).

Implements ``IEmbeddingRepository``. Similarity search lives here and not in
the profile repository, which is the concrete example Table 6 gives for why
the Milestone 1 contract had to be segregated.

The vectors are held in a dense matrix, normalised once when loaded, so a
similarity search is a matrix-vector product followed by ``argpartition``
rather than a scan. The store is persisted as a single compressed NumPy
archive, which needs no vector database for the dataset variant of this
milestone (Section 9.7 leaves an ANN index deliberately out of scope).
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from config.settings import Settings, load_settings
from persistence.interfaces import IEmbeddingRepository, ItemId

#: Name of the archive holding the identifiers and the matrix.
ARCHIVE_NAME = "item_embeddings.npz"


class VectorEmbeddingRepository(IEmbeddingRepository):
    """Dense embedding store with cosine similarity search."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = Path(path) if path is not None else None
        self._lock = threading.Lock()
        self._item_ids: np.ndarray = np.empty(0, dtype=np.int64)
        self._matrix: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self._normalized: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self._position: dict[int, int] = {}
        if self._path is not None and self._path.exists():
            self._load()

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> VectorEmbeddingRepository:
        settings = settings or load_settings()
        return cls(settings.persistence.document_store_root / ARCHIVE_NAME)

    @classmethod
    def from_feature_store(cls, settings: Settings | None = None) -> VectorEmbeddingRepository:
        """Populate the store from the materialised item content embeddings."""
        settings = settings or load_settings()
        import pandas as pd

        source = settings.feature_store.version_dir / "item_features.parquet"
        if not source.exists():
            raise FileNotFoundError(f"missing {source}; run the offline pipeline first")
        frame = pd.read_parquet(source, columns=["item_id", "content_embedding"])

        repository = cls(settings.persistence.document_store_root / ARCHIVE_NAME)
        item_ids = frame["item_id"].to_numpy(dtype=np.int64)
        matrix = np.vstack(
            [np.asarray(row, dtype=np.float32) for row in frame["content_embedding"]]
        )
        repository.replace_all(item_ids, matrix)
        return repository

    def replace_all(self, item_ids: np.ndarray, matrix: np.ndarray) -> None:
        """Replace the whole store in one operation, used when seeding."""
        with self._lock:
            self._item_ids = np.asarray(item_ids, dtype=np.int64)
            self._matrix = np.ascontiguousarray(matrix, dtype=np.float32)
            self._reindex()

    def _reindex(self) -> None:
        self._position = {int(item): i for i, item in enumerate(self._item_ids)}
        self._normalized = _l2_normalize(self._matrix)

    def get_embedding(self, item_id: ItemId) -> np.ndarray | None:
        row = self._position.get(int(item_id))
        return None if row is None else self._matrix[row].copy()

    def upsert_embedding(self, item_id: ItemId, vector: np.ndarray) -> None:
        candidate = np.asarray(vector, dtype=np.float32).reshape(-1)
        with self._lock:
            if self._matrix.size and candidate.shape[0] != self._matrix.shape[1]:
                raise ValueError(
                    f"embedding dimension mismatch: got {candidate.shape[0]}, "
                    f"store holds {self._matrix.shape[1]}"
                )
            row = self._position.get(int(item_id))
            if row is not None:
                self._matrix[row] = candidate
            elif self._matrix.size:
                self._item_ids = np.append(self._item_ids, np.int64(item_id))
                self._matrix = np.vstack([self._matrix, candidate[None, :]])
            else:
                self._item_ids = np.asarray([int(item_id)], dtype=np.int64)
                self._matrix = candidate[None, :].copy()
            self._reindex()

    def similarity_search(self, vector: np.ndarray, k: int = 10) -> list[tuple[ItemId, float]]:
        if self._normalized.size == 0 or k <= 0:
            return []
        query = np.asarray(vector, dtype=np.float32).reshape(-1)
        if query.shape[0] != self._normalized.shape[1]:
            raise ValueError(
                f"query dimension {query.shape[0]} does not match store dimension "
                f"{self._normalized.shape[1]}"
            )
        norm = float(np.linalg.norm(query))
        if norm > 0:
            query = query / norm

        scores = self._normalized @ query
        k = min(k, scores.size)
        partition = np.argpartition(-scores, k - 1)[:k]
        ordered = partition[np.argsort(-scores[partition])]
        return [(int(self._item_ids[i]), float(scores[i])) for i in ordered]

    def count(self) -> int:
        return int(self._item_ids.size)

    def flush(self) -> Path | None:
        """Persist the store to its archive, when one is configured."""
        if self._path is None:
            return None
        self._path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self._path, item_ids=self._item_ids, matrix=self._matrix)
        return self._path

    def _load(self) -> None:
        assert self._path is not None
        with np.load(self._path) as archive:
            self._item_ids = archive["item_ids"].astype(np.int64)
            self._matrix = archive["matrix"].astype(np.float32)
        self._reindex()


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


__all__ = ["VectorEmbeddingRepository"]
