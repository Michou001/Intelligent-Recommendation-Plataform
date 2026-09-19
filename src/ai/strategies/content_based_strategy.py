"""ContentBasedStrategy - cosine similarity over content embeddings (FR-04, Table 9).

Scores each item as the cosine similarity between its metadata embedding and
the aggregated profile of the user's history, both materialised offline by
``data/pipeline/transform.py``.

Normalising the item matrix once, at load time, turns the per-request cosine
similarity into a single matrix-vector product: nothing is normalised again
while serving a request.
"""

from __future__ import annotations

import numpy as np

from ai.interfaces import IRecommendationStrategy, ItemId, ScoredItem, UserFeatures, top_k
from ai.model_factory import register_strategy
from ai.strategies import load_feature_table
from config.settings import Settings, load_settings
from errors import ColdStartError, ModelUnavailableError


@register_strategy("content")
class ContentBasedStrategy(IRecommendationStrategy):
    """Content-based ranking over precomputed item embeddings."""

    def __init__(
        self,
        item_ids: np.ndarray,
        item_embeddings: np.ndarray,
        *,
        model_version: str = "unknown",
    ) -> None:
        self._item_ids = np.asarray(item_ids, dtype=np.int64)
        self._embeddings = _l2_normalize(np.asarray(item_embeddings, dtype=np.float32))
        self._model_version = model_version
        if self._embeddings.shape[0] != self._item_ids.shape[0]:
            raise ModelUnavailableError(
                "item embeddings and item identifiers have different lengths",
                detail=f"{self._embeddings.shape[0]} embeddings vs {self._item_ids.shape[0]} ids",
            )

    @classmethod
    def build(cls, settings: Settings | None = None) -> ContentBasedStrategy:
        """Load the item content embeddings of the configured store version."""
        settings = settings or load_settings()
        frame = load_feature_table(settings.feature_store.version_dir, "item_features")
        if frame is None or frame.empty:
            raise ModelUnavailableError("the item feature table is empty")
        if "content_embedding" not in frame.columns:
            raise ModelUnavailableError("the item feature table carries no content embeddings")

        item_ids = frame["item_id"].to_numpy(dtype=np.int64)
        embeddings = np.vstack(
            [np.asarray(row, dtype=np.float32) for row in frame["content_embedding"]]
        )
        return cls(item_ids, embeddings, model_version=settings.feature_store.version)

    @property
    def dimension(self) -> int:
        """Embedding dimension the strategy expects in a user profile."""
        return int(self._embeddings.shape[1])

    def predict(self, user_features: UserFeatures, k: int = 10) -> list[ScoredItem]:
        """Return the K items closest to the user's content profile (FR-04)."""
        profile = user_features.content_profile
        if profile is None or profile.size == 0 or not np.any(profile):
            raise ColdStartError(
                f"user {user_features.user_id} has no content profile",
                detail="content-based ranking needs at least one positively rated item",
            )
        if profile.shape[0] != self.dimension:
            raise ModelUnavailableError(
                "user profile and item embeddings have incompatible dimensions",
                detail=f"profile={profile.shape[0]} items={self.dimension}",
            )

        query = np.asarray(profile, dtype=np.float32)
        norm = float(np.linalg.norm(query))
        if norm > 0:
            query = query / norm

        scores = self._embeddings @ query
        scores = _mask_seen(self._item_ids, scores, user_features.seen_items)
        return top_k(scores, self._item_ids, k)

    def __len__(self) -> int:
        return int(self._item_ids.size)


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Normalise rows once, so a cosine similarity is a plain dot product."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


def _mask_seen(
    item_ids: np.ndarray, scores: np.ndarray, seen_items: tuple[ItemId, ...]
) -> np.ndarray:
    """Exclude the items the user already interacted with."""
    if not seen_items:
        return scores
    masked = scores.copy()
    seen = np.fromiter((int(i) for i in seen_items), dtype=np.int64, count=len(seen_items))
    masked[np.isin(item_ids, seen)] = -np.inf
    return masked


__all__ = ["ContentBasedStrategy"]
