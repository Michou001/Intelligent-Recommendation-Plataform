"""PopularityStrategy - cold-start fallback (FR-05, Table 9).

Ranking by global popularity, optionally restricted to a content category.
Requires no user history, which is precisely why it is the strategy the
system falls back to when there is nothing to personalise with, when the
feature store cannot be read, or when the configured model is unavailable
(Table 10).

It is an ordinary implementation of ``IRecommendationStrategy``: the fallback
is not a special case bolted onto the controller, it is one more strategy
resolved through the same registry (Section 4.2).
"""

from __future__ import annotations

import numpy as np

from ai.interfaces import IRecommendationStrategy, ItemId, ScoredItem, UserFeatures, top_k
from ai.model_factory import register_strategy
from ai.strategies import load_feature_table
from config.settings import Settings, load_settings
from errors import ModelUnavailableError


@register_strategy("popularity")
class PopularityStrategy(IRecommendationStrategy):
    """Serves the precomputed popularity ranking materialised offline.

    The ranking is a count-weighted mean rating computed by
    ``data/pipeline/transform.py``; this class only selects from it, so the
    cold-start path costs a slice and never a computation.
    """

    def __init__(
        self,
        item_ids: np.ndarray,
        scores: np.ndarray,
        *,
        category_rankings: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
        model_version: str = "unknown",
    ) -> None:
        self._item_ids = np.asarray(item_ids, dtype=np.int64)
        self._scores = np.asarray(scores, dtype=np.float32)
        self._category_rankings = category_rankings or {}
        self._model_version = model_version
        self._position = {int(item): i for i, item in enumerate(self._item_ids)}

    @classmethod
    def build(cls, settings: Settings | None = None) -> PopularityStrategy:
        """Load the global and per-category rankings from the feature store."""
        settings = settings or load_settings()
        store_dir = settings.feature_store.version_dir

        popularity = load_feature_table(store_dir, "popularity")
        if popularity is None or popularity.empty:
            raise ModelUnavailableError("the popularity ranking is empty")

        popularity = popularity.sort_values("popularity_score", ascending=False)
        item_ids = popularity["item_id"].to_numpy(dtype=np.int64)
        scores = popularity["popularity_score"].to_numpy(dtype=np.float32)

        category_rankings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        by_category = load_feature_table(store_dir, "category_popularity", required=False)
        if by_category is not None and not by_category.empty:
            for category, group in by_category.groupby("category"):
                ordered = group.sort_values("popularity_score", ascending=False)
                category_rankings[str(category)] = (
                    ordered["item_id"].to_numpy(dtype=np.int64),
                    ordered["popularity_score"].to_numpy(dtype=np.float32),
                )

        return cls(
            item_ids,
            scores,
            category_rankings=category_rankings,
            model_version=settings.feature_store.version,
        )

    def predict(self, user_features: UserFeatures, k: int = 10) -> list[ScoredItem]:
        """Return the K most popular items the user has not seen (FR-05).

        This method never raises ``ColdStartError``: it is the answer to the
        cold start, so it must always be able to produce a ranking.
        """
        item_ids, scores = self._select_ranking(user_features.preferred_category)
        if item_ids.size == 0:
            raise ModelUnavailableError("no item is available to rank")

        scores = self._mask_seen(item_ids, scores, user_features.seen_items)
        return top_k(scores, item_ids, k)

    def _select_ranking(self, category: str | None) -> tuple[np.ndarray, np.ndarray]:
        """Use the category ranking when one was materialised for it."""
        if category and category in self._category_rankings:
            return self._category_rankings[category]
        return self._item_ids, self._scores

    @staticmethod
    def _mask_seen(
        item_ids: np.ndarray, scores: np.ndarray, seen_items: tuple[ItemId, ...]
    ) -> np.ndarray:
        """Push already-seen items out of the ranking without reordering it."""
        if not seen_items:
            return scores
        masked = scores.copy()
        seen = np.fromiter((int(i) for i in seen_items), dtype=np.int64, count=len(seen_items))
        masked[np.isin(item_ids, seen)] = -np.inf
        return masked

    def __len__(self) -> int:
        return int(self._item_ids.size)


__all__ = ["PopularityStrategy"]
