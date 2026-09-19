"""Abstractions of the AI & Recommendation Engine Layer (Table 11, FR-04).

This module is one of the two contract modules named in Section 9.2. Every
outer layer that needs a recommendation depends on ``IRecommendationStrategy``
declared here, never on a concrete strategy class, which is the Dependency
Inversion Principle of Section 3.2 expressed in code.

The interface is an abstract base class, so an incomplete implementation
fails when it is instantiated rather than when it is first called
(Section 9.2, programming practices).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from config.settings import Settings

#: Identifier of a catalog item. MovieLens uses integer ``movieId`` values.
ItemId = int
#: Identifier of a user. MovieLens uses integer ``userId`` values.
UserId = int


@dataclass(frozen=True)
class ScoredItem:
    """One entry of a Top-K ranking: the item and its relevance score."""

    item_id: ItemId
    score: float


@dataclass(frozen=True)
class UserFeatures:
    """Precomputed feature vector served by the Feature Store (FR-03).

    Strategies receive this object and never read storage themselves. Every
    field is materialised offline by the pipeline of Table 8; nothing here is
    computed at inference time (Section 4.2).

    Attributes:
        user_id: Identifier the features belong to.
        latent_factors: User latent vector of the implicit-feedback
            factorisation, consumed by ``CollaborativeFilteringStrategy``.
            ``None`` when the user has no factorised history.
        content_profile: Aggregated embedding of the items in the user's
            history, consumed by ``ContentBasedStrategy``. ``None`` when the
            user has no history.
        statistics: Aggregated user statistics (interaction count, mean
            rating, rating variance), consumed by ``DeepLearningStrategy``.
        seen_items: Items already interacted with, excluded from the ranking.
        preferred_category: Dominant content category of the user's history,
            used by ``PopularityStrategy`` to restrict the cold-start ranking.
        is_cold_start: True when the user has no usable history and the
            cold-start path of FR-05 must be taken.
        feature_store_version: Version tag of the materialised store the
            vectors were read from; it is part of the cache key.
    """

    user_id: UserId
    latent_factors: np.ndarray | None = None
    content_profile: np.ndarray | None = None
    statistics: Mapping[str, float] = field(default_factory=dict)
    seen_items: tuple[ItemId, ...] = ()
    preferred_category: str | None = None
    is_cold_start: bool = False
    feature_store_version: str = "unknown"


class IRecommendationStrategy(ABC):
    """Contract every recommendation algorithm realises (Strategy, GoF 1994).

    The controller and the decorators depend only on this abstraction, so a
    new algorithm is added by writing a new implementation, without editing
    any client code (Open/Closed Principle, Section 3.3).
    """

    #: Registry identifier the strategy is resolved by (Section 9.5).
    strategy_id: str = "abstract"

    @abstractmethod
    def predict(self, user_features: UserFeatures, k: int = 10) -> list[ScoredItem]:
        """Return the K items with the highest relevance score (FR-04).

        Args:
            user_features: Precomputed vectors for the requesting user.
            k: Size of the ranking to return. The document writes the contract
                as ``predict(user_features)``; ``k`` is carried as a defaulted
                argument because FR-04 and the endpoint of Section 9.6 make
                the ranking size a per-request value.

        Returns:
            At most ``k`` scored items, sorted by descending score.

        Raises:
            ColdStartError: The features carry no usable history for this
                algorithm, so the caller must fall back (FR-05).
            ModelUnavailableError: The model weights cannot be used and the
                caller must degrade the response (NFR-05).
        """

    @classmethod
    def build(cls, settings: Settings) -> IRecommendationStrategy:
        """Construct the strategy with its offline artifacts loaded.

        ``ModelFactory`` calls this hook after resolving the class through the
        registry, which is what keeps the factory free of any per-strategy
        branch (Section 3.3). The default builds a strategy that needs no
        artifacts; implementations that load weights override it.
        """
        return cls()

    @property
    def model_version(self) -> str:
        """Version tag of the loaded weights, used in the cache key and logs."""
        return getattr(self, "_model_version", "unknown")

    def describe(self) -> Mapping[str, str]:
        """Identification emitted by the operational logging decorators."""
        return {"strategy_id": self.strategy_id, "model_version": self.model_version}


def top_k(
    scores: np.ndarray, item_ids: Sequence[ItemId] | np.ndarray, k: int
) -> list[ScoredItem]:
    """Select the K highest scores without sorting the whole catalog.

    ``numpy.argpartition`` is linear in the number of items, against the
    ``n log n`` of a full sort; only the K selected entries are then ordered
    (Section 9.7, third efficiency technique).
    """
    if k <= 0 or scores.size == 0:
        return []
    k = min(k, scores.size)
    partition = np.argpartition(-scores, k - 1)[:k]
    ordered = partition[np.argsort(-scores[partition])]
    return [ScoredItem(item_id=int(item_ids[i]), score=float(scores[i])) for i in ordered]


__all__ = [
    "IRecommendationStrategy",
    "ItemId",
    "ScoredItem",
    "UserFeatures",
    "UserId",
    "top_k",
]
