"""Contract of the AI layer (ai/interfaces.py)."""

from __future__ import annotations

import numpy as np
import pytest

from ai.interfaces import IRecommendationStrategy, ScoredItem, UserFeatures, top_k


def test_the_interface_cannot_be_instantiated() -> None:
    """An incomplete implementation must fail, not silently return None."""
    with pytest.raises(TypeError):
        IRecommendationStrategy()  # type: ignore[abstract]


def test_an_implementation_missing_predict_is_rejected() -> None:
    class Incomplete(IRecommendationStrategy):
        pass

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]


def test_a_complete_implementation_is_accepted() -> None:
    class Complete(IRecommendationStrategy):
        def predict(self, user_features: UserFeatures, k: int = 10) -> list[ScoredItem]:
            return []

    assert Complete().predict(UserFeatures(user_id=1)) == []


def test_user_features_defaults_are_a_cold_start_shaped_object() -> None:
    features = UserFeatures(user_id=7)
    assert features.latent_factors is None
    assert features.content_profile is None
    assert features.seen_items == ()
    assert features.statistics == {}


def test_top_k_returns_the_highest_scores_in_order() -> None:
    scores = np.array([0.1, 0.9, 0.5, 0.7, 0.3], dtype=np.float32)
    result = top_k(scores, [10, 11, 12, 13, 14], 3)
    assert [item.item_id for item in result] == [11, 13, 12]
    assert [round(item.score, 3) for item in result] == [0.9, 0.7, 0.5]


def test_top_k_is_bounded_by_the_catalog_size() -> None:
    scores = np.array([0.4, 0.2], dtype=np.float32)
    assert len(top_k(scores, [1, 2], 50)) == 2


@pytest.mark.parametrize("k", [0, -1])
def test_top_k_of_a_non_positive_k_is_empty(k: int) -> None:
    assert top_k(np.array([1.0, 2.0]), [1, 2], k) == []


def test_top_k_of_an_empty_catalog_is_empty() -> None:
    assert top_k(np.array([], dtype=np.float32), [], 5) == []


def test_top_k_agrees_with_a_full_sort() -> None:
    """The argpartition optimisation must not change the result."""
    rng = np.random.default_rng(0)
    scores = rng.random(500).astype(np.float32)
    item_ids = list(range(1000, 1500))
    fast = [item.item_id for item in top_k(scores, item_ids, 20)]
    slow = [item_ids[i] for i in np.argsort(-scores)[:20]]
    assert fast == slow


def test_describe_reports_the_identity_used_by_the_logs() -> None:
    class Named(IRecommendationStrategy):
        strategy_id = "named"

        def __init__(self) -> None:
            self._model_version = "v9"

        def predict(self, user_features: UserFeatures, k: int = 10) -> list[ScoredItem]:
            return []

    assert Named().describe() == {"strategy_id": "named", "model_version": "v9"}
