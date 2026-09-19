"""ContentBasedStrategy - cosine similarity over content embeddings (FR-04)."""

from __future__ import annotations

import numpy as np
import pytest

from ai.interfaces import UserFeatures
from ai.strategies.content_based_strategy import ContentBasedStrategy
from config.settings import Settings
from errors import ColdStartError, ModelUnavailableError
from tests.conftest import DIMENSION


@pytest.fixture
def strategy(settings: Settings) -> ContentBasedStrategy:
    return ContentBasedStrategy.build(settings)


def test_it_ranks_by_similarity_to_the_user_profile(
    strategy: ContentBasedStrategy, user_features: UserFeatures
) -> None:
    result = strategy.predict(user_features, k=3)
    assert len(result) == 3
    assert all(-1.0001 <= item.score <= 1.0001 for item in result)


def test_a_user_without_a_profile_is_a_cold_start(
    strategy: ContentBasedStrategy, cold_features: UserFeatures
) -> None:
    with pytest.raises(ColdStartError):
        strategy.predict(cold_features, k=3)


def test_an_all_zero_profile_is_also_a_cold_start(strategy: ContentBasedStrategy) -> None:
    """A zero vector carries no preference; a cosine against it is meaningless."""
    empty = UserFeatures(user_id=1, content_profile=np.zeros(DIMENSION, dtype=np.float32))
    with pytest.raises(ColdStartError):
        strategy.predict(empty, k=3)


def test_the_most_similar_item_wins() -> None:
    item_ids = np.array([10, 11, 12], dtype=np.int64)
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]], dtype=np.float32)
    strategy = ContentBasedStrategy(item_ids, embeddings)

    user = UserFeatures(user_id=1, content_profile=np.array([1.0, 0.0], dtype=np.float32))
    result = strategy.predict(user, k=3)
    assert [item.item_id for item in result] == [10, 12, 11]
    assert result[0].score == pytest.approx(1.0, abs=1e-6)


def test_the_score_is_scale_invariant() -> None:
    """Cosine similarity must ignore the magnitude of the user profile."""
    item_ids = np.array([10, 11], dtype=np.int64)
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    strategy = ContentBasedStrategy(item_ids, embeddings)

    small = strategy.predict(
        UserFeatures(user_id=1, content_profile=np.array([1.0, 0.5], dtype=np.float32))
    )
    large = strategy.predict(
        UserFeatures(user_id=1, content_profile=np.array([100.0, 50.0], dtype=np.float32))
    )
    assert [i.item_id for i in small] == [i.item_id for i in large]
    assert small[0].score == pytest.approx(large[0].score, abs=1e-5)


def test_an_incompatible_profile_is_a_model_error(strategy: ContentBasedStrategy) -> None:
    wrong = UserFeatures(user_id=1, content_profile=np.ones(DIMENSION + 2, dtype=np.float32))
    with pytest.raises(ModelUnavailableError):
        strategy.predict(wrong, k=3)


def test_seen_items_are_excluded(
    strategy: ContentBasedStrategy, user_features: UserFeatures
) -> None:
    full = strategy.predict(
        UserFeatures(user_id=1, content_profile=user_features.content_profile), k=4
    )
    seen = tuple(item.item_id for item in full[:2])
    filtered = strategy.predict(
        UserFeatures(user_id=1, content_profile=user_features.content_profile, seen_items=seen), k=4
    )
    assert not set(seen) & {item.item_id for item in filtered}


def test_it_registers_itself_under_the_documented_identifier() -> None:
    assert ContentBasedStrategy.strategy_id == "content"
