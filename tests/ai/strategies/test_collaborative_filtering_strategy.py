"""CollaborativeFilteringStrategy - ALS latent factors (FR-04)."""

from __future__ import annotations

import numpy as np
import pytest

from ai.interfaces import UserFeatures
from ai.strategies.collaborative_filtering_strategy import CollaborativeFilteringStrategy
from config.settings import Settings
from errors import ColdStartError, ModelUnavailableError
from tests.conftest import DIMENSION


@pytest.fixture
def strategy(settings: Settings) -> CollaborativeFilteringStrategy:
    return CollaborativeFilteringStrategy.build(settings)


def test_it_scores_with_the_dot_product_of_the_latent_spaces(
    strategy: CollaborativeFilteringStrategy, user_features: UserFeatures
) -> None:
    result = strategy.predict(user_features, k=3)
    assert len(result) == 3
    assert [item.score for item in result] == sorted((i.score for i in result), reverse=True)


def test_a_user_without_latent_factors_is_a_cold_start(
    strategy: CollaborativeFilteringStrategy, cold_features: UserFeatures
) -> None:
    """The fallback decision belongs to the caller, so the strategy signals it."""
    with pytest.raises(ColdStartError) as excinfo:
        strategy.predict(cold_features, k=3)
    assert excinfo.value.degradable is True
    assert excinfo.value.status_code == 200


def test_an_incompatible_user_vector_is_a_model_error(
    strategy: CollaborativeFilteringStrategy,
) -> None:
    wrong = UserFeatures(user_id=1, latent_factors=np.ones(DIMENSION + 3, dtype=np.float32))
    with pytest.raises(ModelUnavailableError, match="incompatible dimensions"):
        strategy.predict(wrong, k=3)


def test_seen_items_are_excluded(strategy: CollaborativeFilteringStrategy) -> None:
    features = UserFeatures(user_id=1, latent_factors=np.ones(DIMENSION, dtype=np.float32))
    full = strategy.predict(features, k=4)
    seen = tuple(item.item_id for item in full[:2])
    filtered = strategy.predict(
        UserFeatures(user_id=1, latent_factors=features.latent_factors, seen_items=seen), k=4
    )
    assert not set(seen) & {item.item_id for item in filtered}


def test_the_ranking_matches_a_hand_computed_score(settings: Settings) -> None:
    """Pin the arithmetic, not only the shape of the result."""
    item_ids = np.array([10, 11, 12], dtype=np.int64)
    factors = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
    strategy = CollaborativeFilteringStrategy(item_ids, factors, model_version="v-test")

    user = UserFeatures(user_id=1, latent_factors=np.array([1.0, 0.2], dtype=np.float32))
    result = strategy.predict(user, k=3)
    assert [item.item_id for item in result] == [10, 12, 11]
    assert pytest.approx([1.0, 0.6, 0.2], abs=1e-6) == [item.score for item in result]


def test_mismatched_artifacts_are_rejected_at_construction() -> None:
    with pytest.raises(ModelUnavailableError, match="different lengths"):
        CollaborativeFilteringStrategy(
            np.array([1, 2, 3], dtype=np.int64), np.zeros((2, 4), dtype=np.float32)
        )


def test_a_missing_artifact_is_reported_as_unavailable(settings: Settings) -> None:
    (settings.feature_store.version_dir / "item_features.parquet").unlink()
    with pytest.raises(ModelUnavailableError):
        CollaborativeFilteringStrategy.build(settings)


def test_it_registers_itself_under_the_documented_identifier() -> None:
    assert CollaborativeFilteringStrategy.strategy_id == "als"


def test_the_strategy_never_touches_a_repository() -> None:
    """Section 9.5: no strategy accesses the database directly."""
    import inspect

    source = inspect.getsource(CollaborativeFilteringStrategy)
    assert "Repository" not in source
