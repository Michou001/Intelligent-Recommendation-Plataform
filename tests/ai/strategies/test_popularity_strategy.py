"""PopularityStrategy - the cold-start fallback (FR-05)."""

from __future__ import annotations

import numpy as np
import pytest

from ai.interfaces import UserFeatures
from ai.strategies.popularity_strategy import PopularityStrategy
from config.settings import Settings
from errors import ModelUnavailableError


@pytest.fixture
def strategy(settings: Settings) -> PopularityStrategy:
    return PopularityStrategy.build(settings)


def test_it_ranks_by_precomputed_popularity(strategy: PopularityStrategy) -> None:
    result = strategy.predict(UserFeatures(user_id=1), k=3)
    assert len(result) == 3
    assert [item.score for item in result] == sorted((i.score for i in result), reverse=True)


def test_it_serves_a_user_with_no_history_at_all(
    strategy: PopularityStrategy, cold_features: UserFeatures
) -> None:
    """The fallback must always answer: that is the whole point of FR-05."""
    result = strategy.predict(cold_features, k=5)
    assert len(result) == 5


def test_it_never_raises_cold_start(strategy: PopularityStrategy) -> None:
    empty = UserFeatures(user_id=12345, is_cold_start=True)
    assert strategy.predict(empty, k=1)


def test_it_excludes_items_the_user_already_saw(strategy: PopularityStrategy) -> None:
    everything = strategy.predict(UserFeatures(user_id=1), k=4)
    seen = tuple(item.item_id for item in everything[:2])
    filtered = strategy.predict(UserFeatures(user_id=1, seen_items=seen), k=4)
    assert not set(seen) & {item.item_id for item in filtered}


def test_it_restricts_the_ranking_to_the_preferred_category(
    strategy: PopularityStrategy, settings: Settings
) -> None:
    import pandas as pd

    result = strategy.predict(UserFeatures(user_id=1, preferred_category="Drama"), k=3)
    table = pd.read_parquet(settings.feature_store.version_dir / "category_popularity.parquet")
    drama = set(table[table["category"] == "Drama"]["item_id"])
    assert {item.item_id for item in result} <= drama


def test_an_unknown_category_falls_back_to_the_global_ranking(
    strategy: PopularityStrategy,
) -> None:
    result = strategy.predict(UserFeatures(user_id=1, preferred_category="Nonexistent"), k=3)
    assert len(result) == 3


def test_k_is_honoured(strategy: PopularityStrategy) -> None:
    for k in (1, 3, 7):
        assert len(strategy.predict(UserFeatures(user_id=1), k=k)) == k


def test_an_empty_ranking_is_reported_as_unavailable() -> None:
    empty = PopularityStrategy(np.array([], dtype=np.int64), np.array([], dtype=np.float32))
    with pytest.raises(ModelUnavailableError):
        empty.predict(UserFeatures(user_id=1), k=3)


def test_a_missing_artifact_is_reported_as_unavailable(
    settings: Settings, tmp_path: object
) -> None:
    (settings.feature_store.version_dir / "popularity.parquet").unlink()
    with pytest.raises(ModelUnavailableError) as excinfo:
        PopularityStrategy.build(settings)
    assert excinfo.value.degradable is True


def test_it_registers_itself_under_the_documented_identifier() -> None:
    assert PopularityStrategy.strategy_id == "popularity"
