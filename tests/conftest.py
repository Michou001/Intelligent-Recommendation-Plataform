"""Shared fixtures.

Every fixture builds a self-contained environment in a temporary directory:
a small synthetic feature store, a SQLite file, and a local document store.
The suite therefore never depends on the MovieLens artifacts being present,
which is what lets it run on a clean checkout.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ai.interfaces import IRecommendationStrategy, ScoredItem, UserFeatures
from config.settings import Settings, load_settings
from errors import ColdStartError, ModelUnavailableError

#: Size of the synthetic store: small enough to be fast, large enough that a
#: Top-K selection is a real selection.
N_USERS = 6
N_ITEMS = 12
DIMENSION = 4

#: The user identifier that is deliberately absent from the feature store, so
#: the cold-start path can be exercised.
COLD_START_USER = 99
#: A user identifier absent from every store, for the 404 path.
UNKNOWN_USER = 4242


@pytest.fixture
def feature_store_dir(tmp_path: Path) -> Path:
    """Write a deterministic synthetic feature store and return its directory."""
    store = tmp_path / "feature_store" / "v-test"
    store.mkdir(parents=True)
    rng = np.random.default_rng(1234)

    user_ids = list(range(1, N_USERS + 1))
    item_ids = list(range(101, 101 + N_ITEMS))

    users = pd.DataFrame(
        {
            "user_id": user_ids,
            "latent_factors": [
                rng.normal(size=DIMENSION).astype(np.float32).tolist() for _ in user_ids
            ],
            "content_profile": [
                rng.random(DIMENSION).astype(np.float32).tolist() for _ in user_ids
            ],
            "n_interactions": [10 + i for i in range(N_USERS)],
            "mean_rating": [3.5 + 0.1 * i for i in range(N_USERS)],
            "rating_std": [0.5] * N_USERS,
            "min_rating": [1.0] * N_USERS,
            "max_rating": [5.0] * N_USERS,
            # The first user has already seen the two first items, so the
            # exclusion of seen items is exercised by default.
            "seen_items": [[101, 102] if u == 1 else [] for u in user_ids],
            "preferred_category": ["Drama" if u % 2 else "Comedy" for u in user_ids],
        }
    )

    items = pd.DataFrame(
        {
            "item_id": item_ids,
            "latent_factors": [
                rng.normal(size=DIMENSION).astype(np.float32).tolist() for _ in item_ids
            ],
            "content_embedding": [
                rng.random(DIMENSION).astype(np.float32).tolist() for _ in item_ids
            ],
            "popularity_score": [float(N_ITEMS - i) for i in range(N_ITEMS)],
            "interaction_count": [100 - i for i in range(N_ITEMS)],
            "rank": list(range(1, N_ITEMS + 1)),
            "title": [f"Item {i}" for i in item_ids],
            "categories": [["Drama"] if i % 2 else ["Comedy"] for i in item_ids],
        }
    )

    popularity = items[["item_id", "popularity_score", "interaction_count", "rank"]].copy()
    popularity["mean_rating"] = 4.0
    popularity["title"] = items["title"]
    popularity["categories"] = items["categories"]

    category_popularity = (
        popularity.explode("categories")
        .rename(columns={"categories": "category"})
        .assign(category_rank=lambda f: f.groupby("category").cumcount() + 1)[
            ["category", "item_id", "popularity_score", "interaction_count", "category_rank"]
        ]
    )

    users.to_parquet(store / "user_features.parquet", index=False)
    items.to_parquet(store / "item_features.parquet", index=False)
    popularity.to_parquet(store / "popularity.parquet", index=False)
    category_popularity.to_parquet(store / "category_popularity.parquet", index=False)
    (store / "_metadata.json").write_text(
        json.dumps(
            {
                "version": "v-test",
                "dataset_variant": "synthetic",
                "latent_dimension": DIMENSION,
                "embedding_dimension": DIMENSION,
            }
        ),
        encoding="utf-8",
    )
    return store


@pytest.fixture
def settings(tmp_path: Path, feature_store_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Settings pointing every store at the temporary directory."""
    monkeypatch.setenv("RECO_FEATURE_STORE_ROOT", str(feature_store_dir.parent))
    monkeypatch.setenv("RECO_FEATURE_STORE_VERSION", feature_store_dir.name)
    monkeypatch.setenv("RECO_SQL_URL", f"sqlite:///{(tmp_path / 'platform.db').as_posix()}")
    monkeypatch.setenv("RECO_DOCUMENT_STORE_ROOT", str(tmp_path / "documents"))
    monkeypatch.setenv("RECO_RETRY_BUFFER_PATH", str(tmp_path / "retry_buffer.jsonl"))
    monkeypatch.setenv("RECO_WEIGHTS_ROOT", str(tmp_path / "models"))
    monkeypatch.setenv("RECO_INTERIM_ROOT", str(tmp_path / "interim"))
    monkeypatch.setenv("RECO_DATASET_ROOT", str(tmp_path / "datasets"))
    monkeypatch.setenv("RECO_REDIS_URL", "")
    monkeypatch.setenv("RECO_RABBITMQ_URL", "")
    monkeypatch.setenv("RECO_MONGO_URL", "")
    monkeypatch.setenv("RECO_LOG_LEVEL", "CRITICAL")
    return load_settings()


@pytest.fixture(autouse=True)
def _reset_factory_cache() -> Iterator[None]:
    """Strategies cache their loaded artifacts; a test must not inherit them."""
    from ai.model_factory import ModelFactory

    ModelFactory.clear_cache()
    yield
    ModelFactory.clear_cache()


class StubStrategy(IRecommendationStrategy):
    """Predictable strategy used to test the clients of the interface."""

    strategy_id = "stub"

    def __init__(self, items: list[int] | None = None, *, model_version: str = "stub-v1") -> None:
        self._items = items if items is not None else [101, 102, 103]
        self._model_version = model_version
        self.calls = 0

    def predict(self, user_features: UserFeatures, k: int = 10) -> list[ScoredItem]:
        self.calls += 1
        return [
            ScoredItem(item_id=item, score=1.0 - 0.1 * index)
            for index, item in enumerate(self._items[:k])
        ]


class FailingStrategy(IRecommendationStrategy):
    """Strategy that always fails, for the degradation tests."""

    strategy_id = "failing"

    def __init__(self, error: Exception | None = None) -> None:
        self._error = error or ModelUnavailableError("injected model failure")
        self.calls = 0

    def predict(self, user_features: UserFeatures, k: int = 10) -> list[ScoredItem]:
        self.calls += 1
        raise self._error


class ColdStartStrategy(IRecommendationStrategy):
    """Strategy that always reports a cold start."""

    strategy_id = "cold"

    def predict(self, user_features: UserFeatures, k: int = 10) -> list[ScoredItem]:
        raise ColdStartError("injected cold start")


@pytest.fixture
def stub_strategy() -> StubStrategy:
    return StubStrategy()


@pytest.fixture
def user_features() -> UserFeatures:
    """Features of a user with a complete history."""
    return UserFeatures(
        user_id=1,
        latent_factors=np.ones(DIMENSION, dtype=np.float32),
        content_profile=np.ones(DIMENSION, dtype=np.float32),
        statistics={"n_interactions": 10.0, "mean_rating": 4.0, "rating_std": 0.5},
        seen_items=(101, 102),
        preferred_category="Drama",
        feature_store_version="v-test",
    )


@pytest.fixture
def cold_features() -> UserFeatures:
    """Features of a user with nothing to personalise with."""
    return UserFeatures(user_id=COLD_START_USER, is_cold_start=True, feature_store_version="v-test")
