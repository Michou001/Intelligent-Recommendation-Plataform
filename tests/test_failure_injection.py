"""Failure injection over the composed HTTP service (Section 9.9, NFR-05).

    "The test suite includes failure-injection cases in which the feature
     store, Redis, the message broker, and the model loader are replaced by
     doubles that raise exceptions, and the expected result in each case is a
     degraded but successful response instead of an abrupt termination."

Each test below breaks one dependency in the fully composed application and
asserts the same two things: the status code is not 500, and a ranking is
still returned. The last test breaks all four at once.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from ai.decorators.caching_decorator import CacheBackend, CachingDecorator
from application.api import create_app
from config.settings import Settings, load_settings
from data.feature_store_manager import FeatureStoreManager
from errors import EventPublishError, FeatureStoreUnavailableError
from messaging.interaction_event_queue import InteractionEvent, InteractionEventQueue
from persistence.interfaces import CatalogItem, UserProfile
from persistence.sql_catalog_repository import SqlCatalogRepository
from persistence.sql_user_profile_repository import SqlUserProfileRepository
from tests.conftest import N_ITEMS, N_USERS

# --------------------------------------------------------------------------
# Doubles
# --------------------------------------------------------------------------


class DeadFeatureStore(FeatureStoreManager):
    """The feature store cannot be read at all."""

    def get_user_features(self, user_id: int):  # noqa: ANN201
        raise FeatureStoreUnavailableError("injected outage")

    def is_available(self) -> bool:
        return False

    @property
    def version(self) -> str:
        return "v-dead"


class DeadCache(CacheBackend):
    """Redis is down: every operation raises."""

    def get(self, key: str) -> str | None:
        raise ConnectionError("redis connection refused")

    def set(self, key: str, value: str, ttl_seconds: int) -> None:
        raise ConnectionError("redis connection refused")

    def describe(self) -> str:
        return "dead-redis"


class DeadQueue(InteractionEventQueue):
    """The broker refuses every publication."""

    def publish(self, event: InteractionEvent) -> None:
        raise EventPublishError("broker connection refused")

    def consume(self, handler, *, max_events=None, timeout=None) -> int:  # noqa: ANN001
        raise ConnectionError("broker connection refused")

    def describe(self) -> str:
        return "dead-broker"


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def seeded(settings: Settings) -> Settings:
    profiles = SqlUserProfileRepository.from_settings(settings)
    for user_id in range(1, N_USERS + 1):
        profiles.save(UserProfile(user_id=user_id))
    SqlCatalogRepository.from_settings(settings).save_many(
        [CatalogItem(item_id=101 + i, title=f"Item {101 + i}") for i in range(N_ITEMS)]
    )
    return settings


def _client(settings: Settings, **broken: object) -> Iterator[TestClient]:
    """Compose the app, then swap the named dependencies for broken doubles."""
    app = create_app(settings)
    components = app.state.components
    controller = components.controller

    if "feature_store" in broken:
        controller._feature_store = broken["feature_store"]  # noqa: SLF001
    if "cache" in broken:
        chain = controller.strategy
        while chain is not None and not isinstance(chain, CachingDecorator):
            chain = getattr(chain, "wrapped", None)
        assert chain is not None, "no CachingDecorator in the chain"
        chain._backend = broken["cache"]  # noqa: SLF001
    if "queue" in broken:
        controller._event_queue = broken["queue"]  # noqa: SLF001
        # The ingestion service holds its own reference to the broker, so the
        # POST path must be broken too, not only the GET path.
        components.ingestion._event_queue = broken["queue"]  # noqa: SLF001
    if "strategy" in broken:
        controller._strategy = broken["strategy"]  # noqa: SLF001

    with TestClient(app) as client:
        yield client


# --------------------------------------------------------------------------
# One dependency at a time
# --------------------------------------------------------------------------


def test_a_dead_feature_store_returns_a_degraded_ranking(seeded: Settings) -> None:
    for client in _client(seeded, feature_store=DeadFeatureStore()):
        response = client.get("/recommendations/1?k=5")
        assert response.status_code == 200
        body = response.json()
        assert body["degraded"] is True
        assert body["degradation_reason"] == "feature_store_unavailable"
        assert len(body["items"]) == 5


def test_a_dead_cache_still_serves_the_real_inference(seeded: Settings) -> None:
    for client in _client(seeded, cache=DeadCache()):
        first = client.get("/recommendations/1?k=5")
        second = client.get("/recommendations/1?k=5")
        assert first.status_code == second.status_code == 200
        assert first.json()["degraded"] is False
        assert first.json()["items"] == second.json()["items"]


def test_a_dead_broker_does_not_affect_the_response(seeded: Settings) -> None:
    for client in _client(seeded, queue=DeadQueue()):
        response = client.get("/recommendations/1?k=5")
        assert response.status_code == 200
        assert response.json()["degraded"] is False
        assert len(response.json()["items"]) == 5


def test_a_dead_broker_still_accepts_an_interaction(seeded: Settings) -> None:
    for client in _client(seeded, queue=DeadQueue()):
        response = client.post(
            "/interactions", json={"user_id": 1, "event_type": "click", "item_id": 101}
        )
        assert response.status_code == 202
        assert response.json()["queued"] is False


def test_a_dead_model_returns_a_degraded_ranking(seeded: Settings) -> None:
    from tests.conftest import FailingStrategy

    for client in _client(seeded, strategy=FailingStrategy()):
        response = client.get("/recommendations/1?k=5")
        assert response.status_code == 200
        body = response.json()
        assert body["degraded"] is True
        assert body["degradation_reason"] == "model_unavailable"
        assert len(body["items"]) == 5


def test_a_model_that_crashes_unexpectedly_also_degrades(seeded: Settings) -> None:
    from tests.conftest import FailingStrategy

    for client in _client(seeded, strategy=FailingStrategy(RuntimeError("torch exploded"))):
        response = client.get("/recommendations/1?k=5")
        assert response.status_code == 200
        assert response.json()["degraded"] is True


# --------------------------------------------------------------------------
# Everything at once
# --------------------------------------------------------------------------


def test_all_four_dependencies_down_is_still_not_a_500(seeded: Settings) -> None:
    """The strongest form of NFR-05: nothing left but the fallback."""
    from tests.conftest import FailingStrategy

    for client in _client(
        seeded,
        feature_store=DeadFeatureStore(),
        cache=DeadCache(),
        queue=DeadQueue(),
        strategy=FailingStrategy(),
    ):
        response = client.get("/recommendations/1?k=5")
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["degraded"] is True
        assert len(body["items"]) == 5
        assert body["strategy_id"] == "popularity"


@pytest.mark.parametrize("user_id", [1, 2, 3])
def test_no_injected_failure_ever_produces_a_500(seeded: Settings, user_id: int) -> None:
    """Sweep the failure combinations and assert the absence of a 500."""
    from tests.conftest import FailingStrategy

    injections = [
        {"feature_store": DeadFeatureStore()},
        {"cache": DeadCache()},
        {"queue": DeadQueue()},
        {"strategy": FailingStrategy()},
        {"feature_store": DeadFeatureStore(), "strategy": FailingStrategy()},
        {"cache": DeadCache(), "queue": DeadQueue()},
    ]
    for injection in injections:
        for client in _client(load_settings(), **injection):
            response = client.get(f"/recommendations/{user_id}?k=3")
            assert response.status_code < 500, f"{injection}: {response.status_code}"


def test_the_service_starts_with_no_feature_store_at_all(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A clean machine with no materialised artifacts must not crash on boot."""
    monkeypatch.setenv("RECO_FEATURE_STORE_VERSION", "version-that-does-not-exist")
    monkeypatch.setenv("RECO_ACTIVE_STRATEGY", "als")
    fresh = load_settings()

    with TestClient(create_app(fresh)) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["feature_store_available"] is False
        # No model and no fallback can load, so the request cannot be served;
        # it must still be an explicit, translated error and not a crash.
        response = client.get("/recommendations/1?k=3")
        assert response.status_code in {200, 404, 503}
        assert response.status_code != 500
