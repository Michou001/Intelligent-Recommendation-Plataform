"""CachingDecorator (Table 4, NFR-01, Section 9.7)."""

from __future__ import annotations

import pytest

from ai.decorators import current_context, operational_context
from ai.decorators.caching_decorator import (
    CacheBackend,
    CachingDecorator,
    InMemoryCacheBackend,
)
from ai.interfaces import UserFeatures
from tests.conftest import StubStrategy


class BrokenCacheBackend(CacheBackend):
    """Backend that fails on every operation, for the failure-injection tests."""

    def get(self, key: str) -> str | None:
        raise ConnectionError("redis is down")

    def set(self, key: str, value: str, ttl_seconds: int) -> None:
        raise ConnectionError("redis is down")

    def describe(self) -> str:
        return "broken"


@pytest.fixture
def decorated(stub_strategy: StubStrategy) -> CachingDecorator:
    return CachingDecorator(stub_strategy, InMemoryCacheBackend(), ttl_seconds=60)


def test_the_second_identical_request_skips_the_inference(
    decorated: CachingDecorator, stub_strategy: StubStrategy, user_features: UserFeatures
) -> None:
    first = decorated.predict(user_features, 3)
    second = decorated.predict(user_features, 3)
    assert first == second
    assert stub_strategy.calls == 1


def test_a_different_k_is_a_different_key(
    decorated: CachingDecorator, stub_strategy: StubStrategy, user_features: UserFeatures
) -> None:
    decorated.predict(user_features, 2)
    decorated.predict(user_features, 3)
    assert stub_strategy.calls == 2


def test_a_different_user_is_a_different_key(
    decorated: CachingDecorator, stub_strategy: StubStrategy, user_features: UserFeatures
) -> None:
    decorated.predict(user_features, 3)
    decorated.predict(UserFeatures(user_id=2), 3)
    assert stub_strategy.calls == 2


def test_the_key_carries_the_model_version(
    decorated: CachingDecorator, user_features: UserFeatures
) -> None:
    """Section 9.7: the model version is what invalidates the cache on deploy."""
    key = decorated.cache_key(user_features, 10)
    assert key.endswith(":10:stub-v1")
    assert str(user_features.user_id) in key


def test_a_new_model_version_does_not_read_stale_entries(
    user_features: UserFeatures,
) -> None:
    backend = InMemoryCacheBackend()
    old = CachingDecorator(StubStrategy([1, 2, 3], model_version="v1"), backend, ttl_seconds=60)
    new_strategy = StubStrategy([7, 8, 9], model_version="v2")
    new = CachingDecorator(new_strategy, backend, ttl_seconds=60)

    old.predict(user_features, 3)
    result = new.predict(user_features, 3)

    assert [item.item_id for item in result] == [7, 8, 9]
    assert new_strategy.calls == 1


def test_a_broken_cache_degrades_into_a_miss(
    stub_strategy: StubStrategy, user_features: UserFeatures
) -> None:
    """Failure injection: Redis down must not break the inference (Section 9.9)."""
    decorated = CachingDecorator(stub_strategy, BrokenCacheBackend())
    first = decorated.predict(user_features, 3)
    second = decorated.predict(user_features, 3)
    assert first == second
    assert stub_strategy.calls == 2


def test_a_broken_cache_is_reported_in_the_operational_context(
    stub_strategy: StubStrategy, user_features: UserFeatures
) -> None:
    decorated = CachingDecorator(stub_strategy, BrokenCacheBackend())
    with operational_context():
        decorated.predict(user_features, 3)
        assert "cache_error" in current_context()


def test_a_corrupt_entry_is_treated_as_a_miss(
    stub_strategy: StubStrategy, user_features: UserFeatures
) -> None:
    backend = InMemoryCacheBackend()
    decorated = CachingDecorator(stub_strategy, backend, ttl_seconds=60)
    backend.set(decorated.cache_key(user_features, 3), "not json at all", 60)
    assert decorated.predict(user_features, 3)
    assert stub_strategy.calls == 1


def test_the_hit_is_published_to_the_operational_context(
    decorated: CachingDecorator, user_features: UserFeatures
) -> None:
    with operational_context():
        decorated.predict(user_features, 3)
        assert current_context()["cache_hit"] is False
    with operational_context():
        decorated.predict(user_features, 3)
        assert current_context()["cache_hit"] is True


def test_the_in_memory_backend_is_bounded() -> None:
    backend = InMemoryCacheBackend(max_entries=3)
    for i in range(10):
        backend.set(f"key-{i}", "[]", 60)
    assert len(backend) == 3
    assert backend.get("key-0") is None
    assert backend.get("key-9") == "[]"


def test_an_expired_entry_is_a_miss() -> None:
    backend = InMemoryCacheBackend()
    backend.set("key", "[]", 0)
    assert backend.get("key") is None


def test_it_only_wraps_a_strategy(stub_strategy: StubStrategy) -> None:
    with pytest.raises(TypeError):
        CachingDecorator("not a strategy")  # type: ignore[arg-type]
