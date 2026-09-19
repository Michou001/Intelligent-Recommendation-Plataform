"""CachingDecorator - Top-K result cache (Table 4, NFR-01).

Stores the Top-K result under a key composed of the user identifier, the
value of k, and the model version, so that repeated requests skip the
inference entirely. The model version inside the key is what makes the cache
invalidate itself when a new model is deployed: a new version produces new
keys and the stale entries are never read again (Section 9.7).

Redis is the backend of Table 7. When no Redis URL is configured, or when
Redis cannot be reached, the decorator uses an in-process store with the same
interface. A cache failure is never an inference failure: it degrades into a
miss, which is one of the failure-injection cases required by Section 9.9.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any

from ai.decorators import StrategyDecorator, operational_context, record, register_decorator
from ai.interfaces import IRecommendationStrategy, ScoredItem, UserFeatures
from config.settings import Settings, load_settings

#: Prefix of every key written by this decorator, so the namespace is visible
#: in a shared Redis instance.
KEY_PREFIX = "reco:topk"


class CacheBackend(ABC):
    """Minimal key/value contract the decorator needs from a cache."""

    @abstractmethod
    def get(self, key: str) -> str | None:
        """Return the stored payload, or None on a miss."""

    @abstractmethod
    def set(self, key: str, value: str, ttl_seconds: int) -> None:
        """Store a payload with an expiry."""

    @abstractmethod
    def describe(self) -> str:
        """Backend name, emitted in the operational log."""


class InMemoryCacheBackend(CacheBackend):
    """Bounded in-process LRU with per-entry expiry.

    Used when no Redis endpoint is configured. It keeps the decorator
    meaningful in development and in the tests, at the cost of not being
    shared between service instances.
    """

    def __init__(self, max_entries: int = 4096) -> None:
        self._entries: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._max_entries = max_entries

    def get(self, key: str) -> str | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        expires_at, payload = entry
        if expires_at < time.time():
            del self._entries[key]
            return None
        self._entries.move_to_end(key)
        return payload

    def set(self, key: str, value: str, ttl_seconds: int) -> None:
        self._entries[key] = (time.time() + ttl_seconds, value)
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def describe(self) -> str:
        return "in-memory"

    def clear(self) -> None:
        """Drop every entry, used between tests."""
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)


class RedisCacheBackend(CacheBackend):
    """Redis backend of Table 7."""

    def __init__(self, client: Any) -> None:
        self._client = client

    @classmethod
    def connect(cls, url: str) -> RedisCacheBackend:
        """Open a Redis client. Raises when the package or the server is absent."""
        import redis

        client = redis.Redis.from_url(url, decode_responses=True, socket_timeout=0.25)
        client.ping()
        return cls(client)

    def get(self, key: str) -> str | None:
        value = self._client.get(key)
        return None if value is None else str(value)

    def set(self, key: str, value: str, ttl_seconds: int) -> None:
        self._client.setex(key, ttl_seconds, value)

    def describe(self) -> str:
        return "redis"


@register_decorator("caching")
class CachingDecorator(StrategyDecorator):
    """Serves a previously computed Top-K instead of running the inference."""

    def __init__(
        self,
        wrapped: IRecommendationStrategy,
        backend: CacheBackend | None = None,
        *,
        ttl_seconds: int | None = None,
        settings: Settings | None = None,
    ) -> None:
        super().__init__(wrapped)
        settings = settings or load_settings()
        self._backend = backend if backend is not None else _select_backend(settings)
        self._ttl_seconds = ttl_seconds if ttl_seconds is not None else settings.cache.ttl_seconds

    @property
    def backend(self) -> CacheBackend:
        """The cache currently in use, for inspection and tests."""
        return self._backend

    def cache_key(self, user_features: UserFeatures, k: int) -> str:
        """Key composed of the user, the value of k, and the model version."""
        return f"{KEY_PREFIX}:{user_features.user_id}:{k}:{self.model_version}"

    def predict(self, user_features: UserFeatures, k: int = 10) -> list[ScoredItem]:
        key = self.cache_key(user_features, k)

        with operational_context():
            cached = self._read(key)
            if cached is not None:
                record(cache_hit=True, cache_backend=self._backend.describe())
                return cached

            record(cache_hit=False, cache_backend=self._backend.describe())
            result = self._wrapped.predict(user_features, k)
            self._write(key, result)
            return result

    def _read(self, key: str) -> list[ScoredItem] | None:
        """Read the cache, treating any backend failure as a miss."""
        try:
            payload = self._backend.get(key)
        except Exception as exc:  # noqa: BLE001 - a broken cache must not break inference
            record(cache_error=f"{type(exc).__name__}: {exc}")
            return None
        if payload is None:
            return None
        try:
            entries = json.loads(payload)
            return [ScoredItem(item_id=int(e["item_id"]), score=float(e["score"])) for e in entries]
        except (ValueError, KeyError, TypeError) as exc:
            record(cache_error=f"corrupt entry: {type(exc).__name__}: {exc}")
            return None

    def _write(self, key: str, result: list[ScoredItem]) -> None:
        """Write the cache, treating any backend failure as a no-op."""
        try:
            payload = json.dumps(
                [{"item_id": item.item_id, "score": item.score} for item in result]
            )
            self._backend.set(key, payload, self._ttl_seconds)
        except Exception as exc:  # noqa: BLE001 - a broken cache must not break inference
            record(cache_error=f"{type(exc).__name__}: {exc}")


def _select_backend(settings: Settings) -> CacheBackend:
    """Use Redis when it is configured and reachable, otherwise stay in process."""
    if settings.cache.url:
        try:
            return RedisCacheBackend.connect(settings.cache.url)
        except Exception:  # noqa: BLE001 - absent infrastructure is not an error here
            pass
    return InMemoryCacheBackend(max_entries=settings.cache.max_entries)


__all__ = [
    "CacheBackend",
    "CachingDecorator",
    "InMemoryCacheBackend",
    "RedisCacheBackend",
]
