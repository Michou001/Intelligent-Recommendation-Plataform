"""RecommendationController - orchestration of the inference cycle (FR-01, FR-04).

The controller contains no algorithmic logic (Section 3.1, Section 9.6). It
validates the request, asks ``FeatureStoreManager`` for the feature vectors,
invokes ``predict`` on the decorated strategy it was injected with, enriches
the ranking from the catalog, publishes the interaction event, and returns the
Top-K list.

Every collaborator arrives through an abstraction - ``IRecommendationStrategy``,
``FeatureStoreManager``, ``InteractionEventQueue``, and the segregated
repository interfaces - so this class can be unit tested against doubles
without a model, a broker, or a database (Section 7).

Degradation is concentrated here, and it is what makes NFR-05 hold: a feature
store that cannot be read, a user without history, or a model that fails to
load produce a successful degraded response served by the popularity
fallback, never an abrupt termination.
"""

from __future__ import annotations

import time
from typing import Any

from ai.interfaces import IRecommendationStrategy, ScoredItem, UserFeatures
from config.settings import Settings, load_settings
from data.feature_store_manager import FeatureStoreManager
from errors import (
    ColdStartError,
    EventPublishError,
    FeatureStoreUnavailableError,
    ModelUnavailableError,
    RecommendationPlatformError,
    UserNotFoundError,
)
from messaging.interaction_event_queue import InteractionEvent, InteractionEventQueue
from persistence.interfaces import ICatalogRepository, IUserProfileRepository
from presentation.schemas import RecommendationResponse, RecommendedItem

#: Event type published when a ranking is served (FR-06).
SERVED_EVENT_TYPE = "recommendation_served"


class RecommendationController:
    """Orchestrates one recommendation request end to end."""

    def __init__(
        self,
        strategy: IRecommendationStrategy,
        feature_store: FeatureStoreManager,
        *,
        event_queue: InteractionEventQueue | None = None,
        catalog_repository: ICatalogRepository | None = None,
        profile_repository: IUserProfileRepository | None = None,
        fallback_strategy: IRecommendationStrategy | None = None,
        settings: Settings | None = None,
        logger: Any | None = None,
        startup_degradation_reason: str | None = None,
    ) -> None:
        self._strategy = strategy
        self._feature_store = feature_store
        self._event_queue = event_queue
        self._catalog = catalog_repository
        self._profiles = profile_repository
        self._fallback = fallback_strategy
        self._settings = settings or load_settings()
        self._logger = logger
        # Set by the composition root when the configured strategy could not be
        # loaded at startup and the fallback took its place. That degradation
        # lasts for the life of the process, so every response has to declare
        # it: otherwise the one case where the system is *persistently*
        # degraded would be the only one the client cannot see (NFR-05).
        self._startup_degradation_reason = startup_degradation_reason

    @property
    def strategy(self) -> IRecommendationStrategy:
        """The decorated object the controller invokes."""
        return self._strategy

    def recommend(self, user_id: int, k: int | None = None) -> RecommendationResponse:
        """Serve the Top-K ranking for a user.

        Raises:
            ValueError: ``user_id`` or ``k`` is outside the accepted range.
                Fail Fast happens before any collaborator is touched.
            UserNotFoundError: The profile repository does not know the user.
        """
        started = time.perf_counter()
        k = self._validate(user_id, k)

        self._assert_user_exists(user_id)

        runtime_reason: str | None = None
        features = self._load_features(user_id)
        if features is None:
            runtime_reason = FeatureStoreUnavailableError.code
            features = UserFeatures(
                user_id=user_id,
                is_cold_start=True,
                feature_store_version=self._safe_store_version(),
            )
        elif features.is_cold_start:
            runtime_reason = ColdStartError.code

        ranking, strategy_used, reason = self._rank(features, k, runtime_reason)
        # A failure during this request takes precedence over the startup one,
        # because it is the more specific explanation of what the client got.
        degradation_reason = reason or self._startup_degradation_reason

        response = RecommendationResponse(
            user_id=user_id,
            k=k,
            items=self._enrich(ranking),
            strategy_id=strategy_used.strategy_id,
            model_version=strategy_used.model_version,
            feature_store_version=features.feature_store_version,
            degraded=degradation_reason is not None,
            degradation_reason=degradation_reason,
            latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
        )

        self._publish_served_event(response)
        return response

    def _validate(self, user_id: int, k: int | None) -> int:
        """Fail Fast: reject an invalid request before any computation."""
        if not isinstance(user_id, int) or isinstance(user_id, bool) or user_id < 1:
            raise ValueError("user_id must be a positive integer")
        resolved = self._settings.model.default_k if k is None else k
        if not isinstance(resolved, int) or isinstance(resolved, bool) or resolved < 1:
            raise ValueError("k must be a positive integer")
        if resolved > self._settings.model.max_k:
            raise ValueError(f"k must not exceed {self._settings.model.max_k}")
        return resolved

    def _assert_user_exists(self, user_id: int) -> None:
        """A user the profile repository does not know is a 404, not a fallback."""
        if self._profiles is None:
            return
        try:
            known = self._profiles.exists(user_id)
        except Exception as exc:  # noqa: BLE001 - an unreachable profile store must not 500
            self._log("warning", "profile.lookup_failed", user_id=user_id, error=str(exc))
            return
        if not known:
            raise UserNotFoundError(f"user {user_id} does not exist")

    def _load_features(self, user_id: int) -> UserFeatures | None:
        """Retrieve the precomputed vectors, or None when the store is down."""
        try:
            return self._feature_store.get_user_features(user_id)
        except ColdStartError:
            return UserFeatures(
                user_id=user_id,
                is_cold_start=True,
                feature_store_version=self._safe_store_version(),
            )
        except FeatureStoreUnavailableError as exc:
            self._log("error", "feature_store.unavailable", user_id=user_id, detail=exc.detail)
            return None
        except Exception as exc:  # noqa: BLE001 - any storage failure degrades, never 500
            self._log("error", "feature_store.failed", user_id=user_id, error=str(exc))
            return None

    def _rank(
        self, features: UserFeatures, k: int, reason: str | None
    ) -> tuple[list[ScoredItem], IRecommendationStrategy, str | None]:
        """Run the configured strategy, falling back on a degradable error."""
        if reason is None:
            try:
                return self._strategy.predict(features, k), self._strategy, None
            except RecommendationPlatformError as exc:
                if not exc.degradable:
                    raise
                self._log("warning", "strategy.degraded", code=exc.code, detail=exc.detail)
                reason = exc.code
            except Exception as exc:  # noqa: BLE001 - an unexpected model failure degrades too
                self._log("error", "strategy.failed", error=f"{type(exc).__name__}: {exc}")
                reason = ModelUnavailableError.code

        fallback = self._resolve_fallback()
        if fallback is None:
            raise ModelUnavailableError(
                "no recommendation could be produced and no fallback is configured",
                detail=reason,
            )
        try:
            return fallback.predict(features, k), fallback, reason
        except Exception as exc:  # noqa: BLE001 - the last resort is an empty ranking
            self._log("error", "fallback.failed", error=f"{type(exc).__name__}: {exc}")
            return [], fallback, reason or ModelUnavailableError.code

    def _resolve_fallback(self) -> IRecommendationStrategy | None:
        """The configured fallback, or the active strategy when it is already it."""
        if self._fallback is not None:
            return self._fallback
        if self._strategy.strategy_id == self._settings.model.fallback_strategy:
            return self._strategy
        return None

    def _enrich(self, ranking: list[ScoredItem]) -> list[RecommendedItem]:
        """Attach catalog titles and categories to the ranked identifiers."""
        titles: dict[int, tuple[str, list[str]]] = {}
        if self._catalog is not None and ranking:
            try:
                for item in self._catalog.get_items([entry.item_id for entry in ranking]):
                    titles[int(item.item_id)] = (item.title, list(item.categories))
            except Exception as exc:  # noqa: BLE001 - a ranking without titles still serves
                self._log("warning", "catalog.lookup_failed", error=str(exc))

        enriched: list[RecommendedItem] = []
        for position, entry in enumerate(ranking, start=1):
            title, categories = titles.get(entry.item_id, (None, []))
            enriched.append(
                RecommendedItem(
                    item_id=entry.item_id,
                    rank=position,
                    score=entry.score,
                    title=title,
                    categories=categories,
                )
            )
        return enriched

    def _publish_served_event(self, response: RecommendationResponse) -> None:
        """Hand the interaction to the broker; never let it affect the response.

        Section 5.3: this branch runs outside the request/response cycle and
        is guaranteed by the broker, so a publication failure is logged and
        buffered, and the user still receives their ranking (Table 10).
        """
        if self._event_queue is None:
            return
        event = InteractionEvent(
            user_id=response.user_id,
            event_type=SERVED_EVENT_TYPE,
            recommended_items=tuple(item.item_id for item in response.items),
            model_version=response.model_version,
            metadata={
                "strategy_id": response.strategy_id,
                "degraded": str(response.degraded).lower(),
                "k": str(response.k),
            },
        )
        try:
            self._event_queue.publish(event)
        except EventPublishError as exc:
            self._log("warning", "event.publish_failed", code=exc.code, detail=exc.detail)
        except Exception as exc:  # noqa: BLE001 - publication never breaks a served response
            self._log("warning", "event.publish_failed", error=f"{type(exc).__name__}: {exc}")

    def _safe_store_version(self) -> str:
        try:
            return self._feature_store.version
        except Exception:  # noqa: BLE001 - a broken store still reports a response
            return "unknown"

    def _log(self, level: str, event: str, **fields: Any) -> None:
        if self._logger is None:
            return
        getattr(self._logger, level, lambda *a, **k: None)(event, **fields)


__all__ = ["RecommendationController"]
