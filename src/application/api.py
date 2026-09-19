"""FastAPI routes and the composition root (Section 9.6).

This module is the one place in the repository allowed to name concrete
classes from several layers at once. A composition root has to: it is what
wires the abstractions together. Everywhere else, a module depends on the
interfaces of ``ai/interfaces.py`` and ``persistence/interfaces.py``, and the
import-direction test asserts that this file is the only exception.

The composition reproduces the snippet of Section 9.6 literally:

    base      = ModelFactory.create(settings.ACTIVE_STRATEGY)   # registry lookup
    decorated = LoggingDecorator(TimingDecorator(CachingDecorator(base)))
    controller = RecommendationController(strategy=decorated, ...)

It runs once, at startup, so the request path performs no wiring.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import Depends, FastAPI, Path, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ai.decorators import build_chain
from ai.decorators.caching_decorator import CachingDecorator
from ai.decorators.logging_decorator import configure_logging
from ai.decorators.timing_decorator import TimingDecorator
from ai.interfaces import IRecommendationStrategy
from ai.model_factory import ModelFactory
from application.interaction_ingestion_service import InteractionIngestionService
from application.recommendation_controller import RecommendationController
from config.settings import Settings, load_settings
from data.feature_store_manager import ParquetFeatureStoreManager
from errors import ModelUnavailableError, RecommendationPlatformError
from messaging.feedback_history_service import FeedbackHistoryService
from messaging.interaction_event_queue import build_event_queue
from persistence.nosql_interaction_history_repository import NoSqlInteractionHistoryRepository
from persistence.sql_catalog_repository import SqlCatalogRepository
from persistence.sql_user_profile_repository import SqlUserProfileRepository
from presentation.schemas import (
    MAX_K,
    ErrorResponse,
    HealthResponse,
    InteractionAccepted,
    InteractionRequest,
    RecommendationResponse,
)


@dataclass
class ApplicationComponents:
    """Everything the composition root built, kept for the routes and /health."""

    settings: Settings
    controller: RecommendationController
    ingestion: InteractionIngestionService
    feature_store: ParquetFeatureStoreManager
    catalog: SqlCatalogRepository
    profiles: SqlUserProfileRepository
    history: NoSqlInteractionHistoryRepository
    worker: FeedbackHistoryService
    queue_backend: str
    cache_backend: str
    timing: TimingDecorator | None
    active_strategy: str
    startup_error: str | None = None


def build_components(settings: Settings | None = None) -> ApplicationComponents:
    """Wire the whole application from configuration, once."""
    settings = settings or load_settings()
    logger = configure_logging(settings)

    feature_store = ParquetFeatureStoreManager.from_settings(settings)
    catalog = SqlCatalogRepository.from_settings(settings)
    profiles = SqlUserProfileRepository.from_settings(settings)
    history = NoSqlInteractionHistoryRepository.from_settings(settings)
    queue = build_event_queue(settings)

    fallback, fallback_error = _build_fallback(settings)
    base, active_strategy, startup_error = _build_base_strategy(settings, fallback)
    decorated = build_chain(base, settings.model.decorator_chain, settings=settings)

    controller = RecommendationController(
        strategy=decorated,
        feature_store=feature_store,
        event_queue=queue,
        catalog_repository=catalog,
        profile_repository=profiles,
        fallback_strategy=fallback,
        settings=settings,
        logger=logger,
    )

    return ApplicationComponents(
        settings=settings,
        controller=controller,
        ingestion=InteractionIngestionService(queue, logger=logger),
        feature_store=feature_store,
        catalog=catalog,
        profiles=profiles,
        history=history,
        worker=FeedbackHistoryService(queue, history, logger=logger),
        queue_backend=queue.describe(),
        cache_backend=_find_cache_backend(decorated),
        timing=_find_timing(decorated),
        active_strategy=active_strategy,
        startup_error=startup_error or fallback_error,
    )


class _UnavailableStrategy(IRecommendationStrategy):
    """Null object used when neither the model nor the fallback can be built.

    Without it, a machine with no materialised artifacts would fail at
    startup, and a service that cannot start is not the controlled
    degradation NFR-05 asks for. With it, the service starts, ``/health``
    reports ``degraded``, and a recommendation request is answered with an
    explicit translated error instead of a crash.
    """

    strategy_id = "unavailable"

    def __init__(self, reason: str) -> None:
        self._reason = reason
        self._model_version = "none"

    def predict(self, user_features: Any, k: int = 10) -> list[Any]:
        raise ModelUnavailableError(
            "no recommendation strategy could be loaded", detail=self._reason
        )


def _build_base_strategy(
    settings: Settings, fallback: IRecommendationStrategy | None
) -> tuple[IRecommendationStrategy, str, str | None]:
    """Resolve the active strategy, degrading to the fallback if it cannot load.

    A model that fails to load must not stop the service from starting:
    NFR-05 requires catalog navigation to survive a failure of the
    recommendation subsystem.
    """
    try:
        base = ModelFactory.create(settings.model.active_strategy, settings)
        return base, settings.model.active_strategy, None
    except ModelUnavailableError as exc:
        reason = f"{exc.code}: {exc.message}"
        if fallback is None:
            return _UnavailableStrategy(reason), "unavailable", reason
        return fallback, settings.model.fallback_strategy, reason


def _build_fallback(settings: Settings) -> tuple[IRecommendationStrategy | None, str | None]:
    """Build the cold-start fallback, tolerating its absence."""
    try:
        return ModelFactory.create(settings.model.fallback_strategy, settings), None
    except ModelUnavailableError as exc:
        return None, f"fallback unavailable: {exc.message}"


def _find_timing(strategy: Any) -> TimingDecorator | None:
    """Locate the timing decorator in the chain, to report the served p95."""
    while strategy is not None:
        if isinstance(strategy, TimingDecorator):
            return strategy
        strategy = getattr(strategy, "wrapped", None)
    return None


def _find_cache_backend(strategy: Any) -> str:
    while strategy is not None:
        if isinstance(strategy, CachingDecorator):
            return strategy.backend.describe()
        strategy = getattr(strategy, "wrapped", None)
    return "none"


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the ASGI application with its routes and exception handlers."""
    components = build_components(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # With a real broker the worker is its own process (Section 5.2). With
        # the in-process queue there is nothing else to consume it, so the
        # worker runs on a daemon thread and the asynchronous path of FR-06
        # stays exercised.
        if components.queue_backend == "in-memory":
            components.worker.start()
        try:
            yield
        finally:
            components.worker.stop()

    app = FastAPI(
        title="Intelligent Recommendation Platform",
        version="0.2.0",
        summary="Milestone 2 implementation of the layered architecture proposal",
        lifespan=lifespan,
    )
    app.state.components = components

    def get_components() -> ApplicationComponents:
        return components

    @app.exception_handler(RequestValidationError)
    async def _on_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Table 10, row 1: a schema violation is a 400, not the default 422."""
        first = exc.errors()[0] if exc.errors() else {}
        field = ".".join(str(part) for part in first.get("loc", ()) if part != "body")
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                code="validation_error",
                message="the request does not satisfy the declared schema",
                detail=f"{field}: {first.get('msg', 'invalid value')}" if field else None,
            ).model_dump(),
        )

    @app.exception_handler(ValueError)
    async def _on_value_error(request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(code="validation_error", message=str(exc)).model_dump(),
        )

    @app.exception_handler(RecommendationPlatformError)
    async def _on_platform_error(
        request: Request, exc: RecommendationPlatformError
    ) -> JSONResponse:
        """Translate the vocabulary of Table 10 into status codes, once."""
        status = exc.status_code if exc.status_code != 200 else 503
        return JSONResponse(status_code=status, content=exc.to_payload())

    @app.get(
        "/recommendations/{user_id}",
        response_model=RecommendationResponse,
        responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
        summary="Top-K recommendations for a user (FR-04, FR-05)",
    )
    def get_recommendations(
        user_id: int = Path(ge=1, description="Identifier of the requesting user"),
        k: int = Query(default=10, ge=1, le=MAX_K, description="Size of the ranking"),
        parts: ApplicationComponents = Depends(get_components),
    ) -> RecommendationResponse:
        """The endpoint of Section 9.6: ``GET /recommendations/{user_id}?k=10``."""
        return parts.controller.recommend(user_id, k)

    @app.post(
        "/interactions",
        response_model=InteractionAccepted,
        status_code=202,
        responses={400: {"model": ErrorResponse}},
        summary="Publish a user interaction event (FR-02)",
    )
    def post_interaction(
        request: InteractionRequest,
        parts: ApplicationComponents = Depends(get_components),
    ) -> InteractionAccepted:
        """Queue an interaction without blocking navigation."""
        return parts.ingestion.ingest(request)

    @app.get("/health", response_model=HealthResponse, summary="Operational snapshot")
    def health(parts: ApplicationComponents = Depends(get_components)) -> HealthResponse:
        return HealthResponse(
            status="degraded" if parts.startup_error else "ok",
            active_strategy=parts.active_strategy,
            registered_strategies=list(ModelFactory.available()),
            decorator_chain=list(parts.settings.model.decorator_chain),
            feature_store_version=parts.feature_store.version,
            feature_store_available=parts.feature_store.is_available(),
            cache_backend=parts.cache_backend,
            queue_backend=parts.queue_backend,
            history_backend=parts.history.backend,
            catalog_items=_safe_count(parts.catalog.count),
            user_profiles=_safe_count(parts.profiles.count),
        )

    return app


def _safe_count(counter: Any) -> int:
    try:
        return int(counter())
    except Exception:  # noqa: BLE001 - /health must answer even with a broken store
        return -1


app = create_app()


__all__ = ["ApplicationComponents", "app", "build_components", "create_app"]
