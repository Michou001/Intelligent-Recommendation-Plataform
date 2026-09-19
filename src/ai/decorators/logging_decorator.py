"""LoggingDecorator - operational logging (Table 4, Section 9.9).

Emits one structured record per prediction describing which strategy ran,
whether the cache was hit, the measured latency, and the model version.

This is **operational** logging only. The business history - the
recommendations shown to a user and the feedback they gave - is written by
``FeedbackHistoryService`` through ``IInteractionHistoryRepository`` (FR-06).
Table 6 records the conflation of the two as a Milestone 1 defect, so nothing
about recommendation history is written here.

Four levels are used (Section 9.9): DEBUG for the vectors and intermediate
scores, INFO for every served request, WARNING for degraded responses, ERROR
for unavailable dependencies.
"""

from __future__ import annotations

from typing import Any

from ai.decorators import (
    StrategyDecorator,
    current_context,
    operational_context,
    register_decorator,
)
from ai.interfaces import IRecommendationStrategy, ScoredItem, UserFeatures
from config.settings import Settings, load_settings

#: Event name every operational record carries, so logs can be filtered by
#: field instead of by text search.
EVENT_NAME = "recommendation.served"

_configured = False


def configure_logging(settings: Settings | None = None) -> Any:
    """Configure structlog once, emitting JSON records (Table 7).

    Falls back to the standard library when structlog is not installed, so
    the decorator never becomes the reason a deployment fails.
    """
    global _configured
    settings = settings or load_settings()

    try:
        import structlog
    except ImportError:
        import logging

        logging.basicConfig(level=settings.logging.level)
        return logging.getLogger("recommendation")

    if not _configured:
        processors: list[Any] = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
        ]
        processors.append(
            structlog.processors.JSONRenderer()
            if settings.logging.json_output
            else structlog.dev.ConsoleRenderer()
        )
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.make_filtering_bound_logger(
                _level_number(settings.logging.level)
            ),
            cache_logger_on_first_use=True,
        )
        _configured = True
    return structlog.get_logger("recommendation")


def _level_number(name: str) -> int:
    import logging

    return getattr(logging, name.upper(), logging.INFO)


@register_decorator("logging")
class LoggingDecorator(StrategyDecorator):
    """Emits one structured operational record per prediction."""

    def __init__(
        self,
        wrapped: IRecommendationStrategy,
        logger: Any | None = None,
        *,
        settings: Settings | None = None,
    ) -> None:
        super().__init__(wrapped)
        self._logger = logger if logger is not None else configure_logging(settings)

    def predict(self, user_features: UserFeatures, k: int = 10) -> list[ScoredItem]:
        with operational_context():
            try:
                result = self._wrapped.predict(user_features, k)
            except Exception as exc:
                self._emit(
                    "error",
                    user_features,
                    k,
                    outcome="failed",
                    error=type(exc).__name__,
                    message=str(exc),
                )
                raise
            self._emit("info", user_features, k, outcome="served", returned=len(result))
            self._debug_scores(result)
            return result

    def _emit(self, level: str, user_features: UserFeatures, k: int, **fields: Any) -> None:
        """Write one record merging the fields the whole chain contributed."""
        payload: dict[str, Any] = {
            "user_id": user_features.user_id,
            "k": k,
            "strategy_id": self.strategy_id,
            "model_version": self.model_version,
            "feature_store_version": user_features.feature_store_version,
            **dict(current_context()),
            **fields,
        }
        getattr(self._logger, level)(EVENT_NAME, **payload)

    def _debug_scores(self, result: list[ScoredItem]) -> None:
        """Intermediate scores are DEBUG material, not INFO (Section 9.9)."""
        if not result:
            return
        self._logger.debug(
            "recommendation.scores",
            items=[item.item_id for item in result],
            scores=[round(item.score, 6) for item in result],
        )


__all__ = ["EVENT_NAME", "LoggingDecorator", "configure_logging"]
