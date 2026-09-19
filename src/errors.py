"""Exception hierarchy of Table 10 (Section 9.8).

The document specifies the exception types, the HTTP response, and the
recovery behaviour, but it does not assign them a module in the tree of
Section 9.2. They are declared here, at the source root, because they are
raised in four different layers (persistence, data, ai, messaging) and
translated in a single FastAPI handler in the application layer: placing them
inside any one layer would force the other layers to import from it and would
break the dependency direction rule.

``ValidationError`` is the sixth row of Table 10 and is *not* redefined here:
it is Pydantic's own exception, raised by the request schemas of the
presentation layer and translated by the same handler.
"""

from __future__ import annotations

from http import HTTPStatus


class RecommendationPlatformError(Exception):
    """Base class for every error the platform raises on its own behalf.

    Internal components never propagate library exceptions across a layer
    boundary; they translate them into one of the subclasses below, so the
    application layer depends on this vocabulary and not on Redis, pika,
    SQLAlchemy, or PyMongo error types (Section 9.8).
    """

    #: HTTP status the exception handler returns for this type.
    status_code: int = HTTPStatus.INTERNAL_SERVER_ERROR
    #: Stable machine-readable code returned in the error payload.
    code: str = "internal_error"
    #: Whether the request can still be served with a degraded response.
    degradable: bool = False

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail

    def to_payload(self) -> dict[str, str]:
        """Client-facing representation, free of stack traces and internal ids."""
        payload = {"code": self.code, "message": self.message}
        if self.detail is not None:
            payload["detail"] = self.detail
        return payload


class UserNotFoundError(RecommendationPlatformError):
    """The user identifier does not exist in the profile repository (HTTP 404)."""

    status_code = HTTPStatus.NOT_FOUND
    code = "user_not_found"
    degradable = False


class ColdStartError(RecommendationPlatformError):
    """The user or item has no usable interaction history.

    HTTP 200: PopularityStrategy is used as a fallback (FR-05).
    """

    status_code = HTTPStatus.OK
    code = "cold_start"
    degradable = True


class FeatureStoreUnavailableError(RecommendationPlatformError):
    """The feature store cannot be read.

    HTTP 200 degraded: fallback to popularity ranking, logged at error level.
    """

    status_code = HTTPStatus.OK
    code = "feature_store_unavailable"
    degradable = True


class ModelUnavailableError(RecommendationPlatformError):
    """The configured model cannot be loaded or the inference fails.

    HTTP 200 degraded: fallback to popularity ranking, preserving catalog
    navigation (NFR-05).
    """

    status_code = HTTPStatus.OK
    code = "model_unavailable"
    degradable = True


class EventPublishError(RecommendationPlatformError):
    """The interaction event cannot be published to the queue.

    Does not affect the response: the event is written to a local retry
    buffer and the user response is unaffected.
    """

    status_code = HTTPStatus.OK
    code = "event_publish_failed"
    degradable = True


#: Reason codes attached to a degraded response, so a client (and the tests)
#: can tell *why* the popularity fallback was served.
DEGRADATION_REASONS: dict[type[RecommendationPlatformError], str] = {
    ColdStartError: ColdStartError.code,
    FeatureStoreUnavailableError: FeatureStoreUnavailableError.code,
    ModelUnavailableError: ModelUnavailableError.code,
}

__all__ = [
    "DEGRADATION_REASONS",
    "ColdStartError",
    "EventPublishError",
    "FeatureStoreUnavailableError",
    "ModelUnavailableError",
    "RecommendationPlatformError",
    "UserNotFoundError",
]
