"""InteractionIngestionService - real-time interaction capture (FR-02).

Receives user interaction events and publishes them to the
``InteractionEventQueue`` so that ingestion never blocks navigation
(Section 5.2). It performs the Fail Fast validation of the interaction
payload before publishing, which is the entry point NFR-04 names alongside
the controller.

It does not write to the database. The consuming side of the queue is
``FeedbackHistoryService`` (FR-06), and keeping the two apart is what makes
the asynchronous path survive a momentary failure of the persistence layer.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from errors import EventPublishError
from messaging.interaction_event_queue import InteractionEvent, InteractionEventQueue
from presentation.schemas import InteractionAccepted, InteractionRequest


class InteractionIngestionService:
    """Validates interaction events and hands them to the broker."""

    def __init__(
        self,
        event_queue: InteractionEventQueue,
        *,
        logger: Any | None = None,
    ) -> None:
        self._event_queue = event_queue
        self._logger = logger

    def ingest(self, request: InteractionRequest) -> InteractionAccepted:
        """Publish one validated interaction event.

        The payload has already been validated against the schema by the
        framework; this method re-checks the invariants the schema cannot
        express and then publishes without waiting for a consumer.

        Returns:
            An acknowledgement carrying the event identifier. ``queued`` is
            False when the broker refused the event and it was written to the
            retry buffer instead; the caller still receives a success, which
            is the behaviour Table 10 specifies for ``EventPublishError``.
        """
        event = self._to_event(request)
        try:
            self._event_queue.publish(event)
        except EventPublishError as exc:
            self._log(
                "warning", "ingestion.publish_failed", event_id=event.event_id, detail=exc.detail
            )
            return InteractionAccepted(event_id=event.event_id, accepted=True, queued=False)
        except Exception as exc:  # noqa: BLE001 - ingestion never propagates a broker failure
            self._log(
                "warning",
                "ingestion.publish_failed",
                event_id=event.event_id,
                error=f"{type(exc).__name__}: {exc}",
            )
            return InteractionAccepted(event_id=event.event_id, accepted=True, queued=False)

        self._log("info", "ingestion.published", event_id=event.event_id, user_id=event.user_id)
        return InteractionAccepted(event_id=event.event_id, accepted=True, queued=True)

    @staticmethod
    def _to_event(request: InteractionRequest) -> InteractionEvent:
        """Translate a validated request into a transport event."""
        if request.item_id is None and not request.recommended_items:
            raise ValueError("an interaction must reference an item or a shown recommendation")
        return InteractionEvent(
            user_id=request.user_id,
            event_type=request.event_type,
            occurred_at=request.occurred_at or datetime.now(UTC),
            item_id=request.item_id,
            recommended_items=tuple(request.recommended_items),
            model_version=request.model_version,
            metadata=dict(request.metadata),
        )

    def _log(self, level: str, event: str, **fields: Any) -> None:
        if self._logger is None:
            return
        getattr(self._logger, level, lambda *a, **k: None)(event, **fields)


__all__ = ["InteractionIngestionService"]
