"""FeedbackHistoryService - background worker for FR-06 (Section 3.1, 5.2).

Consumes interaction events from ``InteractionEventQueue`` and writes the
interaction and the recommendation shown through
``IInteractionHistoryRepository``, with retries on failure, so that no event
is lost if the persistence layer is momentarily unavailable.

This is **business history** - "user 27 received items A, B, C and clicked B".
It is deliberately separate from the operational logging performed by the
decorators - "model X executed in 143 ms". Table 6 records their conflation as
a Milestone 1 defect, and keeping them apart is why FR-06 traces here and not
to ``LoggingDecorator``.

Run it as its own process:

    python -m messaging.feedback_history_service
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from typing import Any

from config.settings import Settings, load_settings
from messaging.interaction_event_queue import (
    InteractionEvent,
    InteractionEventQueue,
    RetryBuffer,
    build_event_queue,
)
from persistence.interfaces import IInteractionHistoryRepository, InteractionRecord

#: Seconds waited between attempts when the repository rejects a write.
RETRY_BACKOFF_SECONDS = 0.2


class FeedbackHistoryService:
    """Background worker persisting the recommendation and feedback history."""

    def __init__(
        self,
        queue: InteractionEventQueue,
        repository: IInteractionHistoryRepository,
        *,
        max_retries: int = 3,
        backoff_seconds: float = RETRY_BACKOFF_SECONDS,
        logger: Any | None = None,
    ) -> None:
        self._queue = queue
        self._repository = repository
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds
        self._logger = logger
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.processed = 0
        self.duplicates = 0
        self.failures = 0

    @classmethod
    def from_settings(
        cls,
        repository: IInteractionHistoryRepository,
        settings: Settings | None = None,
        *,
        queue: InteractionEventQueue | None = None,
        logger: Any | None = None,
    ) -> FeedbackHistoryService:
        """Build the worker against the configured broker.

        The repository is injected rather than constructed here: this module
        belongs to the messaging layer and must not name a concrete
        persistence adapter. Only a composition root does that, which is why
        ``main`` below - a process entry point - is the one place in this file
        allowed to import one.
        """
        settings = settings or load_settings()
        return cls(
            queue if queue is not None else build_event_queue(settings),
            repository,
            logger=logger,
        )

    def handle(self, event: InteractionEvent) -> bool:
        """Persist one event, retrying a transient repository failure.

        Returns:
            True when the event was written or recognised as a duplicate, so
            the broker can acknowledge it; False when every attempt failed and
            the event must be redelivered.
        """
        record = _to_record(event)
        for attempt in range(1, self._max_retries + 1):
            try:
                written = self._repository.append(record)
                if written:
                    self.processed += 1
                else:
                    # Idempotency: a redelivery after a failure must not
                    # duplicate the history record (Section 9.9).
                    self.duplicates += 1
                self._log("debug", "history.written", event_id=event.event_id, written=written)
                return True
            except Exception as exc:  # noqa: BLE001 - retried, then reported
                self._log(
                    "warning",
                    "history.write_failed",
                    event_id=event.event_id,
                    attempt=attempt,
                    error=f"{type(exc).__name__}: {exc}",
                )
                if attempt < self._max_retries:
                    time.sleep(self._backoff_seconds * attempt)
        self.failures += 1
        self._log("error", "history.write_abandoned", event_id=event.event_id)
        return False

    def run_once(self, *, max_events: int | None = None, timeout: float | None = 0.05) -> int:
        """Drain what is currently in the queue and return how much was written."""
        return self._queue.consume(self.handle, max_events=max_events, timeout=timeout)

    def run_forever(self, *, poll_interval: float = 0.1) -> None:
        """Consume until ``stop`` is called. This is the worker entry point."""
        self._log("info", "worker.started", backend=self._queue.describe())
        while not self._stop.is_set():
            try:
                consumed = self.run_once(max_events=64, timeout=poll_interval)
            except Exception as exc:  # noqa: BLE001 - the worker must survive one bad event
                self._log("error", "worker.iteration_failed", error=f"{type(exc).__name__}: {exc}")
                consumed = 0
            if consumed == 0:
                self._stop.wait(poll_interval)
        self._log("info", "worker.stopped", processed=self.processed, failures=self.failures)

    def start(self) -> None:
        """Run ``run_forever`` on a daemon thread, for in-process deployments."""
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self.run_forever, name="feedback-history", daemon=True
        )
        self._thread.start()

    def stop(self, *, timeout: float = 2.0) -> None:
        """Ask the worker to finish and wait for its thread."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def drain_retry_buffer(self, buffer: RetryBuffer) -> int:
        """Persist the events that could never reach the broker (Table 10)."""
        events = buffer.read_all()
        written = sum(1 for event in events if self.handle(event))
        if written == len(events):
            buffer.clear()
        return written

    def stats(self) -> dict[str, int]:
        """Counters reported by /health and asserted by the tests."""
        return {
            "processed": self.processed,
            "duplicates": self.duplicates,
            "failures": self.failures,
        }

    def _log(self, level: str, event: str, **fields: Any) -> None:
        if self._logger is None:
            return
        getattr(self._logger, level, lambda *a, **k: None)(event, **fields)


def _to_record(event: InteractionEvent) -> InteractionRecord:
    """Translate a transport event into the persistence contract."""
    return InteractionRecord(
        event_id=event.event_id,
        user_id=event.user_id,
        event_type=event.event_type,
        occurred_at=event.occurred_at,
        item_id=event.item_id,
        recommended_items=event.recommended_items,
        model_version=event.model_version,
        metadata=event.metadata,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m messaging.feedback_history_service",
        description="Background worker persisting the recommendation history (FR-06).",
    )
    parser.add_argument("--once", action="store_true", help="Drain the queue and exit")
    parser.add_argument("--max-events", type=int, default=None)
    args = parser.parse_args(argv)

    # Composition root of the worker process: the only place in this module
    # that names a concrete adapter.
    from ai.decorators.logging_decorator import configure_logging
    from persistence.nosql_interaction_history_repository import NoSqlInteractionHistoryRepository

    settings = load_settings()
    service = FeedbackHistoryService.from_settings(
        NoSqlInteractionHistoryRepository.from_settings(settings),
        settings,
        logger=configure_logging(settings),
    )

    if args.once:
        consumed = service.run_once(max_events=args.max_events, timeout=0.2)
        print(json.dumps({"consumed": consumed, **service.stats()}, indent=2))
        return 0

    try:
        service.run_forever()
    except KeyboardInterrupt:
        service.stop()
    print(json.dumps(service.stats(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["FeedbackHistoryService"]
