"""InteractionEventQueue and FeedbackHistoryService (FR-02, FR-06).

Section 5.3 claims the asynchronous persistence path is guaranteed by the
broker and runs outside the request cycle. These tests exercise both halves
of that claim: publication does not run the consumer, and a consumer failure
leaves the event for redelivery instead of losing it.
"""

from __future__ import annotations

import inspect
from datetime import UTC, datetime
from pathlib import Path

import pytest

from config.settings import Settings
from errors import EventPublishError
from messaging.feedback_history_service import FeedbackHistoryService
from messaging.interaction_event_queue import (
    InMemoryInteractionEventQueue,
    InteractionEvent,
    InteractionEventQueue,
    RetryBuffer,
    build_event_queue,
)
from persistence.interfaces import IInteractionHistoryRepository, InteractionRecord
from persistence.nosql_interaction_history_repository import (
    LocalDocumentBackend,
    NoSqlInteractionHistoryRepository,
)


class BrokenQueue(InteractionEventQueue):
    """Queue that always refuses, for the EventPublishError path of Table 10."""

    def __init__(self, retry_buffer: RetryBuffer | None = None) -> None:
        self._retry_buffer = retry_buffer

    def publish(self, event: InteractionEvent) -> None:
        if self._retry_buffer is not None:
            self._retry_buffer.append(event)
        raise EventPublishError("broker is down")

    def consume(self, handler, *, max_events=None, timeout=None) -> int:  # noqa: ANN001
        return 0

    def describe(self) -> str:
        return "broken"


class BrokenRepository(IInteractionHistoryRepository):
    """Repository that always fails, for the retry path."""

    def __init__(self, fail_times: int = 99) -> None:
        self.attempts = 0
        self._fail_times = fail_times
        self.written: list[InteractionRecord] = []

    def append(self, record: InteractionRecord) -> bool:
        self.attempts += 1
        if self.attempts <= self._fail_times:
            raise ConnectionError("document store is down")
        self.written.append(record)
        return True

    def list_by_user(self, user_id: int, limit: int = 100) -> list[InteractionRecord]:
        return list(self.written)

    def count_by_user(self, user_id: int) -> int:
        return len(self.written)

    def exists_event(self, event_id: str) -> bool:
        return any(r.event_id == event_id for r in self.written)


@pytest.fixture
def queue() -> InMemoryInteractionEventQueue:
    return InMemoryInteractionEventQueue()


@pytest.fixture
def repository(tmp_path: Path) -> NoSqlInteractionHistoryRepository:
    return NoSqlInteractionHistoryRepository(LocalDocumentBackend(tmp_path / "history.jsonl"))


def _event(event_id: str = "e1", user_id: int = 7) -> InteractionEvent:
    return InteractionEvent(
        user_id=user_id,
        event_type="recommendation_served",
        event_id=event_id,
        occurred_at=datetime.now(UTC),
        recommended_items=(101, 102),
        model_version="v1",
    )


# --------------------------------------------------------------------------
# InteractionEventQueue
# --------------------------------------------------------------------------


def test_an_event_round_trips_through_json() -> None:
    original = _event()
    restored = InteractionEvent.from_json(original.to_json())
    assert restored.event_id == original.event_id
    assert restored.recommended_items == original.recommended_items
    assert restored.user_id == original.user_id


def test_events_get_an_identifier_automatically() -> None:
    assert InteractionEvent(user_id=1, event_type="click").event_id


def test_publishing_does_not_run_the_consumer(
    queue: InMemoryInteractionEventQueue, repository: NoSqlInteractionHistoryRepository
) -> None:
    """Section 5.3: the write runs outside the request path, not inside it."""
    queue.publish(_event())
    assert queue.pending() == 1
    assert repository.count_by_user(7) == 0


def test_consuming_delivers_the_event(
    queue: InMemoryInteractionEventQueue, repository: NoSqlInteractionHistoryRepository
) -> None:
    queue.publish(_event())
    worker = FeedbackHistoryService(queue, repository)
    assert worker.run_once(max_events=10) == 1
    assert repository.count_by_user(7) == 1


def test_an_empty_queue_consumes_nothing(queue: InMemoryInteractionEventQueue) -> None:
    worker = FeedbackHistoryService(queue, BrokenRepository())
    assert worker.run_once(max_events=10, timeout=0.01) == 0


def test_the_retry_buffer_keeps_an_unpublishable_event(tmp_path: Path) -> None:
    """Table 10: EventPublishError writes the event to a local retry buffer."""
    buffer = RetryBuffer(tmp_path / "retry.jsonl")
    broken = BrokenQueue(buffer)
    with pytest.raises(EventPublishError):
        broken.publish(_event())
    assert len(buffer) == 1
    assert buffer.read_all()[0].user_id == 7


def test_the_retry_buffer_can_be_drained(tmp_path: Path, repository) -> None:
    buffer = RetryBuffer(tmp_path / "retry.jsonl")
    buffer.append(_event("buffered"))
    worker = FeedbackHistoryService(InMemoryInteractionEventQueue(), repository)
    assert worker.drain_retry_buffer(buffer) == 1
    assert repository.exists_event("buffered") is True
    assert len(buffer) == 0


def test_the_backend_is_selected_from_configuration(settings: Settings) -> None:
    """Without a broker URL the in-process backend serves the same contract."""
    selected = build_event_queue(settings)
    assert isinstance(selected, InteractionEventQueue)
    assert selected.describe() == "in-memory"


def test_the_queue_contract_is_an_abstract_base_class() -> None:
    assert inspect.isabstract(InteractionEventQueue)
    with pytest.raises(TypeError):
        InteractionEventQueue()  # type: ignore[abstract]


# --------------------------------------------------------------------------
# FeedbackHistoryService
# --------------------------------------------------------------------------


def test_the_worker_writes_the_business_history(
    queue: InMemoryInteractionEventQueue, repository: NoSqlInteractionHistoryRepository
) -> None:
    worker = FeedbackHistoryService(queue, repository)
    queue.publish(_event())
    worker.run_once(max_events=10)

    stored = repository.list_by_user(7)[0]
    assert stored.recommended_items == (101, 102)
    assert stored.model_version == "v1"
    assert worker.stats() == {"processed": 1, "duplicates": 0, "failures": 0}


def test_a_redelivery_does_not_duplicate_the_history(
    queue: InMemoryInteractionEventQueue, repository: NoSqlInteractionHistoryRepository
) -> None:
    """Section 9.9: the worker is idempotent with respect to the events."""
    worker = FeedbackHistoryService(queue, repository)
    event = _event()
    queue.publish(event)
    worker.run_once(max_events=10)
    queue.publish(event)
    worker.run_once(max_events=10)

    assert repository.count_by_user(7) == 1
    assert worker.stats()["duplicates"] == 1


def test_a_failing_repository_is_retried(
    queue: InMemoryInteractionEventQueue,
) -> None:
    broken = BrokenRepository(fail_times=2)
    worker = FeedbackHistoryService(queue, broken, max_retries=3, backoff_seconds=0.0)
    queue.publish(_event())
    assert worker.run_once(max_events=10) == 1
    assert broken.attempts == 3
    assert worker.stats()["processed"] == 1


def test_an_event_is_kept_when_every_retry_fails(
    queue: InMemoryInteractionEventQueue,
) -> None:
    """No event is lost if the persistence layer is momentarily unavailable."""
    worker = FeedbackHistoryService(queue, BrokenRepository(), max_retries=2, backoff_seconds=0.0)
    queue.publish(_event())
    assert worker.run_once(max_events=10) == 0
    assert queue.pending() == 1
    assert worker.stats()["failures"] == 1


def test_the_event_survives_until_the_repository_recovers(
    queue: InMemoryInteractionEventQueue, repository: NoSqlInteractionHistoryRepository
) -> None:
    queue.publish(_event())
    failing = FeedbackHistoryService(queue, BrokenRepository(), max_retries=1, backoff_seconds=0.0)
    failing.run_once(max_events=10)

    recovered = FeedbackHistoryService(queue, repository)
    assert recovered.run_once(max_events=10) == 1
    assert repository.count_by_user(7) == 1


def test_the_worker_runs_on_its_own_thread(
    queue: InMemoryInteractionEventQueue, repository: NoSqlInteractionHistoryRepository
) -> None:
    import time

    worker = FeedbackHistoryService(queue, repository)
    worker.start()
    try:
        queue.publish(_event())
        deadline = time.time() + 3.0
        while time.time() < deadline and repository.count_by_user(7) == 0:
            time.sleep(0.02)
    finally:
        worker.stop()
    assert repository.count_by_user(7) == 1


def test_the_worker_does_not_emit_operational_logs() -> None:
    """The decorators own operational logging; this worker owns the history."""
    source = inspect.getsource(FeedbackHistoryService)
    for forbidden in ("cache_hit", "latency_ms", "strategy_id"):
        assert forbidden not in source, f"FeedbackHistoryService mentions {forbidden!r}"
