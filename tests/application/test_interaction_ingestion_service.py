"""InteractionIngestionService (FR-02, NFR-04)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from application.interaction_ingestion_service import InteractionIngestionService
from messaging.interaction_event_queue import InMemoryInteractionEventQueue
from presentation.schemas import InteractionRequest


@pytest.fixture
def queue() -> InMemoryInteractionEventQueue:
    return InMemoryInteractionEventQueue()


@pytest.fixture
def service(queue: InMemoryInteractionEventQueue) -> InteractionIngestionService:
    return InteractionIngestionService(queue)


def test_it_queues_a_valid_interaction(
    service: InteractionIngestionService, queue: InMemoryInteractionEventQueue
) -> None:
    accepted = service.ingest(InteractionRequest(user_id=1, event_type="click", item_id=101))
    assert accepted.accepted is True
    assert accepted.queued is True
    assert queue.pending() == 1


def test_ingestion_does_not_wait_for_a_consumer(
    service: InteractionIngestionService, queue: InMemoryInteractionEventQueue
) -> None:
    """FR-02: capture must never block navigation."""
    for index in range(50):
        service.ingest(InteractionRequest(user_id=1, event_type="click", item_id=100 + index))
    assert queue.pending() == 50


def test_the_event_carries_the_payload(
    service: InteractionIngestionService, queue: InMemoryInteractionEventQueue
) -> None:
    service.ingest(
        InteractionRequest(
            user_id=7,
            event_type="rating",
            item_id=101,
            recommended_items=[101, 102],
            model_version="v1",
            metadata={"source": "web"},
        )
    )
    consumed: list = []
    queue.consume(lambda event: (consumed.append(event), True)[1], max_events=1)
    event = consumed[0]
    assert event.user_id == 7
    assert event.event_type == "rating"
    assert event.recommended_items == (101, 102)
    assert event.metadata == {"source": "web"}


def test_a_supplied_timestamp_is_preserved(
    service: InteractionIngestionService, queue: InMemoryInteractionEventQueue
) -> None:
    moment = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    service.ingest(InteractionRequest(user_id=1, event_type="click", item_id=1, occurred_at=moment))
    consumed: list = []
    queue.consume(lambda event: (consumed.append(event), True)[1], max_events=1)
    assert consumed[0].occurred_at == moment


def test_an_interaction_without_a_subject_is_rejected(
    service: InteractionIngestionService,
) -> None:
    """Fail Fast on an invariant the schema cannot express."""
    with pytest.raises(ValueError, match="must reference"):
        service.ingest(InteractionRequest(user_id=1, event_type="click"))


def test_a_broker_failure_still_acknowledges_the_client(
    service: InteractionIngestionService,
) -> None:
    """Table 10: EventPublishError does not affect the response."""
    from tests.messaging.test_messaging import BrokenQueue

    broken = InteractionIngestionService(BrokenQueue())
    accepted = broken.ingest(InteractionRequest(user_id=1, event_type="click", item_id=1))
    assert accepted.accepted is True
    assert accepted.queued is False


def test_an_unexpected_broker_error_is_also_absorbed() -> None:
    class ExplodingQueue(InMemoryInteractionEventQueue):
        def publish(self, event) -> None:  # noqa: ANN001
            raise RuntimeError("unexpected broker failure")

    accepted = InteractionIngestionService(ExplodingQueue()).ingest(
        InteractionRequest(user_id=1, event_type="click", item_id=1)
    )
    assert accepted.accepted is True
    assert accepted.queued is False


def test_it_does_not_write_to_the_database() -> None:
    """Section 5.2: the consuming side is FeedbackHistoryService, not this."""
    import inspect

    source = inspect.getsource(InteractionIngestionService)
    assert "Repository" not in source
