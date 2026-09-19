"""InteractionEventQueue - the message broker of FR-02 and FR-06.

``InteractionIngestionService`` publishes interaction events here and returns
immediately, so ingestion never blocks navigation (FR-02), and
``FeedbackHistoryService`` consumes them on a separate process to write the
business history (FR-06). Section 5.3 is explicit that this path is
guaranteed by the broker rather than by the request thread, which is the
component that Milestone 1 described as behaviour without making anything
responsible for it.

``InteractionEventQueue`` is declared as an abstract base class so the
application layer depends on the contract. RabbitMQ is the backend of
Table 7; an in-process backend implements the same contract when no broker is
available, which keeps the asynchronous path exercisable in development and
in the tests.
"""

from __future__ import annotations

import contextlib
import json
import queue
import threading
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from config.settings import Settings, load_settings
from errors import EventPublishError


@dataclass(frozen=True)
class InteractionEvent:
    """One user interaction, and the recommendation it was shown with.

    ``event_id`` travels with the event so the consumer can be idempotent: a
    redelivery carries the same identifier and is ignored downstream
    (Section 9.9).
    """

    user_id: int
    event_type: str
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    occurred_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    item_id: int | None = None
    recommended_items: tuple[int, ...] = ()
    model_version: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def to_json(self) -> str:
        payload = asdict(self)
        payload["occurred_at"] = self.occurred_at.isoformat()
        payload["recommended_items"] = list(self.recommended_items)
        return json.dumps(payload, default=str)

    @classmethod
    def from_json(cls, payload: str) -> InteractionEvent:
        data = json.loads(payload)
        raw_timestamp = str(data.get("occurred_at") or "")
        try:
            occurred_at = datetime.fromisoformat(raw_timestamp)
        except ValueError:
            occurred_at = datetime.now(UTC)
        return cls(
            user_id=int(data["user_id"]),
            event_type=str(data.get("event_type", "unknown")),
            event_id=str(data.get("event_id", uuid.uuid4())),
            occurred_at=occurred_at,
            item_id=data.get("item_id"),
            recommended_items=tuple(int(i) for i in data.get("recommended_items", [])),
            model_version=data.get("model_version"),
            metadata={str(k): str(v) for k, v in dict(data.get("metadata", {})).items()},
        )


class InteractionEventQueue(ABC):
    """Contract of the message broker that carries interaction events."""

    @abstractmethod
    def publish(self, event: InteractionEvent) -> None:
        """Publish one event.

        Raises:
            EventPublishError: The event could not be handed to the broker.
                Implementations write it to the local retry buffer first, so
                the caller can ignore the failure without losing the event
                (Table 10).
        """

    @abstractmethod
    def consume(
        self,
        handler: Callable[[InteractionEvent], bool],
        *,
        max_events: int | None = None,
        timeout: float | None = None,
    ) -> int:
        """Deliver events to ``handler`` until the limit or the timeout.

        The handler returns True when the event was processed; returning
        False, or raising, leaves the event for redelivery.

        Returns:
            Number of events the handler accepted.
        """

    @abstractmethod
    def describe(self) -> str:
        """Backend name, reported by /health."""

    def pending(self) -> int | None:
        """Number of events waiting, when the backend can report it."""
        return None


class RetryBuffer:
    """Local append-only buffer for events the broker refused.

    Table 10 requires an unpublishable event to land here instead of
    affecting the response. The buffer is replayed by ``drain`` once the
    broker is reachable again.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def append(self, event: InteractionEvent) -> None:
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(event.to_json() + "\n")

    def read_all(self) -> list[InteractionEvent]:
        if not self._path.exists():
            return []
        events: list[InteractionEvent] = []
        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    try:
                        events.append(InteractionEvent.from_json(line))
                    except (ValueError, KeyError):
                        continue
        return events

    def clear(self) -> None:
        with self._lock:
            if self._path.exists():
                self._path.unlink()

    def __len__(self) -> int:
        return len(self.read_all())


class InMemoryInteractionEventQueue(InteractionEventQueue):
    """In-process backend used when no broker is configured.

    It preserves the property the design depends on - publication does not
    run the consumer, so it never adds latency to the response - while making
    the asynchronous path testable without infrastructure.
    """

    def __init__(self, retry_buffer: RetryBuffer | None = None) -> None:
        self._queue: queue.Queue[InteractionEvent] = queue.Queue()
        self._retry_buffer = retry_buffer

    def publish(self, event: InteractionEvent) -> None:
        try:
            self._queue.put_nowait(event)
        except Exception as exc:  # noqa: BLE001 - translated into the platform vocabulary
            if self._retry_buffer is not None:
                self._retry_buffer.append(event)
            raise EventPublishError(
                "the interaction event could not be queued",
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc

    def consume(
        self,
        handler: Callable[[InteractionEvent], bool],
        *,
        max_events: int | None = None,
        timeout: float | None = None,
    ) -> int:
        accepted = 0
        while max_events is None or accepted < max_events:
            try:
                event = self._queue.get(timeout=timeout if timeout is not None else 0.05)
            except queue.Empty:
                break
            try:
                if handler(event):
                    accepted += 1
                else:
                    self._queue.put_nowait(event)
                    break
            except Exception:
                # Redelivery: the event returns to the queue and the worker
                # decides whether to retry (Section 5.2).
                self._queue.put_nowait(event)
                raise
        return accepted

    def describe(self) -> str:
        return "in-memory"

    def pending(self) -> int | None:
        return self._queue.qsize()


class RabbitMqInteractionEventQueue(InteractionEventQueue):
    """RabbitMQ backend of Table 7, reached through pika."""

    def __init__(
        self, connection: Any, queue_name: str, retry_buffer: RetryBuffer | None = None
    ) -> None:
        self._connection = connection
        self._queue_name = queue_name
        self._retry_buffer = retry_buffer
        self._channel = connection.channel()
        self._channel.queue_declare(queue=queue_name, durable=True)

    @classmethod
    def connect(
        cls, url: str, queue_name: str, retry_buffer: RetryBuffer | None = None
    ) -> RabbitMqInteractionEventQueue:
        """Open a broker connection. Raises when pika or the broker is absent."""
        import pika

        parameters = pika.URLParameters(url)
        parameters.socket_timeout = 1.0
        return cls(pika.BlockingConnection(parameters), queue_name, retry_buffer)

    def publish(self, event: InteractionEvent) -> None:
        import pika

        try:
            self._channel.basic_publish(
                exchange="",
                routing_key=self._queue_name,
                body=event.to_json().encode("utf-8"),
                properties=pika.BasicProperties(delivery_mode=2, message_id=event.event_id),
            )
        except Exception as exc:  # noqa: BLE001 - translated into the platform vocabulary
            if self._retry_buffer is not None:
                self._retry_buffer.append(event)
            raise EventPublishError(
                "the interaction event could not be published to the broker",
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc

    def consume(
        self,
        handler: Callable[[InteractionEvent], bool],
        *,
        max_events: int | None = None,
        timeout: float | None = None,
    ) -> int:
        accepted = 0
        while max_events is None or accepted < max_events:
            method, _properties, body = self._channel.basic_get(self._queue_name, auto_ack=False)
            if method is None:
                break
            event = InteractionEvent.from_json(body.decode("utf-8"))
            try:
                processed = handler(event)
            except Exception:
                self._channel.basic_nack(method.delivery_tag, requeue=True)
                raise
            if processed:
                self._channel.basic_ack(method.delivery_tag)
                accepted += 1
            else:
                self._channel.basic_nack(method.delivery_tag, requeue=True)
                break
        return accepted

    def describe(self) -> str:
        return "rabbitmq"

    def close(self) -> None:
        """Close the broker connection."""
        # Closing must never raise on shutdown.
        with contextlib.suppress(Exception):
            self._connection.close()


def build_event_queue(settings: Settings | None = None) -> InteractionEventQueue:
    """Select the broker backend from configuration.

    RabbitMQ when a URL is configured and reachable, otherwise the in-process
    backend behind the same interface.
    """
    settings = settings or load_settings()
    buffer = RetryBuffer(settings.messaging.retry_buffer_path)
    if settings.messaging.broker_url:
        try:
            return RabbitMqInteractionEventQueue.connect(
                settings.messaging.broker_url, settings.messaging.queue_name, buffer
            )
        except Exception:  # noqa: BLE001 - absent infrastructure is not an error here
            pass
    return InMemoryInteractionEventQueue(retry_buffer=buffer)


__all__ = [
    "InMemoryInteractionEventQueue",
    "InteractionEvent",
    "InteractionEventQueue",
    "RabbitMqInteractionEventQueue",
    "RetryBuffer",
    "build_event_queue",
]
