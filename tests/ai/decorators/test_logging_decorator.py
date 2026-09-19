"""LoggingDecorator, and the separation of the two kinds of logging.

Table 6 records the conflation of operational logging with business history
as a Milestone 1 defect. These tests hold that correction in place: the
decorator emits operational fields and nothing about recommendation history
persistence.
"""

from __future__ import annotations

from typing import Any

import pytest

from ai.decorators import build_chain
from ai.decorators.caching_decorator import CachingDecorator, InMemoryCacheBackend
from ai.decorators.logging_decorator import EVENT_NAME, LoggingDecorator
from ai.decorators.timing_decorator import TimingDecorator
from ai.interfaces import UserFeatures
from errors import ModelUnavailableError
from tests.conftest import FailingStrategy, StubStrategy


class RecordingLogger:
    """Captures the structured records instead of writing them."""

    def __init__(self) -> None:
        self.records: list[tuple[str, str, dict[str, Any]]] = []

    def _record(self, level: str):  # noqa: ANN202 - test double
        def emit(event: str, **fields: Any) -> None:
            self.records.append((level, event, fields))

        return emit

    def __getattr__(self, name: str):  # noqa: ANN204 - any level is accepted
        return self._record(name)

    def fields_of(self, event: str) -> dict[str, Any]:
        for _level, name, fields in self.records:
            if name == event:
                return fields
        raise AssertionError(f"no record for {event!r}: {[r[1] for r in self.records]}")


def test_it_emits_one_record_per_served_request(
    stub_strategy: StubStrategy, user_features: UserFeatures
) -> None:
    logger = RecordingLogger()
    LoggingDecorator(stub_strategy, logger).predict(user_features, 3)
    served = [r for r in logger.records if r[1] == EVENT_NAME]
    assert len(served) == 1
    assert served[0][0] == "info"


def test_the_record_carries_the_documented_operational_fields(
    stub_strategy: StubStrategy, user_features: UserFeatures
) -> None:
    """Section 9.9: which strategy ran, cache hit, latency, model version."""
    logger = RecordingLogger()
    chain = LoggingDecorator(
        TimingDecorator(CachingDecorator(stub_strategy, InMemoryCacheBackend())), logger
    )
    chain.predict(user_features, 3)

    fields = logger.fields_of(EVENT_NAME)
    assert fields["strategy_id"] == "stub"
    assert fields["model_version"] == "stub-v1"
    assert fields["cache_hit"] is False
    assert "latency_ms" in fields
    assert fields["user_id"] == user_features.user_id


def test_a_cache_hit_is_visible_in_the_record(
    stub_strategy: StubStrategy, user_features: UserFeatures
) -> None:
    logger = RecordingLogger()
    chain = LoggingDecorator(
        TimingDecorator(CachingDecorator(stub_strategy, InMemoryCacheBackend())), logger
    )
    chain.predict(user_features, 3)
    chain.predict(user_features, 3)
    hits = [f for _l, name, f in logger.records if name == EVENT_NAME and f.get("cache_hit")]
    assert len(hits) == 1


def test_a_failure_is_logged_at_error_level_and_re_raised(
    user_features: UserFeatures,
) -> None:
    logger = RecordingLogger()
    decorated = LoggingDecorator(FailingStrategy(), logger)
    with pytest.raises(ModelUnavailableError):
        decorated.predict(user_features, 3)
    level, _event, fields = next(r for r in logger.records if r[1] == EVENT_NAME)
    assert level == "error"
    assert fields["outcome"] == "failed"


def test_scores_are_debug_material_not_info(
    stub_strategy: StubStrategy, user_features: UserFeatures
) -> None:
    logger = RecordingLogger()
    LoggingDecorator(stub_strategy, logger).predict(user_features, 3)
    scores = [r for r in logger.records if r[1] == "recommendation.scores"]
    assert scores and scores[0][0] == "debug"


def test_it_does_not_write_business_history(
    stub_strategy: StubStrategy, user_features: UserFeatures
) -> None:
    """FR-06 belongs to FeedbackHistoryService, not to this decorator."""
    import inspect

    source = inspect.getsource(LoggingDecorator)
    for forbidden in ("Repository", "repository", "history", "feedback"):
        assert forbidden not in source, f"LoggingDecorator mentions {forbidden!r}"


def test_the_chain_is_built_innermost_first(stub_strategy: StubStrategy) -> None:
    """Section 9.6: LoggingDecorator(TimingDecorator(CachingDecorator(base)))."""
    chain = build_chain(stub_strategy, ["caching", "timing", "logging"])
    assert isinstance(chain, LoggingDecorator)
    assert isinstance(chain.wrapped, TimingDecorator)
    assert isinstance(chain.wrapped.wrapped, CachingDecorator)
    assert chain.unwrap() is stub_strategy


def test_the_chain_order_is_configurable(stub_strategy: StubStrategy) -> None:
    """Decorators are composable: the order comes from configuration."""
    chain = build_chain(stub_strategy, ["logging", "caching"])
    assert isinstance(chain, CachingDecorator)
    assert isinstance(chain.wrapped, LoggingDecorator)


def test_an_empty_chain_returns_the_base_strategy(stub_strategy: StubStrategy) -> None:
    assert build_chain(stub_strategy, []) is stub_strategy


def test_an_unknown_decorator_fails_at_startup(stub_strategy: StubStrategy) -> None:
    with pytest.raises(ValueError, match="unknown decorator"):
        build_chain(stub_strategy, ["nonexistent"])
