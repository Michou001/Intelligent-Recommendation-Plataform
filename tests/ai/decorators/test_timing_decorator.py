"""TimingDecorator (Table 4, NFR-01)."""

from __future__ import annotations

import pytest

from ai.decorators import current_context, operational_context
from ai.decorators.timing_decorator import TimingDecorator
from ai.interfaces import UserFeatures
from errors import ModelUnavailableError
from tests.conftest import FailingStrategy, StubStrategy


def test_it_records_one_sample_per_prediction(
    stub_strategy: StubStrategy, user_features: UserFeatures
) -> None:
    decorated = TimingDecorator(stub_strategy)
    for _ in range(5):
        decorated.predict(user_features, 3)
    assert len(decorated.samples) == 5
    assert all(sample >= 0 for sample in decorated.samples)


def test_it_publishes_the_latency_to_the_operational_context(
    stub_strategy: StubStrategy, user_features: UserFeatures
) -> None:
    decorated = TimingDecorator(stub_strategy)
    with operational_context():
        decorated.predict(user_features, 3)
        assert "latency_ms" in current_context()


def test_a_failed_prediction_is_still_measured(user_features: UserFeatures) -> None:
    """A slow failure is exactly what an operator needs to see."""
    decorated = TimingDecorator(FailingStrategy())
    with pytest.raises(ModelUnavailableError):
        decorated.predict(user_features, 3)
    assert len(decorated.samples) == 1


def test_the_percentile_is_computed_over_the_window(
    stub_strategy: StubStrategy, user_features: UserFeatures
) -> None:
    decorated = TimingDecorator(stub_strategy)
    assert decorated.percentile(95) is None
    for _ in range(100):
        decorated.predict(user_features, 3)
    p95 = decorated.percentile(95)
    assert p95 is not None
    assert decorated.percentile(50) <= p95 <= max(decorated.samples)


def test_the_window_is_bounded(stub_strategy: StubStrategy, user_features: UserFeatures) -> None:
    decorated = TimingDecorator(stub_strategy, window=10)
    for _ in range(50):
        decorated.predict(user_features, 3)
    assert len(decorated.samples) == 10


def test_reset_discards_the_samples(
    stub_strategy: StubStrategy, user_features: UserFeatures
) -> None:
    decorated = TimingDecorator(stub_strategy)
    decorated.predict(user_features, 3)
    decorated.reset()
    assert decorated.samples == ()


def test_it_forwards_the_identity_of_the_wrapped_strategy(stub_strategy: StubStrategy) -> None:
    decorated = TimingDecorator(stub_strategy)
    assert decorated.strategy_id == stub_strategy.strategy_id
    assert decorated.model_version == stub_strategy.model_version
    assert decorated.unwrap() is stub_strategy
