"""TimingDecorator - latency measurement (Table 4, NFR-01).

Measures how long the wrapped object takes to answer and publishes the value
into the operational context, where ``LoggingDecorator`` picks it up. It also
keeps a bounded sample of recent latencies so the service can report its own
p95 without an external collector, which is what makes compliance with NFR-01
a measured value rather than an assumption (Section 9.7).

The decorator measures whatever it wraps. Placed outside ``CachingDecorator``
it measures the served latency including cache hits; placed inside it, it
measures the inference alone.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Sequence

import numpy as np

from ai.decorators import StrategyDecorator, operational_context, record, register_decorator
from ai.interfaces import IRecommendationStrategy, ScoredItem, UserFeatures

#: Number of recent measurements kept in memory for the percentile report.
DEFAULT_WINDOW = 2048


@register_decorator("timing")
class TimingDecorator(StrategyDecorator):
    """Times the wrapped prediction and records the elapsed milliseconds."""

    def __init__(
        self,
        wrapped: IRecommendationStrategy,
        *,
        window: int = DEFAULT_WINDOW,
    ) -> None:
        super().__init__(wrapped)
        self._samples: deque[float] = deque(maxlen=window)

    def predict(self, user_features: UserFeatures, k: int = 10) -> list[ScoredItem]:
        with operational_context():
            started = time.perf_counter()
            try:
                return self._wrapped.predict(user_features, k)
            finally:
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                self._samples.append(elapsed_ms)
                record(latency_ms=round(elapsed_ms, 3))

    @property
    def samples(self) -> Sequence[float]:
        """Recent latencies in milliseconds, oldest first."""
        return tuple(self._samples)

    def percentile(self, percentile: float = 95.0) -> float | None:
        """Return a latency percentile over the current window, in milliseconds."""
        if not self._samples:
            return None
        return float(np.percentile(np.fromiter(self._samples, dtype=np.float64), percentile))

    def reset(self) -> None:
        """Discard the collected samples."""
        self._samples.clear()


__all__ = ["TimingDecorator"]
