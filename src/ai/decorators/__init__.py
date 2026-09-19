"""Operational decorators wrapping IRecommendationStrategy (Decorator, Table 4).

The three decorators add caching, latency measurement, and operational
logging without touching the prediction logic, and they compose in any order,
which is Composition over Inheritance applied to the inference path
(Section 3.5).

What they do **not** do is record business history: the recommendations shown
and the user's feedback are written by ``FeedbackHistoryService`` through
``IInteractionHistoryRepository`` (FR-06). Table 4 makes that separation
explicit and it is preserved here.

This module holds what the three decorators share: the thin base class and
the per-request operational context they write their fields into, so that
``LoggingDecorator`` can emit one record describing the whole chain.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from ai.interfaces import IRecommendationStrategy, ScoredItem, UserFeatures

#: Fields collected during one prediction. A ContextVar rather than an
#: attribute, so concurrent requests served by the same decorator instance do
#: not overwrite each other's measurements.
_context: ContextVar[dict[str, Any] | None] = ContextVar("operational_context", default=None)


@contextmanager
def operational_context() -> Iterator[dict[str, Any]]:
    """Open a per-prediction context the decorators contribute fields to."""
    existing = _context.get()
    if existing is not None:
        # An outer decorator already opened one; contribute to it.
        yield existing
        return
    fields: dict[str, Any] = {}
    token = _context.set(fields)
    try:
        yield fields
    finally:
        _context.reset(token)


def record(**fields: Any) -> None:
    """Add fields to the current operational context, if one is open."""
    current = _context.get()
    if current is not None:
        current.update(fields)


def current_context() -> Mapping[str, Any]:
    """Read the fields collected so far during this prediction."""
    return dict(_context.get() or {})


class StrategyDecorator(IRecommendationStrategy):
    """Base of the three operational decorators.

    Holds the wrapped strategy and forwards the identity of the decorated
    object, so a chain reports the identifier and the model version of the
    concrete strategy at its centre rather than of the wrapper.
    """

    def __init__(self, wrapped: IRecommendationStrategy) -> None:
        if not isinstance(wrapped, IRecommendationStrategy):
            raise TypeError("a decorator can only wrap an IRecommendationStrategy")
        self._wrapped = wrapped

    @property
    def wrapped(self) -> IRecommendationStrategy:
        """The strategy this decorator adds behaviour to."""
        return self._wrapped

    @property
    def strategy_id(self) -> str:  # type: ignore[override]
        return self._wrapped.strategy_id

    @property
    def model_version(self) -> str:
        return self._wrapped.model_version

    def describe(self) -> Mapping[str, str]:
        return self._wrapped.describe()

    def predict(self, user_features: UserFeatures, k: int = 10) -> list[ScoredItem]:
        return self._wrapped.predict(user_features, k)

    def unwrap(self) -> IRecommendationStrategy:
        """Return the innermost strategy, skipping every decorator."""
        target: IRecommendationStrategy = self
        while isinstance(target, StrategyDecorator):
            target = target.wrapped
        return target


#: Registry of decorator builders, keyed by the identifiers used in
#: ``settings.model.decorator_chain``. Like the strategy registry, it means a
#: new decorator is added without editing the code that builds the chain.
_DECORATORS: dict[str, Callable[..., StrategyDecorator]] = {}


def register_decorator(
    identifier: str,
) -> Callable[[type[StrategyDecorator]], type[StrategyDecorator]]:
    """Register a decorator class under the identifier used in configuration."""

    def wrapper(cls: type[StrategyDecorator]) -> type[StrategyDecorator]:
        existing = _DECORATORS.get(identifier)
        if existing is not None and existing is not cls:
            raise ValueError(f"decorator identifier {identifier!r} is already registered")
        _DECORATORS[identifier] = cls
        return cls

    return wrapper


def build_chain(
    base: IRecommendationStrategy,
    chain: list[str] | tuple[str, ...],
    **kwargs: Any,
) -> IRecommendationStrategy:
    """Wrap ``base`` with the named decorators, innermost first.

    ``["caching", "timing", "logging"]`` produces
    ``LoggingDecorator(TimingDecorator(CachingDecorator(base)))``, which is the
    composition written in Section 9.6. An unknown identifier is an error at
    startup, not a silently skipped wrapper.
    """
    from ai.decorators import (
        caching_decorator,  # noqa: F401  (registration)
        logging_decorator,  # noqa: F401  (registration)
        timing_decorator,  # noqa: F401  (registration)
    )

    decorated = base
    for identifier in chain:
        builder = _DECORATORS.get(identifier)
        if builder is None:
            raise ValueError(
                f"unknown decorator {identifier!r}; registered: {', '.join(sorted(_DECORATORS))}"
            )
        decorated = builder(decorated, **kwargs.get(identifier, {}))
    return decorated


def available_decorators() -> tuple[str, ...]:
    """Identifiers currently present in the decorator registry."""
    from ai.decorators import (
        caching_decorator,  # noqa: F401
        logging_decorator,  # noqa: F401
        timing_decorator,  # noqa: F401
    )

    return tuple(sorted(_DECORATORS))


__all__ = [
    "StrategyDecorator",
    "available_decorators",
    "build_chain",
    "current_context",
    "operational_context",
    "record",
    "register_decorator",
]
