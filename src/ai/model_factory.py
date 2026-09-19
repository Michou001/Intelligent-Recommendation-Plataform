"""ModelFactory and the strategy registry (Factory pattern, Table 4, FR-04).

The factory does not contain a switch over strategy types. It resolves a
strategy through a registry - a map from strategy identifier to class - that
each strategy populates when its module is loaded, and the active identifier
is read from configuration (Section 3.3, Section 9.5).

Adding a fifth strategy is therefore a new file plus its ``@register_strategy``
line: no existing branch of this module is edited, which is the Open/Closed
Principle expressed in the one place that would otherwise have to change.
"""

from __future__ import annotations

import importlib
import sys
import threading
from collections.abc import Callable, Mapping

from ai.interfaces import IRecommendationStrategy
from config.settings import Settings, load_settings
from errors import ModelUnavailableError

#: Registry populated by the decorator below. Keys are the identifiers used in
#: ``config/settings.py``; values are the classes that realise them.
_REGISTRY: dict[str, type[IRecommendationStrategy]] = {}

#: Module imported to trigger the registration of the bundled strategies.
_STRATEGIES_PACKAGE = "ai.strategies"

_lock = threading.Lock()
_strategies_loaded = False


def register_strategy(
    identifier: str,
) -> Callable[[type[IRecommendationStrategy]], type[IRecommendationStrategy]]:
    """Register a strategy class under a string identifier.

    Usage::

        @register_strategy("als")
        class CollaborativeFilteringStrategy(IRecommendationStrategy):
            ...

    Args:
        identifier: Key the factory resolves, as written in configuration.

    Raises:
        ValueError: The identifier is empty, or already taken by another class.
    """

    def decorator(cls: type[IRecommendationStrategy]) -> type[IRecommendationStrategy]:
        if not identifier:
            raise ValueError("a strategy identifier cannot be empty")
        existing = _REGISTRY.get(identifier)
        if existing is not None and not _is_same_class(existing, cls):
            raise ValueError(
                f"strategy identifier {identifier!r} is already registered by {existing.__name__}"
            )
        if not issubclass(cls, IRecommendationStrategy):
            raise TypeError(f"{cls.__name__} does not implement IRecommendationStrategy")
        cls.strategy_id = identifier
        _REGISTRY[identifier] = cls
        return cls

    return decorator


def _is_same_class(existing: type, candidate: type) -> bool:
    """Whether two class objects are the same declaration.

    Identity is not enough: running a strategy module with ``python -m``
    executes it a second time under the name ``__main__``, which produces a
    distinct class object for the same source declaration. Treating that as a
    conflict would make the offline training entry points unusable, while
    comparing the qualified name still rejects two different strategies
    claiming the same identifier.
    """
    if existing is candidate:
        return True
    if existing.__qualname__ != candidate.__qualname__:
        return False
    return _defining_file(existing) == _defining_file(candidate)


def _defining_file(cls: type) -> str | None:
    """Path of the source file a class was declared in, when resolvable."""
    module = sys.modules.get(cls.__module__)
    return getattr(module, "__file__", None)


def _ensure_strategies_loaded() -> None:
    """Import the strategies package once, so the registry is populated.

    The import is deferred to call time instead of being written at module
    level, so this module never depends on the strategy modules that depend
    on it.
    """
    global _strategies_loaded
    if _strategies_loaded:
        return
    with _lock:
        if _strategies_loaded:
            return
        importlib.import_module(_STRATEGIES_PACKAGE)
        _strategies_loaded = True


class ModelFactory:
    """Creates, and reuses, the configured recommendation strategy.

    Creating a strategy is not free: it loads the offline artifacts of the
    selected feature-store version. The factory therefore caches one instance
    per (identifier, model version) pair, so the request path never pays that
    cost (Section 9.6: the composition is built once at startup).
    """

    _instances: dict[tuple[str, str], IRecommendationStrategy] = {}

    @staticmethod
    def available() -> tuple[str, ...]:
        """Identifiers currently present in the registry."""
        _ensure_strategies_loaded()
        return tuple(sorted(_REGISTRY))

    @staticmethod
    def registry() -> Mapping[str, type[IRecommendationStrategy]]:
        """Read-only view of the registry, used by the tests and by /health."""
        _ensure_strategies_loaded()
        return dict(_REGISTRY)

    @classmethod
    def create(
        cls,
        strategy_id: str | None = None,
        settings: Settings | None = None,
        *,
        use_cache: bool = True,
    ) -> IRecommendationStrategy:
        """Resolve and build a strategy by registry lookup.

        Args:
            strategy_id: Registry identifier. Defaults to the active strategy
                declared in configuration.
            settings: Configuration to build the strategy from.
            use_cache: Reuse a previously built instance for the same
                identifier and model version.

        Raises:
            ModelUnavailableError: The identifier is not registered, or the
                strategy failed to load its artifacts. The caller degrades to
                the popularity fallback (NFR-05).
        """
        settings = settings or load_settings()
        strategy_id = strategy_id or settings.model.active_strategy
        _ensure_strategies_loaded()

        strategy_class = _REGISTRY.get(strategy_id)
        if strategy_class is None:
            raise ModelUnavailableError(
                f"no recommendation strategy is registered under {strategy_id!r}",
                detail=f"registered: {', '.join(sorted(_REGISTRY)) or 'none'}",
            )

        key = (strategy_id, settings.model.model_version)
        if use_cache and key in cls._instances:
            return cls._instances[key]

        try:
            instance = strategy_class.build(settings)
        except ModelUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001 - translated into the platform vocabulary
            raise ModelUnavailableError(
                f"strategy {strategy_id!r} could not be loaded",
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc

        if use_cache:
            cls._instances[key] = instance
        return instance

    @classmethod
    def clear_cache(cls) -> None:
        """Drop the cached instances, for example after publishing new weights."""
        cls._instances.clear()


__all__ = ["ModelFactory", "register_strategy"]
