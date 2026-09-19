"""ModelFactory and the registry (Section 3.3, Section 9.5).

The point these tests defend is the Open/Closed claim: adding a strategy must
not require editing the factory. They therefore register a new strategy at
runtime and assert the factory resolves it without any change to its code.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from ai.interfaces import IRecommendationStrategy, ScoredItem, UserFeatures
from ai.model_factory import ModelFactory, register_strategy
from config.settings import Settings
from errors import ModelUnavailableError

FACTORY_SOURCE = Path(__file__).resolve().parents[2] / "src" / "ai" / "model_factory.py"


def test_the_four_documented_strategies_are_registered() -> None:
    """Table 9 names four strategies; all four must be resolvable."""
    assert set(ModelFactory.available()) >= {"als", "content", "ncf", "popularity"}


def test_adding_a_strategy_does_not_require_editing_the_factory() -> None:
    """The Open/Closed claim of Section 3.3, exercised rather than asserted."""

    @register_strategy("transformer-test")
    class TransformerRecommendationStrategy(IRecommendationStrategy):
        def predict(self, user_features: UserFeatures, k: int = 10) -> list[ScoredItem]:
            return [ScoredItem(item_id=1, score=1.0)]

    try:
        assert "transformer-test" in ModelFactory.available()
        resolved = ModelFactory.create("transformer-test")
        assert isinstance(resolved, TransformerRecommendationStrategy)
        assert resolved.strategy_id == "transformer-test"
    finally:
        ModelFactory.registry()  # ensure the package is loaded before mutating
        from ai import model_factory

        model_factory._REGISTRY.pop("transformer-test", None)  # noqa: SLF001
        ModelFactory.clear_cache()


def test_the_factory_contains_no_branch_per_strategy() -> None:
    """Rule of Section 9.5: resolution is a lookup, never an if/elif chain.

    The check is structural: no comparison in this module may mention a
    strategy identifier, which is what an ``if strategy_id == "als"`` ladder
    would look like.
    """
    tree = ast.parse(FACTORY_SOURCE.read_text(encoding="utf-8"))
    identifiers = set(ModelFactory.available())
    offending: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for literal in ast.walk(node):
                if isinstance(literal, ast.Constant) and literal.value in identifiers:
                    offending.append(f"line {node.lineno}: comparison against {literal.value!r}")
    assert not offending, "ModelFactory branches per strategy:\n  " + "\n  ".join(offending)


def test_an_unknown_identifier_is_a_model_unavailable_error() -> None:
    with pytest.raises(ModelUnavailableError) as excinfo:
        ModelFactory.create("does-not-exist")
    assert excinfo.value.code == "model_unavailable"
    assert excinfo.value.degradable is True


def test_registering_two_classes_under_one_identifier_is_rejected() -> None:
    @register_strategy("duplicate-test")
    class First(IRecommendationStrategy):
        def predict(self, user_features: UserFeatures, k: int = 10) -> list[ScoredItem]:
            return []

    try:
        with pytest.raises(ValueError, match="already registered"):

            @register_strategy("duplicate-test")
            class Second(IRecommendationStrategy):
                def predict(self, user_features: UserFeatures, k: int = 10) -> list[ScoredItem]:
                    return []

    finally:
        from ai import model_factory

        model_factory._REGISTRY.pop("duplicate-test", None)  # noqa: SLF001


def test_registering_a_non_strategy_is_rejected() -> None:
    with pytest.raises(TypeError):

        @register_strategy("not-a-strategy")
        class NotAStrategy:  # type: ignore[misc]
            pass


def test_an_empty_identifier_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):

        @register_strategy("")
        class Anonymous(IRecommendationStrategy):
            def predict(self, user_features: UserFeatures, k: int = 10) -> list[ScoredItem]:
                return []


def test_instances_are_reused_so_weights_load_once(settings: Settings) -> None:
    """Loading artifacts is the expensive part; the request path must not pay it."""
    first = ModelFactory.create("popularity", settings)
    second = ModelFactory.create("popularity", settings)
    assert first is second

    third = ModelFactory.create("popularity", settings, use_cache=False)
    assert third is not first


def test_clear_cache_forces_a_rebuild(settings: Settings) -> None:
    first = ModelFactory.create("popularity", settings)
    ModelFactory.clear_cache()
    assert ModelFactory.create("popularity", settings) is not first


def test_the_active_strategy_comes_from_configuration(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    from config.settings import load_settings

    monkeypatch.setenv("RECO_ACTIVE_STRATEGY", "popularity")
    reloaded = load_settings()
    assert reloaded.ACTIVE_STRATEGY == "popularity"
    assert ModelFactory.create(settings=reloaded).strategy_id == "popularity"
