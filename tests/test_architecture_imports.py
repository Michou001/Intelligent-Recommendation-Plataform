"""The strict import rule of Section 9.2, verified by analysing the imports.

    "a strict import rule, verified by a test, that forbids any module in an
     outer layer from importing a concrete class of an inner layer, since
     dependencies must travel through the interfaces of Section 3.2"

The rule implemented here reads: a module may import from another layer only

* a contract module (``ai/interfaces.py``, ``persistence/interfaces.py``,
  ``presentation/schemas.py``, ``config/settings.py``, ``errors.py``), or
* an abstract base class, an exception type, or an immutable data transfer
  object (a frozen dataclass or a Pydantic model).

Anything else - a concrete adapter, a service, a decorator, a strategy - is a
violation. The single exception is a composition root, which has to name
concrete classes because wiring them is its entire job; the exempt set is
declared below and asserted to be exactly what it claims.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect
from pathlib import Path

import pytest
from pydantic import BaseModel

SRC = Path(__file__).resolve().parents[1] / "src"

#: Top-level packages, one per architecture layer.
LAYERS = {"presentation", "application", "ai", "data", "persistence", "messaging", "config"}

#: Modules any layer may import in full: they *are* the contracts.
CONTRACT_MODULES = {
    "errors",
    "config",
    "config.settings",
    "ai.interfaces",
    "persistence.interfaces",
    "presentation.schemas",
}

#: Composition roots. A composition root wires concrete implementations
#: together; that is what makes every other module able to depend on
#: abstractions only. ``"*"`` exempts the whole module, otherwise only the
#: named functions are exempt.
COMPOSITION_ROOTS: dict[str, set[str] | str] = {
    # Section 9.6 puts the startup composition in the FastAPI module.
    "application.api": "*",
    # Process entry point of the FR-06 worker.
    "messaging.feedback_history_service": {"main"},
}


def _module_name(path: Path) -> str:
    relative = path.relative_to(SRC).with_suffix("")
    parts = [part for part in relative.parts if part != "__init__"]
    return ".".join(parts)


def _layer_of(module: str) -> str | None:
    root = module.split(".", 1)[0]
    return root if root in LAYERS else None


def _source_modules() -> list[tuple[str, Path]]:
    return sorted(
        (_module_name(path), path) for path in SRC.rglob("*.py") if "__pycache__" not in path.parts
    )


def _imports_of(path: Path) -> list[tuple[str, tuple[str, ...], str | None, int]]:
    """Return (module, imported names, enclosing function, line) per import."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[tuple[str, tuple[str, ...], str | None, int]] = []

    def walk(node: ast.AST, function: str | None) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                walk(child, function or child.name)
                continue
            if isinstance(child, ast.ImportFrom) and child.module and child.level == 0:
                found.append(
                    (child.module, tuple(a.name for a in child.names), function, child.lineno)
                )
            elif isinstance(child, ast.Import):
                for alias in child.names:
                    found.append((alias.name, (), function, child.lineno))
            else:
                walk(child, function)

    walk(tree, None)
    return found


def _is_exempt(module: str, function: str | None) -> bool:
    rule = COMPOSITION_ROOTS.get(module)
    if rule is None:
        return False
    if rule == "*":
        return True
    return function is not None and function in rule


def _is_allowed_symbol(target_module: str, name: str) -> tuple[bool, str]:
    """Whether importing ``name`` across a layer boundary is permitted."""
    try:
        imported = importlib.import_module(target_module)
    except ImportError as exc:  # pragma: no cover - a broken import fails elsewhere
        return False, f"module not importable: {exc}"

    obj = getattr(imported, name, None)
    if obj is None:
        return True, "not a class (constant, alias or submodule)"
    if not inspect.isclass(obj):
        return True, "not a class"
    if issubclass(obj, BaseException):
        return True, "exception type"
    if inspect.isabstract(obj) or getattr(obj, "__abstractmethods__", None):
        return True, "abstract base class"
    if dataclasses.is_dataclass(obj) and obj.__dataclass_params__.frozen:
        return True, "immutable data transfer object"
    if issubclass(obj, BaseModel):
        return True, "pydantic data transfer object"
    return False, "concrete class"


def _violations() -> list[str]:
    problems: list[str] = []
    for module, path in _source_modules():
        layer = _layer_of(module)
        if layer is None:
            continue
        for target, names, function, line in _imports_of(path):
            target_layer = _layer_of(target)
            if target_layer is None or target_layer == layer:
                continue
            if target in CONTRACT_MODULES:
                continue
            if _is_exempt(module, function):
                continue
            if not names:
                problems.append(
                    f"{module}:{line} imports the module {target!r} of layer "
                    f"{target_layer!r} wholesale; import the interface instead"
                )
                continue
            for name in names:
                allowed, reason = _is_allowed_symbol(target, name)
                if not allowed:
                    problems.append(
                        f"{module}:{line} imports {name!r} ({reason}) from {target!r} "
                        f"in layer {target_layer!r}"
                    )
    return problems


def test_no_outer_layer_imports_a_concrete_class_of_another_layer() -> None:
    """The rule of Section 9.2, over the whole of src/."""
    problems = _violations()
    assert not problems, "dependency direction violated:\n  " + "\n  ".join(problems)


def test_the_source_tree_is_actually_being_analysed() -> None:
    """A rule that silently analyses nothing would always pass."""
    modules = [name for name, _ in _source_modules() if _layer_of(name)]
    assert len(modules) >= 20, f"only {len(modules)} modules analysed: {modules}"
    assert "application.recommendation_controller" in modules
    assert "persistence.sql_user_profile_repository" in modules


def test_the_rule_would_catch_a_real_violation(tmp_path: Path) -> None:
    """Verify the detector, not just its verdict.

    A concrete adapter imported across a layer boundary must be reported;
    otherwise a green suite would prove nothing.
    """
    offender = tmp_path / "offender.py"
    offender.write_text(
        "from persistence.sql_user_profile_repository import SqlUserProfileRepository\n",
        encoding="utf-8",
    )
    target, names, _function, _line = _imports_of(offender)[0]
    allowed, reason = _is_allowed_symbol(target, names[0])
    assert not allowed and reason == "concrete class"


def test_the_rule_accepts_the_interfaces_it_is_meant_to_allow() -> None:
    """The four repository contracts and the strategy contract are importable."""
    for name in (
        "IUserProfileRepository",
        "ICatalogRepository",
        "IInteractionHistoryRepository",
        "IEmbeddingRepository",
    ):
        allowed, reason = _is_allowed_symbol("persistence.interfaces", name)
        assert allowed, f"{name}: {reason}"
    allowed, reason = _is_allowed_symbol("ai.interfaces", "IRecommendationStrategy")
    assert allowed, reason


def test_composition_roots_are_declared_and_minimal() -> None:
    """The exemption must stay small and intentional."""
    assert set(COMPOSITION_ROOTS) == {"application.api", "messaging.feedback_history_service"}
    assert COMPOSITION_ROOTS["messaging.feedback_history_service"] == {"main"}


@pytest.mark.parametrize(
    "module",
    [
        "application.recommendation_controller",
        "application.interaction_ingestion_service",
        "data.feature_store_manager",
        "messaging.feedback_history_service",
    ],
)
def test_no_module_outside_the_ai_layer_imports_a_strategy(module: str) -> None:
    """The controller depends on the interface, never on an algorithm."""
    path = SRC / (module.replace(".", "/") + ".py")
    for target, _names, function, _line in _imports_of(path):
        if _is_exempt(module, function):
            continue
        assert not target.startswith("ai.strategies"), f"{module} imports {target}"
        assert not target.startswith("ai.decorators"), f"{module} imports {target}"
