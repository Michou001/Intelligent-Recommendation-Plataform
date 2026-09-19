"""Concrete IRecommendationStrategy implementations (Table 9).

Importing this package loads the four bundled strategies, and loading each
module executes its ``@register_strategy`` decorator, which is what populates
the registry ``ModelFactory`` resolves against (Section 9.5). No strategy is
aware of the others.

The shared loader below is defined before the submodule imports at the bottom
of this file, so the submodules can import it while this package is still
being initialised.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from errors import ModelUnavailableError


def load_feature_table(store_dir: Path, name: str, *, required: bool = True) -> pd.DataFrame | None:
    """Read one materialised table of the feature store.

    Strategies read their item-side artifacts once, when ``ModelFactory``
    builds them at startup; the request path never touches a file. A missing
    artifact is translated into ``ModelUnavailableError`` so the caller
    degrades instead of failing (NFR-05).
    """
    path = Path(store_dir) / f"{name}.parquet"
    if not path.exists():
        if required:
            raise ModelUnavailableError(
                f"feature store artifact {name!r} is missing",
                detail=f"expected at {path}; run the offline pipeline first",
            )
        return None
    try:
        return pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001 - translated into the platform vocabulary
        raise ModelUnavailableError(
            f"feature store artifact {name!r} cannot be read",
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc


from ai.strategies import (
    collaborative_filtering_strategy,  # noqa: E402,F401
    content_based_strategy,  # noqa: E402,F401
    deep_learning_strategy,  # noqa: E402,F401
    popularity_strategy,  # noqa: E402,F401
)

__all__ = ["load_feature_table"]
