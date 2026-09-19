"""Stage 4 of Table 8 - Materialization.

feature tables -> Parquet feature store. Writing of user and item feature
vectors with a version tag, so that ``FeatureStoreManager`` retrieves
precomputed values only (FR-03).

The version tag is what lets a new materialisation be published beside the
one in use: the store is written under ``<feature_store_root>/<version>/``,
and the running service keeps reading the version named in its configuration
until that configuration changes. It is also what makes the cache of
``CachingDecorator`` invalidate itself when a new model is deployed
(Section 9.7).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import Settings, load_settings
from data.pipeline import build_parser, get_logger, print_summary, run_stage

LOGGER = get_logger("materialize")

#: Tables copied into the versioned store, and whether the store can be served
#: without them.
STORE_TABLES: tuple[tuple[str, bool], ...] = (
    ("user_features", True),
    ("item_features", True),
    ("popularity", True),
    ("category_popularity", False),
)

#: Name of the manifest FeatureStoreManager reads to validate a store.
METADATA_FILENAME = "_metadata.json"


def _read_feature_table(interim_root: Path, name: str) -> pd.DataFrame:
    path = interim_root / "features" / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing feature table {path}; run data.pipeline.transform first")
    return pd.read_parquet(path)


def run(settings: Settings | None = None, *, version: str | None = None) -> Mapping[str, Any]:
    """Publish the feature tables as a versioned Parquet feature store."""
    settings = settings or load_settings()
    version = version or settings.feature_store.version
    interim_root = settings.pipeline.interim_root
    target_dir = settings.feature_store.root / version
    target_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, dict[str, Any]] = {}
    embedding_dimension = 0
    latent_dimension = 0

    for name, required in STORE_TABLES:
        try:
            frame = _read_feature_table(interim_root, name)
        except FileNotFoundError:
            if required:
                raise
            LOGGER.warning("optional table %s not present, skipped", name)
            continue

        target = target_dir / f"{name}.parquet"
        frame.to_parquet(target, index=False)
        written[name] = {"rows": int(len(frame)), "path": str(target)}
        LOGGER.info("materialised %-20s rows=%d", name, len(frame))

        if name == "user_features" and len(frame):
            latent_dimension = len(frame["latent_factors"].iloc[0])
            embedding_dimension = len(frame["content_profile"].iloc[0])

    metadata: dict[str, Any] = {
        "version": version,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_variant": settings.dataset.variant,
        "latent_dimension": latent_dimension,
        "embedding_dimension": embedding_dimension,
        "tables": written,
    }
    (target_dir / METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )

    summary: dict[str, Any] = {
        "stage": "materialize",
        "version": version,
        "feature_store": str(target_dir),
        **{f"{name}_rows": info["rows"] for name, info in written.items()},
        "latent_dimension": latent_dimension,
        "embedding_dimension": embedding_dimension,
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = build_parser("materialize", __doc__ or "")
    parser.add_argument("--version", default=None, help="Feature store version tag to publish")
    parser.add_argument("--feature-store-root", default=None, help="Feature store root directory")
    args = parser.parse_args(argv)

    import os

    if args.interim_root:
        os.environ["RECO_INTERIM_ROOT"] = args.interim_root
    if args.feature_store_root:
        os.environ["RECO_FEATURE_STORE_ROOT"] = args.feature_store_root

    summary = run_stage("materialize", run, settings=load_settings(), version=args.version)
    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
