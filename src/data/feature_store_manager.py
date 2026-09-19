"""Data Access & Feature Store Layer - FeatureStoreManager (FR-03, Table 11).

This component **only reads**. It retrieves the precomputed user vectors that
``data/pipeline/`` materialised and serves them to the inference path without
recomputing anything. That separation is the explicit correction recorded in
Table 6 for Milestone 1, where the same component was described as computing
and storing features at once; collapsing it back would re-introduce the
contradiction between FR-03, Section 3.1 and Section 5.2.

``FeatureStoreManager`` is declared as an abstract base class so the
Application layer depends on the contract and not on the storage format. The
Parquet backend of Table 7 is one implementation of it.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ai.interfaces import UserFeatures, UserId
from config.settings import Settings, load_settings
from errors import ColdStartError, FeatureStoreUnavailableError


class FeatureStoreManager(ABC):
    """Serves precomputed feature vectors to the inference path.

    Implementations never compute a feature value, never write to the store,
    and never reach the transactional databases: the only thing they do is
    retrieve what the offline pipeline already materialised.
    """

    @abstractmethod
    def get_user_features(self, user_id: UserId) -> UserFeatures:
        """Return the precomputed vectors of one user.

        Raises:
            ColdStartError: The user is not present in the store, so there is
                no history to serve and the caller must fall back (FR-05).
            FeatureStoreUnavailableError: The store cannot be read at all.
        """

    @abstractmethod
    def is_available(self) -> bool:
        """Whether the store can currently be served."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Version tag of the materialised store being served."""

    def describe(self) -> Mapping[str, str]:
        """Identification emitted in operational logs."""
        return {"feature_store_version": self.version}


class ParquetFeatureStoreManager(FeatureStoreManager):
    """Parquet backend of the feature store (Table 7: pandas + Apache Parquet).

    The user table is read once, at construction, and indexed by user
    identifier. A request therefore costs a dictionary lookup and no file
    access, which is what keeps feature retrieval out of the p95 budget of
    NFR-01. The cost is that the resident set grows with the number of users
    in the selected dataset variant.
    """

    USER_TABLE = "user_features.parquet"
    METADATA_FILE = "_metadata.json"

    def __init__(self, store_dir: Path, *, eager: bool = True) -> None:
        self._store_dir = Path(store_dir)
        self._version: str = self._store_dir.name
        self._rows: dict[int, dict[str, Any]] = {}
        self._loaded = False
        self._load_error: str | None = None
        self._loaded_at: float | None = None
        if eager:
            self._load()

    @classmethod
    def from_settings(
        cls, settings: Settings | None = None, *, eager: bool = True
    ) -> ParquetFeatureStoreManager:
        """Build the manager from the configured store root and version."""
        settings = settings or load_settings()
        return cls(settings.feature_store.version_dir, eager=eager)

    def _load(self) -> None:
        """Read the materialised user table into memory, once."""
        table_path = self._store_dir / self.USER_TABLE
        try:
            if not table_path.exists():
                raise FileNotFoundError(f"feature store table not found: {table_path}")
            frame = pd.read_parquet(table_path)
            self._rows = {int(row["user_id"]): row for row in frame.to_dict(orient="records")}
            metadata_path = self._store_dir / self.METADATA_FILE
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                self._version = str(metadata.get("version", self._store_dir.name))
            self._loaded = True
            self._load_error = None
            self._loaded_at = time.time()
        except Exception as exc:  # noqa: BLE001 - translated into the platform vocabulary
            self._loaded = False
            self._load_error = f"{type(exc).__name__}: {exc}"
            self._rows = {}

    def is_available(self) -> bool:
        return self._loaded

    @property
    def version(self) -> str:
        return self._version

    def get_user_features(self, user_id: UserId) -> UserFeatures:
        if not self._loaded:
            raise FeatureStoreUnavailableError(
                "the feature store cannot be read",
                detail=self._load_error,
            )

        row = self._rows.get(int(user_id))
        if row is None:
            raise ColdStartError(f"user {user_id} has no materialised feature vector")

        latent = _as_vector(row.get("latent_factors"))
        profile = _as_vector(row.get("content_profile"))
        statistics: dict[str, float] = {
            name: float(row[name])
            for name in ("n_interactions", "mean_rating", "rating_std", "min_rating", "max_rating")
            if name in row and row[name] is not None and not _is_missing(row[name])
        }
        seen = row.get("seen_items")
        seen_items = tuple(int(i) for i in seen) if seen is not None and len(seen) else ()
        category = row.get("preferred_category")

        return UserFeatures(
            user_id=int(user_id),
            latent_factors=latent,
            content_profile=profile,
            statistics=statistics,
            seen_items=seen_items,
            preferred_category=None if _is_missing(category) else str(category),
            is_cold_start=latent is None and profile is None,
            feature_store_version=self._version,
        )

    def reload(self) -> bool:
        """Re-read the store, for example after a new version was published."""
        self._load()
        return self._loaded

    def __len__(self) -> int:
        return len(self._rows)


def _is_missing(value: Any) -> bool:
    """True for None and for the NaN pandas writes into absent cells."""
    if value is None:
        return True
    return isinstance(value, float) and np.isnan(value)


def _as_vector(value: Any) -> np.ndarray | None:
    """Convert a Parquet list cell into a float32 vector, or None when empty."""
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    if array.size == 0:
        return None
    return array


__all__ = ["FeatureStoreManager", "ParquetFeatureStoreManager"]
