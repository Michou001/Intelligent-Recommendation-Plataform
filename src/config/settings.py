"""Central configuration for the Intelligent Recommendation Platform.

This module is the single place where the dataset variant, the active
recommendation strategy, the decorator chain, and every infrastructure
endpoint are declared (Section 9.2). Changing the MovieLens variant from
``ml-latest-small`` to ``ml-25m`` is a change of one value here, never a
change of pipeline code (Section 9.3).

Every value can be overridden through an environment variable, so the same
build runs in development, testing, and production without edits.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[2]


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser().resolve() if raw else default


def _env_list(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name)
    if not raw:
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class DatasetSettings:
    """Dataset location and variant (Section 9.3).

    ``variant`` selects the MovieLens release. ``chunk_size`` is the number of
    rows read per chunk by the ingestion stage, which is what allows a larger
    variant to be processed without loading it entirely into memory.
    """

    variant: str = field(
        default_factory=lambda: _env_str("RECO_DATASET_VARIANT", "ml-latest-small")
    )
    root: Path = field(
        default_factory=lambda: _env_path("RECO_DATASET_ROOT", REPO_ROOT / "datasets")
    )
    chunk_size: int = field(default_factory=lambda: _env_int("RECO_INGEST_CHUNK_SIZE", 50_000))
    download_url_template: str = field(
        default_factory=lambda: _env_str(
            "RECO_DATASET_URL_TEMPLATE",
            "https://files.grouplens.org/datasets/movielens/{variant}.zip",
        )
    )

    @property
    def variant_dir(self) -> Path:
        """Directory holding the raw CSV files of the selected variant."""
        return self.root / self.variant

    @property
    def download_url(self) -> str:
        return self.download_url_template.format(variant=self.variant)


@dataclass(frozen=True)
class PipelineSettings:
    """Artifact locations and thresholds for the four offline stages of Table 8."""

    interim_root: Path = field(
        default_factory=lambda: _env_path("RECO_INTERIM_ROOT", REPO_ROOT / "artifacts" / "interim")
    )
    min_user_interactions: int = field(
        default_factory=lambda: _env_int("RECO_MIN_USER_INTERACTIONS", 5)
    )
    min_rating: float = field(default_factory=lambda: _env_float("RECO_MIN_RATING", 0.5))
    max_rating: float = field(default_factory=lambda: _env_float("RECO_MAX_RATING", 5.0))
    positive_rating_threshold: float = field(
        default_factory=lambda: _env_float("RECO_POSITIVE_RATING_THRESHOLD", 3.5)
    )
    embedding_dimension: int = field(default_factory=lambda: _env_int("RECO_EMBEDDING_DIM", 64))
    latent_factors: int = field(default_factory=lambda: _env_int("RECO_LATENT_FACTORS", 64))
    als_iterations: int = field(default_factory=lambda: _env_int("RECO_ALS_ITERATIONS", 20))
    als_regularization: float = field(
        default_factory=lambda: _env_float("RECO_ALS_REGULARIZATION", 0.05)
    )
    random_seed: int = field(default_factory=lambda: _env_int("RECO_RANDOM_SEED", 42))
    # Backend used to embed item metadata. "sentence-transformers" matches
    # Table 7; "tfidf-svd" is the offline scikit-learn backend used when the
    # model cannot be downloaded (see IMPLEMENTATION_NOTES.md).
    content_embedding_backend: str = field(
        default_factory=lambda: _env_str("RECO_CONTENT_EMBEDDING_BACKEND", "tfidf-svd")
    )
    sentence_transformer_model: str = field(
        default_factory=lambda: _env_str("RECO_SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
    )


@dataclass(frozen=True)
class FeatureStoreSettings:
    """Parquet feature store read by FeatureStoreManager (FR-03)."""

    root: Path = field(
        default_factory=lambda: _env_path(
            "RECO_FEATURE_STORE_ROOT", REPO_ROOT / "artifacts" / "feature_store"
        )
    )
    version: str = field(default_factory=lambda: _env_str("RECO_FEATURE_STORE_VERSION", "v1"))

    @property
    def version_dir(self) -> Path:
        return self.root / self.version


@dataclass(frozen=True)
class ModelSettings:
    """Active strategy and decorator chain (Sections 9.5 and 9.6)."""

    active_strategy: str = field(default_factory=lambda: _env_str("RECO_ACTIVE_STRATEGY", "als"))
    fallback_strategy: str = field(
        default_factory=lambda: _env_str("RECO_FALLBACK_STRATEGY", "popularity")
    )
    model_version: str = field(default_factory=lambda: _env_str("RECO_MODEL_VERSION", "v1"))
    weights_root: Path = field(
        default_factory=lambda: _env_path("RECO_WEIGHTS_ROOT", REPO_ROOT / "artifacts" / "models")
    )
    # Applied innermost-first, so the resulting object is
    # LoggingDecorator(TimingDecorator(CachingDecorator(base))).
    decorator_chain: list[str] = field(
        default_factory=lambda: _env_list("RECO_DECORATOR_CHAIN", ["caching", "timing", "logging"])
    )
    default_k: int = field(default_factory=lambda: _env_int("RECO_DEFAULT_K", 10))
    max_k: int = field(default_factory=lambda: _env_int("RECO_MAX_K", 100))


@dataclass(frozen=True)
class CacheSettings:
    """CachingDecorator backend. Falls back to an in-process LRU when Redis is absent."""

    url: str = field(default_factory=lambda: _env_str("RECO_REDIS_URL", ""))
    ttl_seconds: int = field(default_factory=lambda: _env_int("RECO_CACHE_TTL_SECONDS", 300))
    max_entries: int = field(default_factory=lambda: _env_int("RECO_CACHE_MAX_ENTRIES", 4096))


@dataclass(frozen=True)
class MessagingSettings:
    """InteractionEventQueue backend. Falls back to an in-memory queue without RabbitMQ."""

    broker_url: str = field(default_factory=lambda: _env_str("RECO_RABBITMQ_URL", ""))
    queue_name: str = field(
        default_factory=lambda: _env_str("RECO_QUEUE_NAME", "interaction-events")
    )
    retry_buffer_path: Path = field(
        default_factory=lambda: _env_path(
            "RECO_RETRY_BUFFER_PATH", REPO_ROOT / "artifacts" / "retry_buffer.jsonl"
        )
    )
    max_publish_retries: int = field(
        default_factory=lambda: _env_int("RECO_MAX_PUBLISH_RETRIES", 3)
    )


@dataclass(frozen=True)
class PersistenceSettings:
    """Storage endpoints behind the segregated repository interfaces (Section 3.2).

    ``sql_url`` defaults to a local SQLite file so the system runs with no
    database server; pointing it at PostgreSQL is a configuration change.
    An empty ``mongo_url`` selects the local file-backed document store.
    """

    sql_url: str = field(
        default_factory=lambda: _env_str(
            "RECO_SQL_URL",
            "sqlite:///" + (REPO_ROOT / "artifacts" / "platform.db").as_posix(),
        )
    )
    mongo_url: str = field(default_factory=lambda: _env_str("RECO_MONGO_URL", ""))
    mongo_database: str = field(
        default_factory=lambda: _env_str("RECO_MONGO_DB", "recommendation_platform")
    )
    document_store_root: Path = field(
        default_factory=lambda: _env_path(
            "RECO_DOCUMENT_STORE_ROOT", REPO_ROOT / "artifacts" / "documents"
        )
    )


@dataclass(frozen=True)
class LoggingSettings:
    """structlog configuration (Section 9.9)."""

    level: str = field(default_factory=lambda: _env_str("RECO_LOG_LEVEL", "INFO"))
    json_output: bool = field(default_factory=lambda: _env_bool("RECO_LOG_JSON", True))


@dataclass(frozen=True)
class Settings:
    """Aggregate configuration object consumed by the composition root."""

    dataset: DatasetSettings = field(default_factory=DatasetSettings)
    pipeline: PipelineSettings = field(default_factory=PipelineSettings)
    feature_store: FeatureStoreSettings = field(default_factory=FeatureStoreSettings)
    model: ModelSettings = field(default_factory=ModelSettings)
    cache: CacheSettings = field(default_factory=CacheSettings)
    messaging: MessagingSettings = field(default_factory=MessagingSettings)
    persistence: PersistenceSettings = field(default_factory=PersistenceSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)

    @property
    def ACTIVE_STRATEGY(self) -> str:  # noqa: N802 - spelling reproduced from Section 9.6
        """Alias kept identical to the composition snippet of Section 9.6."""
        return self.model.active_strategy


def load_settings() -> Settings:
    """Build a fresh Settings instance from the current environment."""
    return Settings()


settings: Settings = load_settings()
