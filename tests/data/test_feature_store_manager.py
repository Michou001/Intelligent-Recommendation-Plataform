"""FeatureStoreManager (FR-03, Table 6 correction).

The property under test is the one Milestone 1 got wrong: this component
reads, and only reads. A test that merely checked the returned vectors would
not notice a manager that started computing them, so the separation is
asserted directly.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from config.settings import Settings
from data.feature_store_manager import FeatureStoreManager, ParquetFeatureStoreManager
from errors import ColdStartError, FeatureStoreUnavailableError
from tests.conftest import COLD_START_USER, DIMENSION


@pytest.fixture
def manager(settings: Settings) -> ParquetFeatureStoreManager:
    return ParquetFeatureStoreManager.from_settings(settings)


def test_it_serves_the_materialised_vectors(manager: ParquetFeatureStoreManager) -> None:
    features = manager.get_user_features(1)
    assert features.user_id == 1
    assert features.latent_factors is not None
    assert features.latent_factors.shape == (DIMENSION,)
    assert features.content_profile is not None
    assert features.feature_store_version == "v-test"


def test_it_serves_the_aggregated_statistics(manager: ParquetFeatureStoreManager) -> None:
    statistics = manager.get_user_features(1).statistics
    assert set(statistics) >= {"n_interactions", "mean_rating", "rating_std"}


def test_it_serves_the_seen_items_and_the_preferred_category(
    manager: ParquetFeatureStoreManager,
) -> None:
    features = manager.get_user_features(1)
    assert features.seen_items == (101, 102)
    assert features.preferred_category in {"Drama", "Comedy"}


def test_an_unknown_user_is_a_cold_start(manager: ParquetFeatureStoreManager) -> None:
    with pytest.raises(ColdStartError) as excinfo:
        manager.get_user_features(COLD_START_USER)
    assert excinfo.value.degradable is True


def test_an_unreadable_store_is_reported_as_unavailable(tmp_path: Path) -> None:
    broken = ParquetFeatureStoreManager(tmp_path / "does-not-exist")
    assert broken.is_available() is False
    with pytest.raises(FeatureStoreUnavailableError) as excinfo:
        broken.get_user_features(1)
    assert excinfo.value.status_code == 200
    assert excinfo.value.degradable is True


def test_a_corrupt_table_is_reported_as_unavailable(tmp_path: Path) -> None:
    store = tmp_path / "corrupt"
    store.mkdir()
    (store / "user_features.parquet").write_text("this is not parquet", encoding="utf-8")
    manager = ParquetFeatureStoreManager(store)
    assert manager.is_available() is False


def test_it_never_computes_a_feature() -> None:
    """Section 4.2: the values are computed offline, never here."""
    source = inspect.getsource(ParquetFeatureStoreManager)
    for forbidden in ("fit(", "TfidfVectorizer", "AlternatingLeastSquares", "groupby", "mean()"):
        assert forbidden not in source, f"FeatureStoreManager appears to compute: {forbidden}"


def test_it_never_writes_to_the_store() -> None:
    source = inspect.getsource(ParquetFeatureStoreManager)
    for forbidden in ("to_parquet", "write_text", "open(", "savez"):
        assert forbidden not in source, f"FeatureStoreManager appears to write: {forbidden}"


def test_the_contract_is_an_abstract_base_class() -> None:
    """The application layer depends on this, not on the Parquet backend."""
    assert inspect.isabstract(FeatureStoreManager)
    with pytest.raises(TypeError):
        FeatureStoreManager()  # type: ignore[abstract]


def test_the_table_is_read_once_not_per_request(
    manager: ParquetFeatureStoreManager, settings: Settings
) -> None:
    """Deleting the file after startup must not affect a lookup."""
    (settings.feature_store.version_dir / "user_features.parquet").unlink()
    assert manager.get_user_features(1).user_id == 1


def test_reload_picks_up_a_republished_store(
    manager: ParquetFeatureStoreManager, settings: Settings
) -> None:
    (settings.feature_store.version_dir / "user_features.parquet").unlink()
    assert manager.reload() is False
    assert manager.is_available() is False
