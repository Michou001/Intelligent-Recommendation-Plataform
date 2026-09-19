"""The four offline stages of Table 8.

The tests run the real pipeline over a miniature CSV dataset written in a
temporary directory, so they exercise the actual code paths - chunked
reading, the cleaning rules, the factorisation, the versioned write - without
depending on MovieLens being downloaded.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from config.settings import Settings, load_settings
from data.pipeline import clean, ingest, materialize, transform


@pytest.fixture
def mini_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Write a small MovieLens-shaped dataset and point the settings at it."""
    root = tmp_path / "datasets"
    variant = root / "mini"
    variant.mkdir(parents=True)

    rows = []
    for user in range(1, 9):
        for item in range(1, 13):
            rows.append(
                {
                    "userId": user,
                    "movieId": item,
                    "rating": 4.5 if (user + item) % 3 else 2.0,
                    "timestamp": 1000 + user * 10 + item,
                }
            )
    ratings = pd.DataFrame(rows)
    # One duplicate, one out-of-range rating, and one unparseable row, so the
    # cleaning report has something real to report.
    ratings = pd.concat([ratings, ratings.iloc[[0]]], ignore_index=True)
    ratings.loc[len(ratings)] = {"userId": 1, "movieId": 2, "rating": 99.0, "timestamp": 9999}
    ratings.to_csv(variant / "ratings.csv", index=False)
    with (variant / "ratings.csv").open("a", encoding="utf-8") as handle:
        handle.write("not-a-number,3,4.0,1234\n")

    pd.DataFrame(
        [
            {
                "movieId": i,
                "title": f"Movie {i} (2000)",
                "genres": "Drama|Comedy" if i % 2 else "Action",
            }
            for i in range(1, 13)
        ]
        # An item with no genre listed, which cleaning must drop.
        + [{"movieId": 99, "title": "Genreless (2001)", "genres": "(no genres listed)"}]
    ).to_csv(variant / "movies.csv", index=False)

    pd.DataFrame([{"userId": 1, "movieId": 1, "tag": "classic", "timestamp": 1}]).to_csv(
        variant / "tags.csv", index=False
    )

    monkeypatch.setenv("RECO_DATASET_ROOT", str(root))
    monkeypatch.setenv("RECO_DATASET_VARIANT", "mini")
    monkeypatch.setenv("RECO_INTERIM_ROOT", str(tmp_path / "interim"))
    monkeypatch.setenv("RECO_FEATURE_STORE_ROOT", str(tmp_path / "feature_store"))
    monkeypatch.setenv("RECO_FEATURE_STORE_VERSION", "v-pipeline")
    monkeypatch.setenv("RECO_INGEST_CHUNK_SIZE", "17")
    monkeypatch.setenv("RECO_LATENT_FACTORS", "4")
    monkeypatch.setenv("RECO_EMBEDDING_DIM", "4")
    monkeypatch.setenv("RECO_MIN_USER_INTERACTIONS", "3")
    return load_settings()


def test_ingestion_reads_in_chunks(mini_dataset: Settings) -> None:
    """Section 9.3: chunked reading is what makes a larger variant possible."""
    summary = ingest.run(mini_dataset)
    assert summary["sources"]["ratings"]["chunks"] > 1
    assert summary["chunk_size"] == 17


def test_ingestion_reports_unreadable_rows(mini_dataset: Settings) -> None:
    summary = ingest.run(mini_dataset)
    assert summary["sources"]["ratings"]["rows_rejected"] == 1


def test_ingestion_writes_a_report_next_to_the_artifacts(mini_dataset: Settings) -> None:
    ingest.run(mini_dataset)
    assert (mini_dataset.pipeline.interim_root / "raw" / "ingestion_report.json").exists()


def test_a_missing_dataset_is_an_explicit_error(mini_dataset: Settings, monkeypatch) -> None:
    monkeypatch.setenv("RECO_DATASET_VARIANT", "does-not-exist")
    with pytest.raises(ingest.DatasetNotFoundError):
        ingest.run(load_settings())


def test_cleaning_applies_the_five_documented_rules(mini_dataset: Settings) -> None:
    ingest.run(mini_dataset)
    summary = clean.run(mini_dataset)
    report = summary["ratings"]
    assert report["duplicates_removed"] >= 1
    assert report["out_of_range_removed"] >= 1
    assert summary["items"]["no_genre_removed"] >= 1
    assert report["rows_out"] < report["rows_in"]


def test_cleaning_removes_items_without_metadata(mini_dataset: Settings) -> None:
    ingest.run(mini_dataset)
    clean.run(mini_dataset)
    items = pd.read_parquet(mini_dataset.pipeline.interim_root / "clean" / "items.parquet")
    assert 99 not in set(items["item_id"])


def test_cleaning_requires_the_ingestion_artifacts(mini_dataset: Settings) -> None:
    with pytest.raises(FileNotFoundError):
        clean.run(mini_dataset)


def test_transformation_produces_the_four_documented_outputs(mini_dataset: Settings) -> None:
    ingest.run(mini_dataset)
    clean.run(mini_dataset)
    summary = transform.run(mini_dataset)

    features = mini_dataset.pipeline.interim_root / "features"
    assert (features / "user_features.parquet").exists()
    assert (features / "item_features.parquet").exists()
    assert (features / "popularity.parquet").exists()
    assert (features / "category_popularity.parquet").exists()
    assert summary["matrix_nnz"] > 0
    assert summary["latent_factors"] == 4


def test_the_popularity_ranking_is_ordered(mini_dataset: Settings) -> None:
    ingest.run(mini_dataset)
    clean.run(mini_dataset)
    transform.run(mini_dataset)
    popularity = pd.read_parquet(
        mini_dataset.pipeline.interim_root / "features" / "popularity.parquet"
    )
    assert popularity["popularity_score"].is_monotonic_decreasing
    assert list(popularity["rank"]) == list(range(1, len(popularity) + 1))


def test_materialization_writes_a_versioned_store(mini_dataset: Settings) -> None:
    ingest.run(mini_dataset)
    clean.run(mini_dataset)
    transform.run(mini_dataset)
    summary = materialize.run(mini_dataset)

    store = mini_dataset.feature_store.root / "v-pipeline"
    assert summary["version"] == "v-pipeline"
    assert (store / "user_features.parquet").exists()
    assert (store / "_metadata.json").exists()


def test_two_versions_coexist(mini_dataset: Settings) -> None:
    """The version tag is what lets a new store be published beside the old."""
    ingest.run(mini_dataset)
    clean.run(mini_dataset)
    transform.run(mini_dataset)
    materialize.run(mini_dataset, version="v1")
    materialize.run(mini_dataset, version="v2")
    assert (mini_dataset.feature_store.root / "v1").exists()
    assert (mini_dataset.feature_store.root / "v2").exists()


def test_changing_the_variant_needs_no_code_change(mini_dataset: Settings, monkeypatch) -> None:
    """Rule of Section 9.3, stated as a property of the configuration."""
    monkeypatch.setenv("RECO_DATASET_VARIANT", "another-variant")
    reloaded = load_settings()
    assert reloaded.dataset.variant == "another-variant"
    assert reloaded.dataset.variant_dir.name == "another-variant"
    assert reloaded.dataset.download_url.endswith("another-variant.zip")


def test_the_materialised_store_feeds_the_feature_store_manager(mini_dataset: Settings) -> None:
    """The pipeline and the reader agree on the schema, end to end."""
    from data.feature_store_manager import ParquetFeatureStoreManager

    ingest.run(mini_dataset)
    clean.run(mini_dataset)
    transform.run(mini_dataset)
    materialize.run(mini_dataset)

    manager = ParquetFeatureStoreManager.from_settings(mini_dataset)
    assert manager.is_available()
    features = manager.get_user_features(1)
    assert features.latent_factors is not None
    assert features.latent_factors.shape == (4,)
