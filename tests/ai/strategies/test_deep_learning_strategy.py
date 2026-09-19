"""DeepLearningStrategy - neural collaborative filtering (FR-04).

The model is trained offline, so the tests that need weights train a tiny one
inside the temporary directory rather than depending on a checkpoint produced
elsewhere. They are skipped when PyTorch is not installed, because the
strategy is an optional extra.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ai.interfaces import UserFeatures
from ai.strategies.deep_learning_strategy import (
    STATISTIC_FEATURES,
    DeepLearningStrategy,
    train,
)
from config.settings import Settings
from errors import ModelUnavailableError

torch = pytest.importorskip("torch", reason="the deep-learning strategy is an optional extra")


@pytest.fixture
def trained(settings: Settings) -> Settings:
    """Train a minimal checkpoint against the synthetic feature store."""
    clean = settings.pipeline.interim_root / "clean"
    clean.mkdir(parents=True, exist_ok=True)
    users = pd.read_parquet(settings.feature_store.version_dir / "user_features.parquet")
    items = pd.read_parquet(settings.feature_store.version_dir / "item_features.parquet")
    ratings = pd.DataFrame(
        [
            {"user_id": int(u), "item_id": int(i), "rating": 5.0, "timestamp": 0}
            for u in users["user_id"]
            for i in items["item_id"][:5]
        ]
    )
    ratings.to_parquet(clean / "ratings.parquet", index=False)
    train(settings, epochs=1, batch_size=32, negatives=1)
    return settings


def test_a_missing_checkpoint_is_reported_as_unavailable(settings: Settings) -> None:
    """NFR-05: an unavailable model degrades, it does not crash the service."""
    with pytest.raises(ModelUnavailableError) as excinfo:
        DeepLearningStrategy.build(settings)
    assert excinfo.value.degradable is True
    assert "checkpoint" in excinfo.value.message


def test_training_writes_a_versioned_checkpoint(trained: Settings) -> None:
    checkpoint = trained.model.weights_root / trained.model.model_version / "ncf.pt"
    assert checkpoint.exists()
    report = checkpoint.parent / "training_report.json"
    assert report.exists()


def test_a_trained_model_produces_a_ranking(trained: Settings, user_features: UserFeatures) -> None:
    strategy = DeepLearningStrategy.build(trained)
    result = strategy.predict(user_features, k=3)
    assert len(result) == 3
    assert [item.score for item in result] == sorted((i.score for i in result), reverse=True)


def test_scores_are_probabilities(trained: Settings, user_features: UserFeatures) -> None:
    result = DeepLearningStrategy.build(trained).predict(user_features, k=5)
    assert all(0.0 <= item.score <= 1.0 for item in result)


def test_a_user_outside_the_training_set_is_a_cold_start(trained: Settings) -> None:
    from errors import ColdStartError

    strategy = DeepLearningStrategy.build(trained)
    with pytest.raises(ColdStartError):
        strategy.predict(UserFeatures(user_id=987654), k=3)


def test_the_statistics_inputs_are_the_documented_ones() -> None:
    """Table 9 lists aggregated statistics among the model inputs."""
    assert STATISTIC_FEATURES == ("n_interactions", "mean_rating", "rating_std")


def test_inference_never_trains(trained: Settings, user_features: UserFeatures) -> None:
    """Section 9.5: the inference path never fits a model."""
    strategy = DeepLearningStrategy.build(trained)
    before = [p.detach().clone() for p in strategy._module.parameters()]  # noqa: SLF001
    strategy.predict(user_features, k=3)
    after = list(strategy._module.parameters())  # noqa: SLF001
    assert all(torch.equal(b, a) for b, a in zip(before, after, strict=True))


def test_it_registers_itself_under_the_documented_identifier() -> None:
    assert DeepLearningStrategy.strategy_id == "ncf"
