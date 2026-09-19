"""DeepLearningStrategy - neural collaborative filtering (FR-04, Table 9).

User and item embedding layers plus a multilayer perceptron (PyTorch),
trained offline and loaded by ``ModelFactory`` at startup. The training is an
offline script that writes versioned weights, so the inference path never
trains (Section 9.5):

    python -m ai.strategies.deep_learning_strategy --train

PyTorch is imported lazily, inside ``build`` and ``train``, so importing the
strategies package to populate the registry does not require the optional
deep-learning dependency to be installed.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from typing import Any

import numpy as np

from ai.interfaces import IRecommendationStrategy, ItemId, ScoredItem, UserFeatures, top_k
from ai.model_factory import register_strategy
from ai.strategies import load_feature_table
from config.settings import Settings, load_settings
from errors import ColdStartError, ModelUnavailableError

#: Name of the checkpoint written by the offline training script.
WEIGHTS_FILENAME = "ncf.pt"

#: Aggregated statistics fed to the MLP alongside the embeddings. MovieLens
#: ships no demographic attributes, so the "demographic attributes" input of
#: Table 9 is empty for this dataset variant (see IMPLEMENTATION_NOTES.md).
STATISTIC_FEATURES: tuple[str, ...] = ("n_interactions", "mean_rating", "rating_std")


def _import_torch() -> Any:
    """Import PyTorch, translating its absence into the platform vocabulary."""
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ModelUnavailableError(
            "PyTorch is not installed, the deep-learning strategy cannot be served",
            detail="install the 'deep' extra to enable it",
        ) from exc


def _make_module(torch: Any) -> Any:
    """Define the neural collaborative-filtering module.

    The class is defined inside a function because it subclasses
    ``torch.nn.Module``, which must not be imported when the strategies
    package is merely being loaded to populate the registry.
    """

    class NeuralCollaborativeFiltering(torch.nn.Module):
        """Embedding layers plus an MLP over the concatenated representation."""

        def __init__(self, n_users: int, n_items: int, dim: int, n_statistics: int) -> None:
            super().__init__()
            self.user_embedding = torch.nn.Embedding(n_users, dim)
            self.item_embedding = torch.nn.Embedding(n_items, dim)
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(2 * dim + n_statistics, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 1),
            )
            torch.nn.init.normal_(self.user_embedding.weight, std=0.05)
            torch.nn.init.normal_(self.item_embedding.weight, std=0.05)

        def forward(self, users: Any, items: Any, statistics: Any) -> Any:
            features = torch.cat(
                [self.user_embedding(users), self.item_embedding(items), statistics], dim=1
            )
            return self.mlp(features).squeeze(1)

    return NeuralCollaborativeFiltering


@register_strategy("ncf")
class DeepLearningStrategy(IRecommendationStrategy):
    """Serves a neural collaborative-filtering model trained offline."""

    def __init__(
        self,
        module: Any,
        torch_module: Any,
        item_ids: np.ndarray,
        user_index: Mapping[int, int],
        *,
        model_version: str = "unknown",
    ) -> None:
        self._module = module
        self._torch = torch_module
        self._item_ids = np.asarray(item_ids, dtype=np.int64)
        self._user_index = dict(user_index)
        self._model_version = model_version
        self._item_tensor = torch_module.arange(len(self._item_ids), dtype=torch_module.long)

    @classmethod
    def build(cls, settings: Settings | None = None) -> DeepLearningStrategy:
        """Load the versioned checkpoint written by the training script."""
        settings = settings or load_settings()
        torch = _import_torch()

        checkpoint_path = (
            settings.model.weights_root / settings.model.model_version / WEIGHTS_FILENAME
        )
        if not checkpoint_path.exists():
            raise ModelUnavailableError(
                "the neural collaborative-filtering checkpoint is missing",
                detail=(
                    f"expected at {checkpoint_path}; train it with "
                    "'python -m ai.strategies.deep_learning_strategy --train'"
                ),
            )

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        module_class = _make_module(torch)
        module = module_class(
            n_users=int(checkpoint["n_users"]),
            n_items=int(checkpoint["n_items"]),
            dim=int(checkpoint["dim"]),
            n_statistics=int(checkpoint["n_statistics"]),
        )
        module.load_state_dict(checkpoint["state_dict"])
        module.eval()

        return cls(
            module,
            torch,
            np.asarray(checkpoint["item_ids"], dtype=np.int64),
            {int(k): int(v) for k, v in checkpoint["user_index"].items()},
            model_version=str(checkpoint.get("model_version", settings.model.model_version)),
        )

    def predict(self, user_features: UserFeatures, k: int = 10) -> list[ScoredItem]:
        """Score the whole catalog in one forward pass and select the Top-K."""
        row = self._user_index.get(int(user_features.user_id))
        if row is None:
            raise ColdStartError(
                f"user {user_features.user_id} was not part of the training set",
                detail="the neural model only scores users it has an embedding for",
            )

        torch = self._torch
        n_items = len(self._item_ids)
        statistics = _statistics_vector(user_features.statistics)

        with torch.no_grad():
            users = torch.full((n_items,), row, dtype=torch.long)
            stats = torch.from_numpy(np.tile(statistics, (n_items, 1)))
            logits = self._module(users, self._item_tensor, stats)
            scores = torch.sigmoid(logits).numpy().astype(np.float32)

        scores = _mask_seen(self._item_ids, scores, user_features.seen_items)
        return top_k(scores, self._item_ids, k)

    def __len__(self) -> int:
        return int(self._item_ids.size)


def _statistics_vector(statistics: Mapping[str, float]) -> np.ndarray:
    """Build the fixed-order statistics row the MLP was trained with."""
    values = [float(statistics.get(name, 0.0)) for name in STATISTIC_FEATURES]
    # The interaction count is the only unbounded input; compressing it keeps
    # the MLP inputs on a comparable scale.
    values[0] = float(np.log1p(values[0]))
    return np.asarray(values, dtype=np.float32)


def _mask_seen(
    item_ids: np.ndarray, scores: np.ndarray, seen_items: tuple[ItemId, ...]
) -> np.ndarray:
    """Exclude the items the user already interacted with."""
    if not seen_items:
        return scores
    masked = scores.copy()
    seen = np.fromiter((int(i) for i in seen_items), dtype=np.int64, count=len(seen_items))
    masked[np.isin(item_ids, seen)] = -np.inf
    return masked


def train(
    settings: Settings | None = None,
    *,
    epochs: int = 3,
    batch_size: int = 1024,
    negatives: int = 4,
    learning_rate: float = 1e-3,
) -> Mapping[str, Any]:
    """Offline training script: writes versioned weights, never called at inference.

    Positives are the interactions the pipeline kept; negatives are sampled
    uniformly from the items a user did not interact with, which is the
    standard implicit-feedback setup for this model family.
    """
    settings = settings or load_settings()
    torch = _import_torch()
    rng = np.random.default_rng(settings.pipeline.random_seed)
    torch.manual_seed(settings.pipeline.random_seed)

    ratings_path = settings.pipeline.interim_root / "clean" / "ratings.parquet"
    if not ratings_path.exists():
        raise ModelUnavailableError(
            "clean ratings are required to train",
            detail=f"expected at {ratings_path}; run the offline pipeline first",
        )

    import pandas as pd

    ratings = pd.read_parquet(ratings_path)
    positives = ratings[ratings["rating"] >= settings.pipeline.positive_rating_threshold]

    item_frame = load_feature_table(settings.feature_store.version_dir, "item_features")
    if item_frame is None or item_frame.empty:
        raise ModelUnavailableError("the item feature table is empty")
    item_ids = item_frame["item_id"].to_numpy(dtype=np.int64)
    item_index = {int(item): i for i, item in enumerate(item_ids)}

    user_ids = np.sort(positives["user_id"].unique())
    user_index = {int(user): i for i, user in enumerate(user_ids)}

    pairs = [
        (user_index[int(u)], item_index[int(i)])
        for u, i in zip(positives["user_id"], positives["item_id"], strict=True)
        if int(i) in item_index
    ]
    if not pairs:
        raise ModelUnavailableError("no positive interaction maps onto the catalog")

    pos_users = np.asarray([p[0] for p in pairs], dtype=np.int64)
    pos_items = np.asarray([p[1] for p in pairs], dtype=np.int64)

    statistics_frame = load_feature_table(settings.feature_store.version_dir, "user_features")
    if statistics_frame is None:
        raise ModelUnavailableError("feature store artifact 'user_features' is missing")
    statistics_lookup = {
        int(row["user_id"]): _statistics_vector(
            {name: float(row.get(name, 0.0) or 0.0) for name in STATISTIC_FEATURES}
        )
        for row in statistics_frame.to_dict(orient="records")
    }
    default_statistics = np.zeros(len(STATISTIC_FEATURES), dtype=np.float32)
    statistics_matrix = np.vstack(
        [statistics_lookup.get(int(u), default_statistics) for u in user_ids]
    )

    module_class = _make_module(torch)
    module = module_class(
        n_users=len(user_ids),
        n_items=len(item_ids),
        dim=settings.pipeline.embedding_dimension,
        n_statistics=len(STATISTIC_FEATURES),
    )
    optimizer = torch.optim.Adam(module.parameters(), lr=learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()
    statistics_tensor = torch.from_numpy(statistics_matrix)

    history: list[float] = []
    module.train()
    for epoch in range(epochs):
        order = rng.permutation(len(pos_users))
        epoch_users = np.concatenate([pos_users[order]] + [pos_users[order]] * negatives)
        epoch_items = np.concatenate(
            [pos_items[order]]
            + [rng.integers(0, len(item_ids), size=len(pos_items)) for _ in range(negatives)]
        )
        epoch_labels = np.concatenate(
            [np.ones(len(pos_users), dtype=np.float32)]
            + [np.zeros(len(pos_users), dtype=np.float32) for _ in range(negatives)]
        )
        shuffle = rng.permutation(len(epoch_users))
        epoch_users, epoch_items, epoch_labels = (
            epoch_users[shuffle],
            epoch_items[shuffle],
            epoch_labels[shuffle],
        )

        total = 0.0
        batches = 0
        for start in range(0, len(epoch_users), batch_size):
            users = torch.from_numpy(epoch_users[start : start + batch_size])
            items = torch.from_numpy(epoch_items[start : start + batch_size])
            labels = torch.from_numpy(epoch_labels[start : start + batch_size])
            stats = statistics_tensor[users]

            optimizer.zero_grad()
            loss = criterion(module(users, items, stats), labels)
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            batches += 1

        history.append(total / max(batches, 1))
        print(f"epoch {epoch + 1}/{epochs} loss={history[-1]:.4f}")

    target_dir = settings.model.weights_root / settings.model.model_version
    target_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = target_dir / WEIGHTS_FILENAME
    module.eval()
    torch.save(
        {
            "state_dict": module.state_dict(),
            "n_users": len(user_ids),
            "n_items": len(item_ids),
            "dim": settings.pipeline.embedding_dimension,
            "n_statistics": len(STATISTIC_FEATURES),
            "item_ids": item_ids,
            "user_index": user_index,
            "model_version": settings.model.model_version,
        },
        checkpoint_path,
    )

    summary: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "model_version": settings.model.model_version,
        "users": len(user_ids),
        "items": len(item_ids),
        "positives": len(pos_users),
        "epochs": epochs,
        "loss_history": [round(value, 4) for value in history],
    }
    (target_dir / "training_report.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ai.strategies.deep_learning_strategy",
        description="Offline training of the neural collaborative-filtering model.",
    )
    parser.add_argument("--train", action="store_true", help="Train and write versioned weights")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--negatives", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args(argv)

    if not args.train:
        parser.error("nothing to do: pass --train")

    summary = train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        negatives=args.negatives,
        learning_rate=args.learning_rate,
    )
    print(json.dumps(dict(summary), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DeepLearningStrategy", "train"]
