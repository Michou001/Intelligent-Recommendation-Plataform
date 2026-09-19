"""Stage 3 of Table 8 - Transformation.

clean -> feature tables. Construction of the user-item interaction matrix,
aggregated user statistics, item content embeddings, and popularity ranking
for the cold-start strategy.

This stage is where feature *values* are computed. ``FeatureStoreManager``
never runs any of this code: it only reads what stage 4 materialises
(Section 4.2, correction of the Milestone 1 contradiction).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from config.settings import Settings, load_settings
from data.pipeline import build_parser, get_logger, print_summary, run_stage, write_report

LOGGER = get_logger("transform")


@dataclass(frozen=True)
class IndexMaps:
    """Dense row/column indices of the interaction matrix."""

    user_ids: np.ndarray
    item_ids: np.ndarray

    @property
    def user_position(self) -> dict[int, int]:
        return {int(u): i for i, u in enumerate(self.user_ids)}

    @property
    def item_position(self) -> dict[int, int]:
        return {int(m): i for i, m in enumerate(self.item_ids)}


def _read_clean(interim_root: Path, name: str) -> pd.DataFrame:
    path = interim_root / "clean" / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing cleaning artifact {path}; run data.pipeline.clean first")
    return pd.read_parquet(path)


def build_interaction_matrix(
    ratings: pd.DataFrame, settings: Settings
) -> tuple[csr_matrix, IndexMaps]:
    """Build the implicit-feedback user-item matrix.

    Ratings at or above the positive threshold become implicit positives whose
    confidence is the rating itself; weaker ratings carry no positive signal
    and are left out of the matrix.
    """
    positives = ratings[ratings["rating"] >= settings.pipeline.positive_rating_threshold]
    if positives.empty:
        raise ValueError("no interaction is above the positive rating threshold")

    user_ids = np.sort(positives["user_id"].unique())
    item_ids = np.sort(positives["item_id"].unique())
    maps = IndexMaps(user_ids=user_ids, item_ids=item_ids)

    rows = positives["user_id"].map(maps.user_position).to_numpy(dtype=np.int32)
    cols = positives["item_id"].map(maps.item_position).to_numpy(dtype=np.int32)
    values = positives["rating"].to_numpy(dtype=np.float32)

    matrix = csr_matrix((values, (rows, cols)), shape=(len(user_ids), len(item_ids)))
    LOGGER.info(
        "interaction matrix %dx%d nnz=%d density=%.5f%%",
        matrix.shape[0],
        matrix.shape[1],
        matrix.nnz,
        100 * matrix.nnz / (matrix.shape[0] * matrix.shape[1]),
    )
    return matrix, maps


def factorize(matrix: csr_matrix, settings: Settings) -> tuple[np.ndarray, np.ndarray, str]:
    """Factorise the interaction matrix into user and item latent factors.

    Uses Alternating Least Squares from ``implicit`` (Table 7). When that
    package is not installed the stage falls back to a truncated SVD of the
    same matrix, which produces latent factors with the same shape and the
    same consumption contract, so the strategy code is unaffected.
    """
    factors = settings.pipeline.latent_factors
    try:
        from implicit.als import AlternatingLeastSquares
        from threadpoolctl import threadpool_limits

        model = AlternatingLeastSquares(
            factors=factors,
            regularization=settings.pipeline.als_regularization,
            iterations=settings.pipeline.als_iterations,
            random_state=settings.pipeline.random_seed,
            use_gpu=False,
        )
        # ALS already parallelises its own loops; letting BLAS open a second
        # thread pool on top of that degrades the fit on a multi-core laptop.
        with threadpool_limits(1, "blas"):
            model.fit(matrix, show_progress=False)
        LOGGER.info("factorisation backend=implicit-als factors=%d", factors)
        return (
            np.asarray(model.user_factors, dtype=np.float32),
            np.asarray(model.item_factors, dtype=np.float32),
            "implicit-als",
        )
    except ImportError:
        from scipy.sparse.linalg import svds

        k = min(factors, min(matrix.shape) - 1)
        u, s, vt = svds(matrix.astype(np.float64), k=k)
        order = np.argsort(-s)
        scale = np.sqrt(s[order])
        user_factors = (u[:, order] * scale).astype(np.float32)
        item_factors = (vt[order, :].T * scale).astype(np.float32)
        LOGGER.warning("implicit not installed, factorisation backend=truncated-svd factors=%d", k)
        return user_factors, item_factors, "truncated-svd"


def build_item_content_embeddings(
    items: pd.DataFrame, tags: pd.DataFrame | None, settings: Settings
) -> tuple[np.ndarray, str]:
    """Embed the textual metadata of every item.

    The document specifies sentence-transformers (Table 7). That backend is
    selected by configuration; the default backend is a TF-IDF projected with
    truncated SVD, which needs no model download and is deterministic, so the
    pipeline is reproducible offline.
    """
    documents = items["title"].fillna("") + " " + items["genres"].fillna("").str.replace("|", " ")
    if tags is not None and not tags.empty:
        joined = tags.groupby("item_id")["tag"].apply(lambda values: " ".join(values.astype(str)))
        documents = documents + " " + items["item_id"].map(joined).fillna("")

    corpus = documents.tolist()
    backend = settings.pipeline.content_embedding_backend

    if backend == "sentence-transformers":
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(settings.pipeline.sentence_transformer_model)
            vectors = np.asarray(model.encode(corpus, show_progress_bar=False), dtype=np.float32)
            LOGGER.info("content embeddings backend=sentence-transformers dim=%d", vectors.shape[1])
            return vectors, "sentence-transformers"
        except ImportError:
            LOGGER.warning("sentence-transformers not installed, falling back to tfidf-svd")

    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(max_features=20_000, stop_words="english", sublinear_tf=True)
    tfidf = vectorizer.fit_transform(corpus)
    components = min(settings.pipeline.embedding_dimension, min(tfidf.shape) - 1)
    svd = TruncatedSVD(n_components=components, random_state=settings.pipeline.random_seed)
    vectors = svd.fit_transform(tfidf).astype(np.float32)
    LOGGER.info(
        "content embeddings backend=tfidf-svd dim=%d explained_variance=%.3f",
        vectors.shape[1],
        float(svd.explained_variance_ratio_.sum()),
    )
    return vectors, "tfidf-svd"


def build_user_statistics(ratings: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-user statistics consumed by DeepLearningStrategy."""
    grouped = ratings.groupby("user_id")["rating"]
    stats = pd.DataFrame(
        {
            "n_interactions": grouped.size(),
            "mean_rating": grouped.mean(),
            "rating_std": grouped.std().fillna(0.0),
            "min_rating": grouped.min(),
            "max_rating": grouped.max(),
        }
    ).reset_index()
    return stats


def build_popularity_ranking(ratings: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    """Rank items by a count-weighted mean rating (cold-start input, FR-05).

    A plain mean rating puts an item rated once at the top, so the score is
    shrunk towards the global mean in proportion to how little evidence the
    item has. The weight ``m`` is the median interaction count.
    """
    grouped = ratings.groupby("item_id")["rating"]
    counts = grouped.size()
    means = grouped.mean()
    global_mean = float(ratings["rating"].mean())
    m = float(max(counts.median(), 1.0))

    score = (counts / (counts + m)) * means + (m / (counts + m)) * global_mean
    popularity = pd.DataFrame(
        {
            "item_id": counts.index,
            "interaction_count": counts.to_numpy(),
            "mean_rating": means.to_numpy(),
            "popularity_score": score.to_numpy(),
        }
    )
    popularity = popularity.sort_values("popularity_score", ascending=False).reset_index(drop=True)
    popularity["rank"] = np.arange(1, len(popularity) + 1)
    return popularity.merge(items[["item_id", "title", "categories"]], on="item_id", how="left")


def build_category_popularity(popularity: pd.DataFrame) -> pd.DataFrame:
    """Explode the popularity ranking per content category.

    PopularityStrategy can restrict the cold-start ranking to one category
    (Table 9), and doing the explosion offline keeps that path free of
    per-request work.
    """
    exploded = popularity.explode("categories").dropna(subset=["categories"])
    exploded = exploded.rename(columns={"categories": "category"})
    exploded = exploded.sort_values(["category", "popularity_score"], ascending=[True, False])
    exploded["category_rank"] = exploded.groupby("category").cumcount() + 1
    return exploded[
        ["category", "item_id", "popularity_score", "interaction_count", "category_rank"]
    ].reset_index(drop=True)


def build_user_content_profiles(
    ratings: pd.DataFrame,
    item_embeddings: np.ndarray,
    maps: IndexMaps,
    settings: Settings,
) -> dict[int, np.ndarray]:
    """Average the embeddings of the items a user rated positively."""
    positives = ratings[ratings["rating"] >= settings.pipeline.positive_rating_threshold]
    item_position = maps.item_position
    profiles: dict[int, np.ndarray] = {}
    for user_id, group in positives.groupby("user_id"):
        rows = [item_position[int(i)] for i in group["item_id"] if int(i) in item_position]
        if rows:
            profiles[int(user_id)] = item_embeddings[rows].mean(axis=0).astype(np.float32)
    return profiles


def run(settings: Settings | None = None) -> Mapping[str, Any]:
    """Execute the transformation stage and write the feature tables."""
    settings = settings or load_settings()
    interim_root = settings.pipeline.interim_root
    output_dir = interim_root / "features"
    output_dir.mkdir(parents=True, exist_ok=True)

    ratings = _read_clean(interim_root, "ratings")
    items = _read_clean(interim_root, "items")
    tags_path = interim_root / "clean" / "tags.parquet"
    tags = pd.read_parquet(tags_path) if tags_path.exists() else None

    matrix, maps = build_interaction_matrix(ratings, settings)
    user_factors, item_factors, factorisation_backend = factorize(matrix, settings)

    # Item embeddings are aligned with the matrix column order, so every
    # downstream lookup is a position, never a join.
    items_aligned = (
        items.set_index("item_id")
        .reindex(maps.item_ids)
        .reset_index()
        .rename(columns={"index": "item_id"})
    )
    items_aligned["item_id"] = maps.item_ids
    item_embeddings, embedding_backend = build_item_content_embeddings(
        items_aligned, tags, settings
    )

    statistics = build_user_statistics(ratings)
    popularity = build_popularity_ranking(ratings, items)
    category_popularity = build_category_popularity(popularity)
    content_profiles = build_user_content_profiles(ratings, item_embeddings, maps, settings)

    seen = (
        ratings.groupby("user_id")["item_id"]
        .apply(lambda values: np.asarray(values, dtype=np.int64))
        .rename("seen_items")
        .reset_index()
    )
    preferred = (
        popularity.explode("categories")
        .dropna(subset=["categories"])
        .merge(ratings[["user_id", "item_id"]], on="item_id")
        .groupby(["user_id", "categories"])
        .size()
        .reset_index(name="n")
        .sort_values(["user_id", "n"], ascending=[True, False])
        .drop_duplicates("user_id")
        .rename(columns={"categories": "preferred_category"})[["user_id", "preferred_category"]]
    )

    user_position = maps.user_position
    user_features = pd.DataFrame({"user_id": maps.user_ids})
    user_features["latent_factors"] = [
        user_factors[user_position[int(u)]].tolist() for u in maps.user_ids
    ]
    user_features["content_profile"] = [
        content_profiles.get(int(u), np.zeros(item_embeddings.shape[1], dtype=np.float32)).tolist()
        for u in maps.user_ids
    ]
    user_features = (
        user_features.merge(statistics, on="user_id", how="left")
        .merge(seen, on="user_id", how="left")
        .merge(preferred, on="user_id", how="left")
    )
    user_features["seen_items"] = user_features["seen_items"].apply(
        lambda v: v.tolist() if isinstance(v, np.ndarray) else []
    )

    item_features = pd.DataFrame({"item_id": maps.item_ids})
    item_features["latent_factors"] = [row.tolist() for row in item_factors]
    item_features["content_embedding"] = [row.tolist() for row in item_embeddings]
    item_features = item_features.merge(
        popularity[["item_id", "popularity_score", "interaction_count", "rank"]],
        on="item_id",
        how="left",
    ).merge(items[["item_id", "title", "categories"]], on="item_id", how="left")
    item_features["popularity_score"] = item_features["popularity_score"].fillna(0.0)

    user_features.to_parquet(output_dir / "user_features.parquet", index=False)
    item_features.to_parquet(output_dir / "item_features.parquet", index=False)
    popularity.to_parquet(output_dir / "popularity.parquet", index=False)
    category_popularity.to_parquet(output_dir / "category_popularity.parquet", index=False)

    summary: dict[str, Any] = {
        "stage": "transform",
        "output_dir": str(output_dir),
        "matrix_shape": list(matrix.shape),
        "matrix_nnz": int(matrix.nnz),
        "factorisation_backend": factorisation_backend,
        "latent_factors": int(user_factors.shape[1]),
        "embedding_backend": embedding_backend,
        "embedding_dimension": int(item_embeddings.shape[1]),
        "users": int(len(user_features)),
        "items": int(len(item_features)),
        "categories": int(category_popularity["category"].nunique()),
    }
    write_report(output_dir / "transformation_report.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = build_parser("transform", __doc__ or "")
    args = parser.parse_args(argv)

    import os as _os

    if args.interim_root:
        _os.environ["RECO_INTERIM_ROOT"] = args.interim_root

    summary = run_stage("transform", run, settings=load_settings())
    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
