"""CollaborativeFilteringStrategy - Alternating Least Squares (FR-04, Table 9).

Scores every catalog item as the dot product between the user latent vector
served by the Feature Store and the item latent factors produced offline by
the implicit-feedback factorisation of ``data/pipeline/transform.py``.

The strategy performs no training: the factorisation is an offline stage and
this class only consumes its output, so the inference path never fits a model
(Section 9.5).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ai.interfaces import IEstrategiaConPerfil, ItemId, ScoredItem, UserFeatures, top_k
from ai.model_factory import register_strategy
from ai.strategies import load_feature_table
from config.settings import Settings, load_settings
from errors import ColdStartError, ModelUnavailableError

#: Un "me gusta" equivale a una calificación de 4.5 (el entrenamiento usa la
#: calificación como confianza de cada interacción positiva).
CONFIANZA_ME_GUSTA = 4.5


@register_strategy("als")
class CollaborativeFilteringStrategy(IEstrategiaConPerfil):
    """Implicit-feedback matrix factorisation served as a matrix product.

    Scoring is one dense ``(n_items, d) @ (d,)`` product followed by
    ``numpy.argpartition``; nothing is sorted over the whole catalog
    (Section 9.7, third efficiency technique).
    """

    def __init__(
        self,
        item_ids: np.ndarray,
        item_factors: np.ndarray,
        *,
        model_version: str = "unknown",
        regularizacion: float = 0.05,
    ) -> None:
        self._item_ids = np.asarray(item_ids, dtype=np.int64)
        self._item_factors = np.ascontiguousarray(item_factors, dtype=np.float32)
        self._model_version = model_version
        self._regularizacion = regularizacion
        # YᵀY no depende del usuario: se calcula una vez para el fold-in.
        self._yty = self._item_factors.T @ self._item_factors
        if self._item_factors.shape[0] != self._item_ids.shape[0]:
            raise ModelUnavailableError(
                "item factors and item identifiers have different lengths",
                detail=f"{self._item_factors.shape[0]} factors vs {self._item_ids.shape[0]} ids",
            )

    @classmethod
    def build(cls, settings: Settings | None = None) -> CollaborativeFilteringStrategy:
        """Load the item latent factors of the configured feature-store version."""
        settings = settings or load_settings()
        frame = load_feature_table(settings.feature_store.version_dir, "item_features")
        if frame is None or frame.empty:
            raise ModelUnavailableError("the item feature table is empty")
        if "latent_factors" not in frame.columns:
            raise ModelUnavailableError("the item feature table carries no latent factors")

        item_ids = frame["item_id"].to_numpy(dtype=np.int64)
        item_factors = np.vstack(
            [np.asarray(row, dtype=np.float32) for row in frame["latent_factors"]]
        )
        return cls(
            item_ids,
            item_factors,
            model_version=settings.feature_store.version,
            regularizacion=settings.pipeline.als_regularization,
        )

    @property
    def dimension(self) -> int:
        """Number of latent factors the strategy expects in a user vector."""
        return int(self._item_factors.shape[1])

    def predict(self, user_features: UserFeatures, k: int = 10) -> list[ScoredItem]:
        """Return the K items with the highest latent-space affinity (FR-04)."""
        user_vector = user_features.latent_factors
        if user_vector is None or user_vector.size == 0:
            raise ColdStartError(
                f"user {user_features.user_id} has no latent factors",
                detail="collaborative filtering needs interaction history",
            )
        if user_vector.shape[0] != self.dimension:
            raise ModelUnavailableError(
                "user vector and item factors have incompatible dimensions",
                detail=f"user={user_vector.shape[0]} items={self.dimension}",
            )

        scores = self._item_factors @ np.asarray(user_vector, dtype=np.float32)
        scores = _mask_seen(self._item_ids, scores, user_features.seen_items)
        return top_k(scores, self._item_ids, k)

    def perfil_desde_peliculas(self, item_ids: Sequence[ItemId]) -> np.ndarray | None:
        """Fold-in: vector latente de un usuario nuevo sin reentrenar el modelo.

        Es el mismo paso que ALS hace para cada usuario durante el
        entrenamiento, con los factores de las películas ya fijos:

            u = (YᵀY + Yₛᵀ(c - 1)Yₛ + λI)⁻¹ · Yₛᵀc

        Yₛ son los factores de las películas que le gustaron y c la confianza
        de un "me gusta" (en el entrenamiento la confianza es la calificación).
        """
        filas = np.isin(self._item_ids, np.asarray(list(item_ids), dtype=np.int64))
        if not filas.any():
            return None
        gustadas = self._item_factors[filas]
        a = (
            self._yty
            + (CONFIANZA_ME_GUSTA - 1.0) * gustadas.T @ gustadas
            + self._regularizacion * np.eye(self.dimension, dtype=np.float32)
        )
        b = CONFIANZA_ME_GUSTA * gustadas.sum(axis=0)
        return np.linalg.solve(a, b).astype(np.float32)

    def __len__(self) -> int:
        return int(self._item_ids.size)


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


__all__ = ["CollaborativeFilteringStrategy"]
