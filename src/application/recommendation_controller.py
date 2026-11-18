"""RecommendationController - orchestration of the inference cycle (FR-01, FR-04).

The controller contains no algorithmic logic (Section 3.1, Section 9.6). It
validates the request, asks ``FeatureStoreManager`` for the feature vectors,
invokes ``predict`` on the decorated strategy it was injected with, enriches
the ranking from the catalog, publishes the interaction event, and returns the
Top-K list.

Every collaborator arrives through an abstraction - ``IRecommendationStrategy``,
``FeatureStoreManager``, ``InteractionEventQueue``, and the segregated
repository interfaces - so this class can be unit tested against doubles
without a model, a broker, or a database (Section 7).

Degradation is concentrated here, and it is what makes NFR-05 hold: a feature
store that cannot be read, a user without history, or a model that fails to
load produce a successful degraded response served by the popularity
fallback, never an abrupt termination.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any

import numpy as np

from ai.interfaces import IEstrategiaConPerfil, IRecommendationStrategy, ScoredItem, UserFeatures
from application.buscador import BuscadorPeliculas
from application.catalogo_es import CatalogoEspanol
from application.error_journal import ErrorJournal
from application.tendencias import ContadorTendencias
from config.settings import Settings, load_settings
from data.feature_store_manager import FeatureStoreManager
from errors import (
    MENSAJES_ES,
    ColdStartError,
    EventPublishError,
    FeatureStoreUnavailableError,
    ModelUnavailableError,
    RecommendationPlatformError,
    UserNotFoundError,
)
from messaging.interaction_event_queue import InteractionEvent, InteractionEventQueue
from persistence.interfaces import (
    ICatalogRepository,
    IInteractionHistoryRepository,
    IUserProfileRepository,
)
from presentation.schemas import RecommendationResponse, RecommendedItem

#: Event type published when a ranking is served (FR-06).
SERVED_EVENT_TYPE = "recommendation_served"


def _fase_de(features: UserFeatures | None, es_nuevo: bool) -> str:
    """Qué tanto conoce el sistema al usuario (se muestra en la interfaz)."""
    if features is None or features.is_cold_start:
        return "arranque_en_frio"
    return "calentamiento" if es_nuevo else "personalizado"


#: Eventos que agregan o quitan una película de las que le gustan al usuario.
ME_GUSTA = "me_gusta"
YA_NO_ME_GUSTA = "ya_no_me_gusta"


class RecommendationController:
    """Orchestrates one recommendation request end to end."""

    def __init__(
        self,
        strategy: IRecommendationStrategy,
        feature_store: FeatureStoreManager,
        *,
        event_queue: InteractionEventQueue | None = None,
        catalog_repository: ICatalogRepository | None = None,
        profile_repository: IUserProfileRepository | None = None,
        fallback_strategy: IRecommendationStrategy | None = None,
        settings: Settings | None = None,
        logger: Any | None = None,
        startup_degradation_reason: str | None = None,
        error_journal: ErrorJournal | None = None,
        history_repository: IInteractionHistoryRepository | None = None,
        estrategia_calentamiento: IEstrategiaConPerfil | None = None,
        catalogo_es: CatalogoEspanol | None = None,
        tendencias: ContadorTendencias | None = None,
        buscador: BuscadorPeliculas | None = None,
        mas_vistas_por_genero: dict[str, list[int]] | None = None,
    ) -> None:
        self._buscador = buscador
        self._mas_vistas_por_genero = mas_vistas_por_genero or {}
        self._historial = history_repository
        self._calentamiento = estrategia_calentamiento
        self._catalogo_es = catalogo_es or CatalogoEspanol()
        self._tendencias = tendencias
        self._strategy = strategy
        self._feature_store = feature_store
        self._event_queue = event_queue
        self._catalog = catalog_repository
        self._profiles = profile_repository
        self._fallback = fallback_strategy
        self._settings = settings or load_settings()
        self._logger = logger
        self._journal = error_journal
        # Set by the composition root when the configured strategy could not be
        # loaded at startup and the fallback took its place. That degradation
        # lasts for the life of the process, so every response has to declare
        # it: otherwise the one case where the system is *persistently*
        # degraded would be the only one the client cannot see (NFR-05).
        self._startup_degradation_reason = startup_degradation_reason
        # Latencias recientes de cada recomendación servida, sea del modelo
        # principal o del fold-in de un usuario nuevo (la vista de métricas).
        self._latencias: deque[float] = deque(maxlen=2048)

    @property
    def strategy(self) -> IRecommendationStrategy:
        """The decorated object the controller invokes."""
        return self._strategy

    def recommend(self, user_id: int, k: int | None = None) -> RecommendationResponse:
        """Serve the Top-K ranking for a user.

        Raises:
            ValueError: ``user_id`` or ``k`` is outside the accepted range.
                Fail Fast happens before any collaborator is touched.
            UserNotFoundError: The profile repository does not know the user.
        """
        started = time.perf_counter()
        k = self._validate(user_id, k)

        self._assert_user_exists(user_id)

        runtime_reason: str | None = None
        features, es_nuevo = self._load_features(user_id)
        fase = _fase_de(features, es_nuevo)
        if features is None:
            runtime_reason = FeatureStoreUnavailableError.code
            features = UserFeatures(
                user_id=user_id,
                is_cold_start=True,
                feature_store_version=self._safe_store_version(),
            )
        elif features.is_cold_start:
            runtime_reason = ColdStartError.code

        estrategia = self._calentamiento if fase == "calentamiento" else self._strategy
        ranking, strategy_used, reason = self._rank(features, k, runtime_reason, estrategia)
        # A failure during this request takes precedence over the startup one,
        # because it is the more specific explanation of what the client got.
        degradation_reason = reason or self._startup_degradation_reason

        if degradation_reason is not None and self._journal is not None:
            self._journal.record(
                "degradation",
                degradation_reason,
                MENSAJES_ES.get(degradation_reason, degradation_reason),
                route=f"/recommendations/{user_id}",
                user_id=user_id,
                status=200,
                strategy=strategy_used.strategy_id,
            )

        response = RecommendationResponse(
            user_id=user_id,
            k=k,
            items=self._enrich(ranking),
            strategy_id=strategy_used.strategy_id,
            model_version=strategy_used.model_version,
            feature_store_version=features.feature_store_version,
            degraded=degradation_reason is not None,
            degradation_reason=degradation_reason,
            latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
            fase=fase,
        )

        self._latencias.append(response.latency_ms)
        self._publish_served_event(response)
        return response

    def latencia_p95(self) -> float | None:
        """p95 en milisegundos de las últimas recomendaciones, o None si no hay."""
        if not self._latencias:
            return None
        return float(np.percentile(np.fromiter(self._latencias, dtype=np.float64), 95.0))

    def tendencias(self, k: int) -> tuple[list[RecommendedItem], bool]:
        """Top-k del momento y si viene de actividad en vivo.

        Sin interacciones recientes (por ejemplo, recién arrancado el servidor)
        se responde con la popularidad histórica, para no mostrar una fila vacía.
        """
        recientes = self._tendencias.top(k) if self._tendencias else []
        if recientes:
            ranking = [ScoredItem(item, float(puntos)) for item, puntos in recientes]
            return self._enrich(ranking), True
        fallback = self._resolve_fallback()
        if fallback is None:
            return [], False
        anonimo = UserFeatures(user_id=0, is_cold_start=True)
        return self._enrich(fallback.predict(anonimo, k)), False

    def tendencias_por_sexo(self, k: int) -> dict[str, list[RecommendedItem]]:
        if self._tendencias is None:
            return {}
        return {
            sexo: self._enrich([ScoredItem(i, float(p)) for i, p in self._tendencias.top(k, sexo)])
            for sexo in ("mujer", "hombre")
        }

    def peliculas_para_registro(self, generos: list[str]) -> list[RecommendedItem]:
        """Las más vistas de cada género elegido, para que marque las que le gustaron.

        Se muestran las más vistas y no las mejor calificadas porque la gente
        reconoce más títulos: en la evaluación offline marcan casi el doble.
        """
        ids = [i for g in generos for i in self._mas_vistas_por_genero.get(g, [])]
        return self._enrich([ScoredItem(i, 0.0) for i in dict.fromkeys(ids)])

    def buscar(self, texto: str, k: int) -> list[RecommendedItem]:
        if self._buscador is None:
            return []
        return self._enrich([ScoredItem(i, 0.0) for i in self._buscador.buscar(texto, k)])

    def _validate(self, user_id: int, k: int | None) -> int:
        """Fail Fast: reject an invalid request before any computation."""
        if not isinstance(user_id, int) or isinstance(user_id, bool) or user_id < 1:
            raise ValueError("user_id must be a positive integer")
        resolved = self._settings.model.default_k if k is None else k
        if not isinstance(resolved, int) or isinstance(resolved, bool) or resolved < 1:
            raise ValueError("k must be a positive integer")
        if resolved > self._settings.model.max_k:
            raise ValueError(f"k must not exceed {self._settings.model.max_k}")
        return resolved

    def _assert_user_exists(self, user_id: int) -> None:
        """A user the profile repository does not know is a 404, not a fallback."""
        if self._profiles is None:
            return
        try:
            known = self._profiles.exists(user_id)
        except Exception as exc:  # noqa: BLE001 - an unreachable profile store must not 500
            self._log("warning", "profile.lookup_failed", user_id=user_id, error=str(exc))
            return
        if not known:
            raise UserNotFoundError(f"user {user_id} does not exist")

    def _load_features(self, user_id: int) -> tuple[UserFeatures | None, bool]:
        """Retrieve the precomputed vectors, or None when the store is down.

        El segundo valor indica si es un usuario nuevo (sin fila en el Feature Store).
        """
        try:
            return self._feature_store.get_user_features(user_id), False
        except ColdStartError:
            return self._caracteristicas_de_usuario_nuevo(user_id), True
        except FeatureStoreUnavailableError as exc:
            self._log("error", "feature_store.unavailable", user_id=user_id, detail=exc.detail)
            return None, False
        except Exception as exc:  # noqa: BLE001 - any storage failure degrades, never 500
            self._log("error", "feature_store.failed", user_id=user_id, error=str(exc))
            return None, False

    def _caracteristicas_de_usuario_nuevo(self, user_id: int) -> UserFeatures:
        """Lo que se sabe de un usuario que el pipeline offline todavía no procesó.

        - Arranque en frío: no marcó ninguna película; se le recomienda lo más
          popular del primer género que eligió al registrarse.
        - Calentamiento: con las películas que eligió en el registro y sus
          "me gusta" se calcula su vector latente con fold-in de ALS, sin
          reentrenar. En la evaluación offline triplica la precisión de la
          popularidad (reports/arranque_en_frio.json).
        """
        perfil = self._profiles.get_by_id(user_id) if self._profiles else None
        generos = perfil.preferred_categories if perfil else ()
        gustadas = self.peliculas_gustadas(user_id)
        vector = None
        if self._calentamiento is not None and gustadas:
            vector = self._calentamiento.perfil_desde_peliculas(gustadas)
        return UserFeatures(
            user_id=user_id,
            latent_factors=vector,
            seen_items=tuple(gustadas),
            preferred_category=generos[0] if generos else None,
            is_cold_start=vector is None,
            feature_store_version=self._safe_store_version(),
        )

    def peliculas_gustadas(self, user_id: int) -> list[int]:
        """Las elegidas en el registro más los "me gusta", menos los que quitó.

        Se recorre el historial en orden: un "ya_no_me_gusta" posterior quita la
        película, y volver a marcarla la pone otra vez (al final, como la más
        reciente).
        """
        perfil = self._profiles.get_by_id(user_id) if self._profiles else None
        iniciales = (perfil.attributes.get("peliculas_iniciales", "") if perfil else "").split(",")
        gustadas = dict.fromkeys(int(i) for i in iniciales if i)
        if self._historial is not None:
            try:
                registros = self._historial.list_by_user(user_id, limit=500)
            except Exception as exc:  # noqa: BLE001 - sin historial quedan las del registro
                self._log("warning", "history.lookup_failed", user_id=user_id, error=str(exc))
                registros = []
            for registro in sorted(registros, key=lambda r: r.occurred_at):
                if registro.item_id is None:
                    continue
                if registro.event_type in (ME_GUSTA, YA_NO_ME_GUSTA):
                    gustadas.pop(registro.item_id, None)
                if registro.event_type == ME_GUSTA:
                    gustadas[registro.item_id] = None
        return list(gustadas)

    def parecidas(self, item_id: int, k: int) -> list[RecommendedItem]:
        """Películas que suelen gustarle a quien le gustó esta.

        Es el mismo fold-in de ALS de un usuario nuevo, pero con una sola
        película: el vector resultante apunta a lo que comparte público con ella.
        """
        if self._calentamiento is None:
            return []
        vector = self._calentamiento.perfil_desde_peliculas([item_id])
        if vector is None:
            return []
        perfil = UserFeatures(user_id=0, latent_factors=vector, seen_items=(item_id,))
        return self._enrich(self._calentamiento.predict(perfil, k))

    def ficha(self, item_id: int) -> RecommendedItem | None:
        """Título, póster y datos en español de una película."""
        fichas = self._enrich([ScoredItem(item_id, 0.0)])
        return fichas[0] if fichas else None

    def populares_de_genero(self, genero: str, k: int) -> list[RecommendedItem]:
        ids = self._mas_vistas_por_genero.get(genero, [])[:k]
        return self._enrich([ScoredItem(i, 0.0) for i in ids])

    def _rank(
        self,
        features: UserFeatures,
        k: int,
        reason: str | None,
        estrategia: IRecommendationStrategy | None = None,
    ) -> tuple[list[ScoredItem], IRecommendationStrategy, str | None]:
        """Run the configured strategy, falling back on a degradable error."""
        estrategia = estrategia or self._strategy
        if reason is None:
            try:
                return estrategia.predict(features, k), estrategia, None
            except RecommendationPlatformError as exc:
                if not exc.degradable:
                    raise
                self._log("warning", "strategy.degraded", code=exc.code, detail=exc.detail)
                reason = exc.code
            except Exception as exc:  # noqa: BLE001 - an unexpected model failure degrades too
                self._log("error", "strategy.failed", error=f"{type(exc).__name__}: {exc}")
                reason = ModelUnavailableError.code

        fallback = self._resolve_fallback()
        if fallback is None:
            raise ModelUnavailableError(
                "no recommendation could be produced and no fallback is configured",
                detail=reason,
            )
        try:
            return fallback.predict(features, k), fallback, reason
        except Exception as exc:  # noqa: BLE001 - the last resort is an empty ranking
            self._log("error", "fallback.failed", error=f"{type(exc).__name__}: {exc}")
            return [], fallback, reason or ModelUnavailableError.code

    def _resolve_fallback(self) -> IRecommendationStrategy | None:
        """The configured fallback, or the active strategy when it is already it."""
        if self._fallback is not None:
            return self._fallback
        if self._strategy.strategy_id == self._settings.model.fallback_strategy:
            return self._strategy
        return None

    def _enrich(self, ranking: list[ScoredItem]) -> list[RecommendedItem]:
        """Attach catalog titles and categories to the ranked identifiers."""
        titles: dict[int, tuple[str, list[str]]] = {}
        if self._catalog is not None and ranking:
            try:
                for item in self._catalog.get_items([entry.item_id for entry in ranking]):
                    titles[int(item.item_id)] = (item.title, list(item.categories))
            except Exception as exc:  # noqa: BLE001 - a ranking without titles still serves
                self._log("warning", "catalog.lookup_failed", error=str(exc))

        enriched: list[RecommendedItem] = []
        for position, entry in enumerate(ranking, start=1):
            title, categories = titles.get(entry.item_id, (None, []))
            enriched.append(
                RecommendedItem(
                    item_id=entry.item_id,
                    rank=position,
                    score=entry.score,
                    title=title,
                    categories=categories,
                    **self._catalogo_es.ficha(entry.item_id, title, categories),
                )
            )
        return enriched

    def _publish_served_event(self, response: RecommendationResponse) -> None:
        """Hand the interaction to the broker; never let it affect the response.

        Section 5.3: this branch runs outside the request/response cycle and
        is guaranteed by the broker, so a publication failure is logged and
        buffered, and the user still receives their ranking (Table 10).
        """
        if self._event_queue is None:
            return
        event = InteractionEvent(
            user_id=response.user_id,
            event_type=SERVED_EVENT_TYPE,
            recommended_items=tuple(item.item_id for item in response.items),
            model_version=response.model_version,
            metadata={
                "strategy_id": response.strategy_id,
                "degraded": str(response.degraded).lower(),
                "k": str(response.k),
            },
        )
        try:
            self._event_queue.publish(event)
        except EventPublishError as exc:
            self._log("warning", "event.publish_failed", code=exc.code, detail=exc.detail)
            self._record_publish_failure(response, exc.code, exc.detail)
        except Exception as exc:  # noqa: BLE001 - publication never breaks a served response
            self._log("warning", "event.publish_failed", error=f"{type(exc).__name__}: {exc}")
            self._record_publish_failure(response, EventPublishError.code, str(exc))

    def _record_publish_failure(
        self, response: RecommendationResponse, code: str, detail: str | None
    ) -> None:
        if self._journal is not None:
            self._journal.record(
                "degradation",
                code,
                MENSAJES_ES.get(code, code),
                route=f"/recommendations/{response.user_id}",
                user_id=response.user_id,
                status=200,
                detail=detail,
            )

    def _safe_store_version(self) -> str:
        try:
            return self._feature_store.version
        except Exception:  # noqa: BLE001 - a broken store still reports a response
            return "unknown"

    def _log(self, level: str, event: str, **fields: Any) -> None:
        if self._logger is None:
            return
        getattr(self._logger, level, lambda *a, **k: None)(event, **fields)


__all__ = ["RecommendationController"]
