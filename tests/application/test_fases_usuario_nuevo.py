"""Un usuario nuevo pasa de arranque en frío a calentamiento (ALS fold-in)."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from ai.strategies.collaborative_filtering_strategy import CollaborativeFilteringStrategy
from application.recommendation_controller import RecommendationController
from application.tendencias import ContadorTendencias
from config.settings import Settings
from persistence.interfaces import IInteractionHistoryRepository, InteractionRecord, UserProfile
from presentation.schemas import InteractionRequest
from tests.application.test_recommendation_controller import (
    ColdFeatureStore,
    StubCatalog,
    StubProfiles,
)
from tests.conftest import StubStrategy

# Cuatro películas en un espacio latente de 2 dimensiones: 1 y 2 se parecen
# entre sí, 3 y 4 también.
ALS = CollaborativeFilteringStrategy(
    np.array([1, 2, 3, 4]),
    np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]], dtype=np.float32),
)


class HistorialEnMemoria(IInteractionHistoryRepository):
    def __init__(self) -> None:
        self.registros: list[InteractionRecord] = []

    def me_gusta(self, user_id: int, item_id: int) -> None:
        self.registros.append(
            InteractionRecord(
                event_id=f"{user_id}-{item_id}-{len(self.registros)}",
                user_id=user_id,
                event_type="me_gusta",
                occurred_at=datetime.now(UTC),
                item_id=item_id,
            )
        )

    def ya_no_me_gusta(self, user_id: int, item_id: int) -> None:
        self.me_gusta(user_id, item_id)
        self.registros[-1] = InteractionRecord(
            event_id=f"{user_id}-{item_id}-quitar-{len(self.registros)}",
            user_id=user_id,
            event_type="ya_no_me_gusta",
            occurred_at=datetime.now(UTC),
            item_id=item_id,
        )

    def append(self, record: InteractionRecord) -> bool:
        self.registros.append(record)
        return True

    def list_by_user(self, user_id: int, limit: int = 100) -> list[InteractionRecord]:
        return [r for r in self.registros if r.user_id == user_id][:limit]

    def count_by_user(self, user_id: int) -> int:
        return len(self.list_by_user(user_id))

    def exists_event(self, event_id: str) -> bool:
        return False


class PerfilesDelRegistro(StubProfiles):
    def __init__(self, peliculas_iniciales: str = "") -> None:
        super().__init__({700})
        self._iniciales = peliculas_iniciales

    def get_by_id(self, user_id: int) -> UserProfile | None:
        return UserProfile(
            user_id=user_id,
            preferred_categories=("Comedy",),
            attributes={"peliculas_iniciales": self._iniciales},
        )


def armar(
    settings: Settings, historial: HistorialEnMemoria, iniciales: str = ""
) -> RecommendationController:
    return RecommendationController(
        strategy=StubStrategy(),
        feature_store=ColdFeatureStore(),
        catalog_repository=StubCatalog(),
        profile_repository=PerfilesDelRegistro(iniciales),
        fallback_strategy=StubStrategy([901, 902, 903], model_version="pop-v1"),
        settings=settings,
        history_repository=historial,
        estrategia_calentamiento=ALS,
    )


def test_el_fold_in_acerca_al_usuario_a_lo_que_eligio() -> None:
    vector = ALS.perfil_desde_peliculas([1])
    assert vector is not None
    # Le gustó la 1: la 2 (parecida) debe puntuar más que la 3 (distinta).
    puntajes = ALS._item_factors @ vector
    assert puntajes[1] > puntajes[2]
    assert ALS.perfil_desde_peliculas([999]) is None


def test_sin_peliculas_recibe_popularidad_de_su_genero(settings: Settings) -> None:
    respuesta = armar(settings, HistorialEnMemoria()).recommend(700, 3)
    assert respuesta.fase == "arranque_en_frio"
    assert respuesta.degradation_reason == "cold_start"
    assert [i.item_id for i in respuesta.items] == [901, 902, 903]


def test_las_peliculas_del_registro_personalizan_desde_la_primera_peticion(
    settings: Settings,
) -> None:
    respuesta = armar(settings, HistorialEnMemoria(), iniciales="1").recommend(700, 1)
    assert respuesta.fase == "calentamiento"
    assert respuesta.strategy_id == "als"
    assert respuesta.degraded is False
    # La elegida no se repite y la más parecida va primero.
    assert [i.item_id for i in respuesta.items] == [2]


def test_la_latencia_cuenta_tambien_a_los_usuarios_nuevos(settings: Settings) -> None:
    controlador = armar(settings, HistorialEnMemoria(), iniciales="1")
    assert controlador.latencia_p95() is None
    controlador.recommend(700, 1)  # sirve el fold-in, no la estrategia principal
    latencia = controlador.latencia_p95()
    assert latencia is not None and latencia >= 0


def test_los_me_gusta_se_suman_a_las_del_registro(settings: Settings) -> None:
    historial = HistorialEnMemoria()
    historial.me_gusta(700, 3)
    historial.me_gusta(700, 3)  # repetido: cuenta una vez
    controlador = armar(settings, historial, iniciales="1")
    assert controlador.peliculas_gustadas(700) == [1, 3]
    respuesta = controlador.recommend(700, 4)
    assert {i.item_id for i in respuesta.items} == {2, 4}


def test_tendencias_sin_actividad_usa_la_popularidad_historica(settings: Settings) -> None:
    items, en_vivo = armar(settings, HistorialEnMemoria()).tendencias(2)
    assert en_vivo is False
    assert [i.item_id for i in items] == [901, 902]


def test_tendencias_en_vivo_y_por_sexo(settings: Settings) -> None:
    sexos = {1: "mujer", 2: "hombre"}
    contador = ContadorTendencias(sexo_de=sexos.get)
    controlador = RecommendationController(
        strategy=StubStrategy(),
        feature_store=ColdFeatureStore(),
        catalog_repository=StubCatalog(),
        settings=settings,
        tendencias=contador,
    )
    contador.registrar(InteractionRequest(user_id=1, event_type="me_gusta", item_id=42))
    contador.registrar(InteractionRequest(user_id=2, event_type="clic", item_id=7))
    items, en_vivo = controlador.tendencias(5)
    assert en_vivo is True
    assert items[0].item_id == 42
    assert items[0].titulo == "Item 42"
    por_sexo = controlador.tendencias_por_sexo(5)
    assert [i.item_id for i in por_sexo["mujer"]] == [42]
    assert [i.item_id for i in por_sexo["hombre"]] == [7]


def test_quitar_un_me_gusta_lo_saca_de_sus_peliculas(settings: Settings) -> None:
    historial = HistorialEnMemoria()
    historial.me_gusta(700, 3)
    historial.ya_no_me_gusta(700, 3)
    historial.ya_no_me_gusta(700, 1)  # también se puede quitar una del registro
    controlador = armar(settings, historial, iniciales="1,2")
    assert controlador.peliculas_gustadas(700) == [2]


def test_volver_a_marcarla_la_pone_como_la_mas_reciente(settings: Settings) -> None:
    historial = HistorialEnMemoria()
    historial.ya_no_me_gusta(700, 1)
    historial.me_gusta(700, 1)
    controlador = armar(settings, historial, iniciales="1,2")
    assert controlador.peliculas_gustadas(700) == [2, 1]


def test_parecidas_usa_el_fold_in_con_una_sola_pelicula(settings: Settings) -> None:
    controlador = armar(settings, HistorialEnMemoria())
    parecidas = [p.item_id for p in controlador.parecidas(1, 2)]
    assert parecidas[0] == 2  # la 1 y la 2 se parecen
    assert 1 not in parecidas
    assert controlador.parecidas(999, 2) == []  # película que el modelo no conoce
