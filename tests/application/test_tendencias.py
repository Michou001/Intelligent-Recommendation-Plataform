"""Top del momento: ventana deslizante de interacciones."""

from __future__ import annotations

from application.tendencias import ContadorTendencias
from presentation.schemas import InteractionRequest


class Reloj:
    def __init__(self) -> None:
        self.ahora = 1_000_000.0

    def __call__(self) -> float:
        return self.ahora


def evento(item_id: int, tipo: str = "me_gusta") -> InteractionRequest:
    return InteractionRequest(user_id=1, event_type=tipo, item_id=item_id)


def test_un_me_gusta_pesa_mas_que_un_clic() -> None:
    contador = ContadorTendencias(minutos=30)
    contador.registrar(evento(10, "clic"))
    contador.registrar(evento(10, "clic"))
    contador.registrar(evento(20, "me_gusta"))
    assert contador.top(2) == [(20, 3), (10, 2)]


def test_los_eventos_viejos_salen_de_la_ventana() -> None:
    reloj = Reloj()
    contador = ContadorTendencias(minutos=30, reloj=reloj)
    contador.registrar(evento(10))
    reloj.ahora += 31 * 60
    contador.registrar(evento(20))
    assert contador.top(5) == [(20, 3)]


def test_otros_eventos_no_cuentan() -> None:
    contador = ContadorTendencias()
    contador.registrar(evento(10, "recommendation_served"))
    contador.registrar(InteractionRequest(user_id=1, event_type="me_gusta"))
    assert contador.top(5) == []


def test_actividad_por_minuto() -> None:
    reloj = Reloj()
    contador = ContadorTendencias(minutos=3, reloj=reloj)
    contador.registrar(evento(10))
    reloj.ahora += 60
    contador.registrar(evento(10, "clic"))
    contador.registrar(evento(11, "clic"))
    assert contador.actividad_por_minuto() == [0, 1, 2]
