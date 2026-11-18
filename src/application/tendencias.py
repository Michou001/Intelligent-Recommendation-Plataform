"""Lo más popular "del momento": interacciones de los últimos minutos.

La popularidad del pipeline offline es histórica (todas las calificaciones de
MovieLens). Este contador es la versión en vivo: una ventana deslizante que
solo recuerda lo que pasó en los últimos N minutos, así que el top cambia
mientras la gente usa la aplicación desde varias máquinas.

MovieLens no trae datos demográficos, así que la popularidad por sexo solo
puede salir de los usuarios de la plataforma que lo indicaron al registrarse.
"""

from __future__ import annotations

import threading
import time
from collections import Counter, deque
from collections.abc import Callable

from presentation.schemas import InteractionRequest

# Un "me gusta" dice más del interés que abrir la ficha de una película.
PESOS = {"me_gusta": 3, "clic": 1}


class ContadorTendencias:
    def __init__(
        self,
        minutos: int = 30,
        reloj: Callable[[], float] = time.time,
        sexo_de: Callable[[int], str | None] = lambda user_id: None,
    ) -> None:
        self.minutos = minutos
        self._reloj = reloj
        self._sexo_de = sexo_de
        # (momento, item_id, peso, sexo de quien interactuó o None)
        self._eventos: deque[tuple[float, int, int, str | None]] = deque()
        self._candado = threading.Lock()

    def registrar(self, interaccion: InteractionRequest) -> None:
        """Observador de la ingesta: se llama con cada interacción aceptada."""
        peso = PESOS.get(interaccion.event_type)
        if peso is None or interaccion.item_id is None:
            return
        sexo = self._sexo_de(interaccion.user_id)
        with self._candado:
            self._eventos.append((self._reloj(), interaccion.item_id, peso, sexo))

    def top(self, k: int, sexo: str | None = None) -> list[tuple[int, int]]:
        """Las k películas con más puntos en la ventana, como (item_id, puntos).

        Con ``sexo`` solo cuentan las interacciones de usuarios que lo indicaron.
        """
        conteo: Counter[int] = Counter()
        with self._candado:
            self._olvidar_viejos()
            for _, item_id, peso, sexo_evento in self._eventos:
                if sexo is None or sexo == sexo_evento:
                    conteo[item_id] += peso
        return conteo.most_common(k)

    def actividad_por_minuto(self) -> list[int]:
        """Interacciones de cada minuto de la ventana, de la más antigua a la actual."""
        ahora = self._reloj()
        cubetas = [0] * self.minutos
        with self._candado:
            self._olvidar_viejos()
            for momento, *_ in self._eventos:
                hace = int((ahora - momento) // 60)
                cubetas[self.minutos - 1 - min(hace, self.minutos - 1)] += 1
        return cubetas

    def _olvidar_viejos(self) -> None:
        limite = self._reloj() - self.minutos * 60
        while self._eventos and self._eventos[0][0] < limite:
            self._eventos.popleft()


__all__ = ["PESOS", "ContadorTendencias"]
