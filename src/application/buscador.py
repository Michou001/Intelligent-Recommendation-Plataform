"""Buscador de películas por título, sin importar mayúsculas ni acentos.

El índice se arma una vez al arrancar con el título original de MovieLens y el
título en español de TMDB, así "matrix", "Matrix" y "el padrino" encuentran su
película. Va ordenado por popularidad, para que al buscar "star" salga primero
Star Wars y no una película que casi nadie vio.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable

MIN_CARACTERES = 2


def normalizar(texto: str) -> str:
    """'Amélie' -> 'amelie': sin acentos y en minúsculas."""
    descompuesto = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in descompuesto if not unicodedata.combining(c)).casefold()


class BuscadorPeliculas:
    def __init__(
        self,
        titulos: dict[int, str],
        titulos_es: dict[int, str] | None = None,
        orden_popularidad: Iterable[int] = (),
    ) -> None:
        titulos_es = titulos_es or {}
        populares = [i for i in dict.fromkeys(orden_popularidad) if i in titulos]
        ya_ordenados = set(populares)
        resto = [i for i in titulos if i not in ya_ordenados]
        self._indice = [
            (item_id, normalizar(f"{titulos[item_id]} {titulos_es.get(item_id, '')}"))
            for item_id in populares + resto
        ]

    def buscar(self, texto: str, k: int) -> list[int]:
        consulta = normalizar(texto.strip())
        if len(consulta) < MIN_CARACTERES:
            return []
        coincidencias = [(item, titulo) for item, titulo in self._indice if consulta in titulo]
        # Primero donde una palabra empieza con lo buscado: "star" -> Star Wars
        # antes que "Bastardos". El orden de popularidad se respeta en cada grupo.
        coincidencias.sort(key=lambda par: f" {consulta}" not in f" {par[1]}")
        return [item_id for item_id, _ in coincidencias[:k]]


__all__ = ["BuscadorPeliculas", "normalizar"]
