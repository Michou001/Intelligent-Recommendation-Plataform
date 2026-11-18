"""Fichas de película en español: título, año, géneros, póster y sinopsis.

Los datos de TMDB los genera una sola vez ``data.pipeline.enriquecer_tmdb``.
Si ese archivo no existe la aplicación sigue funcionando: el título se toma
de MovieLens (separando el año) y los géneros se traducen con un diccionario.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

GENEROS_ES = {
    "Action": "Acción",
    "Adventure": "Aventura",
    "Animation": "Animación",
    "Children": "Infantil",
    "Comedy": "Comedia",
    "Crime": "Crimen",
    "Documentary": "Documental",
    "Drama": "Drama",
    "Fantasy": "Fantasía",
    "Film-Noir": "Cine negro",
    "Horror": "Terror",
    "IMAX": "IMAX",
    "Musical": "Musical",
    "Mystery": "Misterio",
    "Romance": "Romance",
    "Sci-Fi": "Ciencia ficción",
    "Thriller": "Suspenso",
    "War": "Bélica",
    "Western": "Western",
}

_ANIO = re.compile(r"\s*\((\d{4})\)\s*$")
# MovieLens escribe "Matrix, The (1999)"; el artículo vuelve al inicio.
_ARTICULO_AL_FINAL = re.compile(r"^(.*), (The|A|An|La|Le|Les|El|Il|Das|Der|Die)$")
_ALIAS = re.compile(r"\s*\(a\.k\.a\. [^)]*\)")


def separar_titulo(titulo_movielens: str) -> tuple[str, int | None]:
    """'Matrix, The (1999)' -> ('The Matrix', 1999)."""
    anio = None
    coincidencia = _ANIO.search(titulo_movielens)
    if coincidencia:
        anio = int(coincidencia.group(1))
        titulo_movielens = titulo_movielens[: coincidencia.start()]
    titulo = _ALIAS.sub("", titulo_movielens).strip()
    articulo = _ARTICULO_AL_FINAL.match(titulo)
    if articulo:
        titulo = f"{articulo.group(2)} {articulo.group(1)}"
    return titulo, anio


class CatalogoEspanol:
    def __init__(self, fichas_tmdb: dict[int, dict[str, Any]] | None = None) -> None:
        self._tmdb = fichas_tmdb or {}

    @classmethod
    def desde_csv(cls, ruta: Path) -> CatalogoEspanol:
        if not ruta.exists():
            return cls()
        # keep_default_na=False deja las celdas vacías como "" en lugar de NaN.
        tabla = pd.read_csv(ruta, keep_default_na=False)
        return cls({int(fila["item_id"]): fila for fila in tabla.to_dict(orient="records")})

    @property
    def con_tmdb(self) -> bool:
        return bool(self._tmdb)

    def titulos(self) -> dict[int, str]:
        """Títulos en español que trajo TMDB, para el buscador."""
        return {item_id: f["titulo"] for item_id, f in self._tmdb.items() if f.get("titulo")}

    def ficha(self, item_id: int, titulo: str | None, categorias: list[str]) -> dict[str, Any]:
        """Campos en español que completan un RecommendedItem."""
        original, anio = separar_titulo(titulo or "")
        tmdb = self._tmdb.get(item_id, {})
        return {
            "titulo": tmdb.get("titulo") or original or None,
            "anio": anio,
            "generos": [GENEROS_ES.get(c, c) for c in categorias if c != "(no genres listed)"],
            "poster": tmdb.get("poster") or None,
            "sinopsis": tmdb.get("sinopsis") or None,
        }


__all__ = ["GENEROS_ES", "CatalogoEspanol", "separar_titulo"]
