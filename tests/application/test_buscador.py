"""Buscador de películas por título."""

from __future__ import annotations

from application.buscador import BuscadorPeliculas

TITULOS = {
    1: "Matrix, The (1999)",
    2: "Amélie (Fabuleux destin d'Amélie Poulain, Le) (2001)",
    3: "Godfather, The (1972)",
    4: "Inglourious Basterds (2009)",
    5: "Star Wars: Episode IV - A New Hope (1977)",
    6: "Lone Star (1996)",
}
TITULOS_ES = {3: "El Padrino", 4: "Bastardos sin gloria", 5: "La guerra de las galaxias"}


def test_ignora_mayusculas_y_acentos() -> None:
    buscador = BuscadorPeliculas(TITULOS)
    assert buscador.buscar("MATRIX", 5) == [1]
    assert buscador.buscar("amelie", 5) == [2]


def test_encuentra_por_titulo_en_espanol() -> None:
    assert BuscadorPeliculas(TITULOS, TITULOS_ES).buscar("padrino", 5) == [3]


def test_ordena_por_popularidad_y_prefiere_inicio_de_palabra() -> None:
    buscador = BuscadorPeliculas(TITULOS, TITULOS_ES, orden_popularidad=[4, 6, 5])
    # "Bastardos" es la más popular, pero "star" no empieza ninguna de sus palabras.
    assert buscador.buscar("star", 5) == [6, 5, 4]


def test_textos_muy_cortos_no_buscan() -> None:
    assert BuscadorPeliculas(TITULOS).buscar(" a ", 5) == []
