"""Fichas en español a partir de los títulos de MovieLens y de TMDB."""

from __future__ import annotations

from pathlib import Path

import pytest

from application.catalogo_es import CatalogoEspanol, separar_titulo


@pytest.mark.parametrize(
    ("movielens", "esperado"),
    [
        ("Toy Story (1995)", ("Toy Story", 1995)),
        ("Matrix, The (1999)", ("The Matrix", 1999)),
        (
            "Léon: The Professional (a.k.a. The Professional) (Léon) (1994)",
            ("Léon: The Professional (Léon)", 1994),
        ),
        ("Sin año", ("Sin año", None)),
    ],
)
def test_separar_titulo(movielens: str, esperado: tuple[str, int | None]) -> None:
    assert separar_titulo(movielens) == esperado


def test_sin_tmdb_traduce_generos_y_usa_el_titulo_original() -> None:
    ficha = CatalogoEspanol().ficha(1, "Toy Story (1995)", ["Animation", "Comedy"])
    assert ficha == {
        "titulo": "Toy Story",
        "anio": 1995,
        "generos": ["Animación", "Comedia"],
        "poster": None,
        "sinopsis": None,
    }


def test_con_tmdb_usa_el_titulo_en_espanol(tmp_path: Path) -> None:
    csv = tmp_path / "peliculas_es.csv"
    csv.write_text(
        "item_id,titulo,poster,sinopsis\n"
        "1,Toy Story (Juguetes),https://image.tmdb.org/t/p/w342/a.jpg,Los juguetes cobran vida.\n"
        "2,,,\n",
        encoding="utf-8",
    )
    catalogo = CatalogoEspanol.desde_csv(csv)
    assert catalogo.ficha(1, "Toy Story (1995)", [])["titulo"] == "Toy Story (Juguetes)"
    # Una película sin datos en TMDB conserva el título original y no tiene póster.
    sin_datos = catalogo.ficha(2, "Jumanji (1995)", [])
    assert sin_datos["titulo"] == "Jumanji"
    assert sin_datos["poster"] is None
