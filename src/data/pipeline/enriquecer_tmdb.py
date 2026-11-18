"""Etapa opcional del pipeline: títulos en español, sinopsis y pósters desde TMDB.

    $env:TMDB_TOKEN = "<token de acceso de lectura de tu cuenta de TMDB>"
    python -m data.pipeline.enriquecer_tmdb

``links.csv`` de MovieLens trae el ``tmdbId`` de cada película. Se consulta
cada una y se guarda ``datasets/peliculas_es.csv``. La aplicación solo lee ese
CSV: no depende de TMDB mientras atiende peticiones.

Muchas películas no tienen sinopsis en español de México, así que se pide la
película con todas sus traducciones y se usa la primera que exista: es-MX,
luego es-ES y al final inglés. Al volver a correrla solo consulta las que
faltan o las que se quedaron sin sinopsis, y guarda el avance cada 500.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from config.settings import load_settings

API = "https://api.themoviedb.org/3/movie/{tmdb_id}?language=es-MX&append_to_response=translations"
POSTER = "https://image.tmdb.org/t/p/w342{ruta}"
COLUMNAS = ["item_id", "titulo", "poster", "sinopsis"]
PREFERENCIA_SINOPSIS = [("es", "MX"), ("es", "ES"), ("es", None), ("en", None)]


def consultar(tmdb_id: str, token: str) -> dict[str, Any]:
    peticion = urllib.request.Request(
        API.format(tmdb_id=tmdb_id),
        headers={"Authorization": f"Bearer {token}", "accept": "application/json"},
    )
    for _ in range(3):
        try:
            with urllib.request.urlopen(peticion, timeout=15) as respuesta:
                return json.load(respuesta)
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return {}
            if error.code != 429:  # 429 = límite de peticiones: esperar y reintentar
                raise
            time.sleep(2)
    return {}


def elegir_sinopsis(datos: dict[str, Any]) -> str:
    traducciones = datos.get("translations", {}).get("translations", [])
    for idioma, pais in PREFERENCIA_SINOPSIS:
        for t in traducciones:
            coincide = t["iso_639_1"] == idioma and pais in (None, t["iso_3166_1"])
            if coincide and t["data"].get("overview"):
                return str(t["data"]["overview"])
    return str(datos.get("overview") or "")


def guardar(filas: dict[str, dict[str, str]], salida: Path) -> None:
    with salida.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=COLUMNAS)
        escritor.writeheader()
        escritor.writerows(filas.values())


def main() -> int:
    token = os.environ.get("TMDB_TOKEN")
    if not token:
        print("Falta la variable TMDB_TOKEN (token de acceso de lectura de TMDB).")
        return 1

    settings = load_settings()
    links = settings.dataset.root / settings.dataset.variant / "links.csv"
    salida: Path = settings.web.catalogo_es

    filas: dict[str, dict[str, str]] = {}
    if salida.exists():
        with salida.open(encoding="utf-8") as archivo:
            filas = {f["item_id"]: f for f in csv.DictReader(archivo)}
    with links.open(encoding="utf-8") as archivo:
        pendientes = {
            f["movieId"]: f["tmdbId"]
            for f in csv.DictReader(archivo)
            if f["tmdbId"] and not filas.get(f["movieId"], {}).get("sinopsis")
        }
    print(f"Películas por consultar: {len(pendientes)}")

    with ThreadPoolExecutor(max_workers=8) as hilos:
        resultados = hilos.map(lambda tmdb_id: consultar(tmdb_id, token), pendientes.values())
        for numero, (item_id, datos) in enumerate(zip(pendientes, resultados, strict=True), 1):
            ruta = datos.get("poster_path")
            filas[item_id] = {
                "item_id": item_id,
                "titulo": datos.get("title", ""),
                "poster": POSTER.format(ruta=ruta) if ruta else "",
                "sinopsis": elegir_sinopsis(datos),
            }
            if numero % 500 == 0:
                guardar(filas, salida)
                print(f"  {numero} listas")
    guardar(filas, salida)
    con_sinopsis = sum(1 for f in filas.values() if f["sinopsis"])
    print(f"Listo: {salida} ({con_sinopsis} de {len(filas)} con sinopsis)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
