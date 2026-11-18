"""Evaluación del arranque en frío: ¿qué tan buenas son las primeras recomendaciones?

    python benchmarks/evaluar_arranque_en_frio.py

Usa la misma división temporal que ``evaluate_models.py`` (artifacts/evaluation):
el 80 % más antiguo de cada usuario es "lo que pasó antes" y el 20 % más reciente
es el examen. Cada usuario de MovieLens se trata como si acabara de registrarse:

1. Elige sus 3 géneros favoritos (los más frecuentes entre sus películas bien
   calificadas antes del corte).
2. El registro le muestra las 16 películas más vistas de cada género y él
   marca las que le gustaron antes del corte. Es lo mismo que ve en la app.
3. Con solo eso se le recomiendan 10 películas y se comparan con las que
   calificó bien en el examen (calificación >= 3.5).

No hay fuga de datos: el registro solo usa calificaciones anteriores al corte.

Salida: ``reports/arranque_en_frio.json``.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = REPO_ROOT / "artifacts" / "evaluation"
sys.path.insert(0, str(REPO_ROOT / "src"))
os.environ["RECO_FEATURE_STORE_ROOT"] = str(EVAL_ROOT / "feature_store")
os.environ["RECO_FEATURE_STORE_VERSION"] = "eval"

from ai.evaluation import MetricAccumulator  # noqa: E402
from ai.interfaces import IEstrategiaConPerfil, UserFeatures  # noqa: E402
from ai.model_factory import ModelFactory  # noqa: E402
from config.settings import load_settings  # noqa: E402

K = 10
GENEROS_POR_USUARIO = 3
PELICULAS_POR_GENERO = 16  # las mismas que muestra el registro de la app


def main() -> int:
    settings = load_settings()
    umbral = settings.pipeline.positive_rating_threshold
    entrenamiento = pd.read_parquet(EVAL_ROOT / "clean" / "ratings.parquet")
    examen = pd.read_parquet(EVAL_ROOT / "holdout" / "ratings.parquet")
    peliculas = pd.read_parquet(EVAL_ROOT / "clean" / "items.parquet")
    generos_de = dict(zip(peliculas["item_id"], peliculas["genres"].str.split("|"), strict=True))
    # El registro muestra las más vistas de cada género (las que más gente reconoce).
    por_genero = pd.read_parquet(settings.feature_store.version_dir / "category_popularity.parquet")
    mas_vistas = {
        str(g): grupo.sort_values("interaction_count", ascending=False)["item_id"].tolist()
        for g, grupo in por_genero.groupby("category")
    }

    popularidad = ModelFactory.create("popularity", settings, use_cache=False)
    contenido = ModelFactory.create("content", settings, use_cache=False)
    als = ModelFactory.create("als", settings, use_cache=False)
    assert isinstance(contenido, IEstrategiaConPerfil) and isinstance(als, IEstrategiaConPerfil)

    gustadas_antes = (
        entrenamiento[entrenamiento["rating"] >= umbral].groupby("user_id")["item_id"].apply(set)
    )
    relevantes = examen[examen["rating"] >= umbral].groupby("user_id")["item_id"].apply(set)

    estrategias = {
        "popularidad_global": "Popularidad global (E3)",
        "popularidad_genero": "Popularidad del género",
        "contenido": "Contenido",
        "als_fold_in": "ALS fold-in",
    }
    acumuladores = {clave: MetricAccumulator(K) for clave in estrategias}
    elegidas_por_usuario = []

    for user_id, relevantes_usuario in relevantes.items():
        antes = gustadas_antes.get(user_id, set())
        conteo = Counter(g for item in antes for g in generos_de.get(item, []))
        favoritos = [g for g, _ in conteo.most_common(GENEROS_POR_USUARIO)]
        if not favoritos:
            continue

        mostradas = [i for g in favoritos for i in mas_vistas.get(g, [])[:PELICULAS_POR_GENERO]]
        elegidas = [item for item in dict.fromkeys(mostradas) if item in antes]
        elegidas_por_usuario.append(len(elegidas))
        vistas = tuple(elegidas)

        global_ = UserFeatures(user_id=user_id, seen_items=vistas, is_cold_start=True)
        genero = UserFeatures(user_id=user_id, seen_items=vistas, preferred_category=favoritos[0])
        rankings = {
            "popularidad_global": popularidad.predict(global_, K),
            "popularidad_genero": popularidad.predict(genero, K),
        }
        if elegidas:
            perfil = contenido.perfil_desde_peliculas(elegidas)
            vector = als.perfil_desde_peliculas(elegidas)
            rankings["contenido"] = contenido.predict(
                UserFeatures(user_id=user_id, content_profile=perfil, seen_items=vistas), K
            )
            rankings["als_fold_in"] = als.predict(
                UserFeatures(user_id=user_id, latent_factors=vector, seen_items=vistas), K
            )
        else:
            # Igual que en la app: si no eligió ninguna se usa el género.
            rankings["contenido"] = rankings["als_fold_in"] = rankings["popularidad_genero"]

        for clave, ranking in rankings.items():
            acumuladores[clave].add([s.item_id for s in ranking], relevantes_usuario)

    catalogo = int(len(peliculas))
    reporte = {
        "k": K,
        "usuarios": len(elegidas_por_usuario),
        "peliculas_mostradas_en_registro": GENEROS_POR_USUARIO * PELICULAS_POR_GENERO,
        "promedio_elegidas": round(sum(elegidas_por_usuario) / len(elegidas_por_usuario), 2),
        "usuarios_sin_elegidas": sum(n == 0 for n in elegidas_por_usuario),
        "estrategias": [
            {"clave": clave, "nombre": nombre, **acumuladores[clave].summary(catalogo)}
            for clave, nombre in estrategias.items()
        ],
    }
    salida = REPO_ROOT / "reports" / "arranque_en_frio.json"
    salida.write_text(json.dumps(reporte, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(reporte, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
