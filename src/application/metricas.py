"""Resumen del reporte de evaluación offline para las gráficas de la interfaz.

``benchmarks/evaluate_models.py`` escribe ``reports/evaluation_report.json``
con muchas cifras; aquí solo se toman las que se grafican.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

METRICAS = ("precision@10", "recall@10", "ndcg@10", "hit@10")
NOMBRES_ESTRATEGIA = {
    "popularity": "Popularidad",
    "als": "Filtrado colaborativo (ALS)",
    "content": "Basado en contenido",
    "ncf": "Red neuronal (NCF)",
}


def resumen_de_evaluacion(ruta: Path) -> dict[str, Any] | None:
    if not ruta.exists():
        return None
    reporte = json.loads(ruta.read_text(encoding="utf-8"))
    estrategias = []
    for clave, resultado in reporte["results"].items():
        estrategias.append(
            {
                "clave": clave,
                "nombre": NOMBRES_ESTRATEGIA.get(clave, clave),
                "metricas": {m: resultado.get(m, 0.0) for m in METRICAS},
                "cobertura": resultado.get("catalog_coverage", 0.0),
                # hit@10 por segmento: usuarios con poco, medio y mucho historial.
                "por_segmento": {
                    segmento: valores.get("hit@10", 0.0)
                    for segmento, valores in resultado.get("by_segment", {}).items()
                },
            }
        )
    return {
        "k": reporte["k"],
        "usuarios_evaluados": reporte["users_evaluated"],
        "estrategias": estrategias,
    }


def resumen_arranque_en_frio(ruta: Path) -> dict[str, Any] | None:
    """Lo escribe ``benchmarks/evaluar_arranque_en_frio.py``; se grafica tal cual."""
    if not ruta.exists():
        return None
    reporte: dict[str, Any] = json.loads(ruta.read_text(encoding="utf-8"))
    return reporte


__all__ = ["resumen_arranque_en_frio", "resumen_de_evaluacion"]
