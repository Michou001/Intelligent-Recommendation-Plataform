"""Offline feature materialization pipeline (Table 8, Section 9.3).

The pipeline is four independent stages, each one an executable module with a
single responsibility, chained by the artifacts they write rather than by
direct calls. This is the structure that enforces the separation clarified in
Section 4.2: the pipeline *computes* the feature values offline and
``FeatureStoreManager`` only *reads* them at inference time.

Run one stage:

    python -m data.pipeline.ingest
    python -m data.pipeline.clean
    python -m data.pipeline.transform
    python -m data.pipeline.materialize

Run all four in order:

    python -m data.pipeline
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

# Set before the stage modules import numpy: the ALS backend parallelises its
# own loops, and a second BLAS thread pool underneath only adds contention on
# a batch job. This package imports nothing numeric itself, so assigning it
# here is early enough for the pipeline entry points.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

LOGGER_NAME = "data.pipeline"


def get_logger(stage: str) -> logging.Logger:
    """Return a stage logger that prints to stderr with a stable format."""
    logger = logging.getLogger(f"{LOGGER_NAME}.{stage}")
    if not logging.getLogger(LOGGER_NAME).handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s")
        )
        root = logging.getLogger(LOGGER_NAME)
        root.addHandler(handler)
        root.setLevel(logging.INFO)
    return logger


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    """Persist a stage report next to the artifacts it describes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")


def build_parser(stage: str, description: str) -> argparse.ArgumentParser:
    """Common CLI surface of every stage.

    The dataset variant and every path come from ``config/settings.py``; the
    flags below only override them for a single run, so the default execution
    stays configuration-driven (Section 9.3).
    """
    parser = argparse.ArgumentParser(
        prog=f"python -m data.pipeline.{stage}", description=description
    )
    parser.add_argument(
        "--variant", default=None, help="MovieLens variant (default: from settings)"
    )
    parser.add_argument(
        "--dataset-root", default=None, help="Directory holding the dataset variants"
    )
    parser.add_argument("--interim-root", default=None, help="Directory for intermediate artifacts")
    parser.add_argument(
        "--chunk-size", type=int, default=None, help="Rows per chunk on CSV reads (ingestion only)"
    )
    return parser


def run_stage(
    stage: str, runner: Callable[..., Mapping[str, Any]], **kwargs: Any
) -> Mapping[str, Any]:
    """Execute a stage, timing it and logging a one-line summary."""
    logger = get_logger(stage)
    logger.info("stage started")
    started = time.perf_counter()
    summary = runner(**kwargs)
    elapsed = time.perf_counter() - started
    logger.info("stage finished in %.2fs | %s", elapsed, json.dumps(dict(summary), default=str))
    return summary


def print_summary(summary: Mapping[str, Any]) -> None:
    """Emit the machine-readable summary on stdout for shell composition."""
    print(json.dumps(dict(summary), indent=2, default=str))


def as_paths(values: Sequence[str | None], defaults: Sequence[Path]) -> list[Path]:
    """Resolve optional CLI overrides against their configured defaults."""
    return [
        Path(v).expanduser().resolve() if v else d for v, d in zip(values, defaults, strict=True)
    ]


__all__ = [
    "as_paths",
    "build_parser",
    "get_logger",
    "print_summary",
    "run_stage",
    "write_report",
]
