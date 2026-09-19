"""Run the four pipeline stages of Table 8 in order.

    python -m data.pipeline [--download]

The stages remain independent modules chained by the artifacts they write;
this entry point only saves typing four commands.
"""

from __future__ import annotations

import argparse
from typing import Any

from config.settings import load_settings
from data.pipeline import clean as clean_stage
from data.pipeline import get_logger, print_summary
from data.pipeline import ingest as ingest_stage
from data.pipeline import materialize as materialize_stage
from data.pipeline import transform as transform_stage

LOGGER = get_logger("all")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m data.pipeline",
        description="Run ingestion, cleaning, transformation and materialization.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the dataset variant when it is not on disk",
    )
    parser.add_argument("--version", default=None, help="Feature store version tag to publish")
    args = parser.parse_args(argv)

    settings = load_settings()
    summaries: list[Any] = [
        ingest_stage.run(settings, download_if_missing=args.download),
        clean_stage.run(settings),
        transform_stage.run(settings),
        materialize_stage.run(settings, version=args.version),
    ]
    print_summary({"pipeline": [dict(s) for s in summaries]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
