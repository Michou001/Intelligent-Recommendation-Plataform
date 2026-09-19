"""Stage 1 of Table 8 - Ingestion.

CSV files -> raw dataframes. Chunked reading of ratings, items, and metadata;
schema check on load; rejection report for unreadable rows.

The dataset path and variant are read from ``config/settings.py`` and the
reading is chunked, so moving from ``ml-latest-small`` to ``ml-25m`` is a
configuration change and no modification of this code (Section 9.3).
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import Settings, load_settings
from data.pipeline import build_parser, get_logger, print_summary, run_stage, write_report

LOGGER = get_logger("ingest")


@dataclass(frozen=True)
class SourceSchema:
    """Declared schema of one source file, checked on load."""

    name: str
    filename: str
    required_columns: tuple[str, ...]
    numeric_columns: tuple[str, ...]
    required: bool = True


#: MovieLens layout. Every variant published by GroupLens uses these names,
#: which is why switching variant needs no code change.
SOURCES: tuple[SourceSchema, ...] = (
    SourceSchema(
        name="ratings",
        filename="ratings.csv",
        required_columns=("userId", "movieId", "rating", "timestamp"),
        numeric_columns=("userId", "movieId", "rating", "timestamp"),
    ),
    SourceSchema(
        name="items",
        filename="movies.csv",
        required_columns=("movieId", "title", "genres"),
        numeric_columns=("movieId",),
    ),
    SourceSchema(
        name="tags",
        filename="tags.csv",
        required_columns=("userId", "movieId", "tag", "timestamp"),
        numeric_columns=("userId", "movieId", "timestamp"),
        required=False,
    ),
)


class DatasetNotFoundError(FileNotFoundError):
    """The configured dataset variant is not present on disk."""


def download_variant(settings: Settings) -> Path:
    """Fetch and unpack the configured variant into the dataset root."""
    url = settings.dataset.download_url
    root = settings.dataset.root
    root.mkdir(parents=True, exist_ok=True)
    LOGGER.info("downloading %s", url)
    payload = urllib.request.urlopen(url, timeout=300).read()  # noqa: S310 - configured URL
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(root)
    LOGGER.info("extracted into %s", root)
    return settings.dataset.variant_dir


def _iter_chunks(path: Path, chunk_size: int) -> Iterator[pd.DataFrame]:
    """Read a CSV in chunks, as strings, so no row can abort the parse."""
    yield from pd.read_csv(
        path,
        chunksize=chunk_size,
        dtype=str,
        keep_default_na=False,
        na_values=[""],
        encoding="utf-8",
    )


def _coerce(frame: pd.DataFrame, schema: SourceSchema) -> tuple[pd.DataFrame, int]:
    """Coerce the declared numeric columns, returning the rejected row count.

    A row whose numeric column cannot be parsed is unreadable for the purpose
    of the pipeline; it is dropped here and counted in the rejection report
    rather than silently propagated into the feature tables.
    """
    before = len(frame)
    for column in schema.numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=list(schema.numeric_columns))
    for column in schema.numeric_columns:
        if column in {"rating"}:
            frame[column] = frame[column].astype("float64")
        else:
            frame[column] = frame[column].astype("int64")
    return frame, before - len(frame)


def _read_source(
    path: Path, schema: SourceSchema, chunk_size: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read one source file in chunks and check its declared schema."""
    frames: list[pd.DataFrame] = []
    rows_read = 0
    rows_rejected = 0
    chunks = 0

    for chunk in _iter_chunks(path, chunk_size):
        chunks += 1
        rows_read += len(chunk)
        missing = set(schema.required_columns) - set(chunk.columns)
        if missing:
            raise ValueError(f"{schema.filename}: missing required columns {sorted(missing)}")
        chunk = chunk[list(schema.required_columns)]
        cleaned, rejected = _coerce(chunk, schema)
        rows_rejected += rejected
        frames.append(cleaned)

    frame = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=list(schema.required_columns))
    )
    report = {
        "file": str(path),
        "chunks": chunks,
        "rows_read": rows_read,
        "rows_rejected": rows_rejected,
        "rows_kept": len(frame),
    }
    LOGGER.info(
        "%-7s read=%d rejected=%d kept=%d chunks=%d",
        schema.name,
        rows_read,
        rows_rejected,
        len(frame),
        chunks,
    )
    return frame, report


def run(
    settings: Settings | None = None,
    *,
    download_if_missing: bool = False,
) -> Mapping[str, Any]:
    """Execute the ingestion stage and write the raw dataframes."""
    settings = settings or load_settings()
    variant_dir = settings.dataset.variant_dir

    if not variant_dir.exists():
        if download_if_missing:
            variant_dir = download_variant(settings)
        else:
            raise DatasetNotFoundError(
                f"dataset variant {settings.dataset.variant!r} not found "
                f"in {settings.dataset.root}; run with --download to fetch it"
            )

    output_dir = settings.pipeline.interim_root / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    reports: dict[str, Any] = {}
    written: dict[str, str] = {}

    for schema in SOURCES:
        path = variant_dir / schema.filename
        if not path.exists():
            if schema.required:
                raise DatasetNotFoundError(f"required source file missing: {path}")
            LOGGER.warning("optional source %s not present, skipped", schema.filename)
            reports[schema.name] = {"file": str(path), "skipped": True}
            continue
        frame, report = _read_source(path, schema, settings.dataset.chunk_size)
        target = output_dir / f"{schema.name}.parquet"
        frame.to_parquet(target, index=False)
        reports[schema.name] = report
        written[schema.name] = str(target)

    summary: dict[str, Any] = {
        "stage": "ingest",
        "variant": settings.dataset.variant,
        "chunk_size": settings.dataset.chunk_size,
        "output_dir": str(output_dir),
        "artifacts": written,
        "sources": reports,
    }
    write_report(output_dir / "ingestion_report.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = build_parser("ingest", __doc__ or "")
    parser.add_argument(
        "--download", action="store_true", help="Download the variant when it is not on disk"
    )
    args = parser.parse_args(argv)

    import os

    if args.variant:
        os.environ["RECO_DATASET_VARIANT"] = args.variant
    if args.dataset_root:
        os.environ["RECO_DATASET_ROOT"] = args.dataset_root
    if args.interim_root:
        os.environ["RECO_INTERIM_ROOT"] = args.interim_root
    if args.chunk_size:
        os.environ["RECO_INGEST_CHUNK_SIZE"] = str(args.chunk_size)

    summary = run_stage("ingest", run, settings=load_settings(), download_if_missing=args.download)
    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
