"""Stage 2 of Table 8 - Cleaning.

raw -> clean dataframes. Removal of duplicates, null handling, ratings outside
the valid range, items without metadata, and users below a minimum
interaction threshold.

Every threshold is a configuration value (``config/settings.py``), so tuning
the cleaning policy never edits this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import Settings, load_settings
from data.pipeline import build_parser, get_logger, print_summary, run_stage, write_report

LOGGER = get_logger("clean")

#: Placeholder MovieLens uses for items with no genre information.
NO_GENRE = "(no genres listed)"


def _read_raw(interim_root: Path, name: str) -> pd.DataFrame:
    path = interim_root / "raw" / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"missing ingestion artifact {path}; run data.pipeline.ingest first"
        )
    return pd.read_parquet(path)


def _clean_items(items: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop duplicated and metadata-less items, and split genres into a list."""
    report: dict[str, int] = {"rows_in": len(items)}

    items = items.drop_duplicates(subset=["movieId"], keep="first")
    report["duplicates_removed"] = report["rows_in"] - len(items)

    before = len(items)
    items = items.dropna(subset=["title", "genres"])
    items = items[items["title"].str.strip() != ""]
    report["null_metadata_removed"] = before - len(items)

    before = len(items)
    items = items[items["genres"].str.strip() != ""]
    items = items[items["genres"] != NO_GENRE]
    report["no_genre_removed"] = before - len(items)

    items = items.assign(categories=items["genres"].str.split("|"))
    items = items.rename(columns={"movieId": "item_id"})
    report["rows_out"] = len(items)
    return items[["item_id", "title", "genres", "categories"]].reset_index(drop=True), report


def _clean_ratings(
    ratings: pd.DataFrame,
    valid_item_ids: set[int],
    settings: Settings,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply the five cleaning rules of Table 8, stage 2, in order."""
    report: dict[str, int] = {"rows_in": len(ratings)}
    ratings = ratings.rename(columns={"userId": "user_id", "movieId": "item_id"})

    before = len(ratings)
    ratings = ratings.dropna(subset=["user_id", "item_id", "rating", "timestamp"])
    report["null_rows_removed"] = before - len(ratings)

    before = len(ratings)
    ratings = ratings.sort_values("timestamp").drop_duplicates(
        subset=["user_id", "item_id"], keep="last"
    )
    report["duplicates_removed"] = before - len(ratings)

    before = len(ratings)
    ratings = ratings[
        ratings["rating"].between(settings.pipeline.min_rating, settings.pipeline.max_rating)
    ]
    report["out_of_range_removed"] = before - len(ratings)

    before = len(ratings)
    ratings = ratings[ratings["item_id"].isin(valid_item_ids)]
    report["orphan_items_removed"] = before - len(ratings)

    # Applied last, because the previous filters can push a user below the
    # threshold and the rule must hold on the final table.
    before = len(ratings)
    counts = ratings.groupby("user_id")["item_id"].transform("size")
    ratings = ratings[counts >= settings.pipeline.min_user_interactions]
    report["sparse_users_removed"] = before - len(ratings)

    report["rows_out"] = len(ratings)
    return ratings.reset_index(drop=True), report


def _clean_tags(
    tags: pd.DataFrame, valid_item_ids: set[int]
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep only non-empty tags attached to a surviving item."""
    report: dict[str, int] = {"rows_in": len(tags)}
    tags = tags.rename(columns={"userId": "user_id", "movieId": "item_id"})
    tags = tags.dropna(subset=["item_id", "tag"])
    tags = tags[tags["tag"].str.strip() != ""]
    tags = tags[tags["item_id"].isin(valid_item_ids)]
    report["rows_out"] = len(tags)
    return tags[["user_id", "item_id", "tag"]].reset_index(drop=True), report


def run(settings: Settings | None = None) -> Mapping[str, Any]:
    """Execute the cleaning stage and write the clean dataframes."""
    settings = settings or load_settings()
    interim_root = settings.pipeline.interim_root
    output_dir = interim_root / "clean"
    output_dir.mkdir(parents=True, exist_ok=True)

    items_raw = _read_raw(interim_root, "items")
    ratings_raw = _read_raw(interim_root, "ratings")

    items, items_report = _clean_items(items_raw)
    valid_item_ids = set(items["item_id"].tolist())
    ratings, ratings_report = _clean_ratings(ratings_raw, valid_item_ids, settings)

    # An item nobody rated after cleaning carries no signal for any strategy.
    rated_item_ids = set(ratings["item_id"].unique().tolist())
    before_items = len(items)
    items = items[items["item_id"].isin(rated_item_ids)].reset_index(drop=True)
    items_report["unrated_items_removed"] = before_items - len(items)
    items_report["rows_out"] = len(items)

    tags_path = interim_root / "raw" / "tags.parquet"
    if tags_path.exists():
        tags, tags_report = _clean_tags(_read_raw(interim_root, "tags"), set(items["item_id"]))
        tags.to_parquet(output_dir / "tags.parquet", index=False)
    else:
        tags_report = {"rows_in": 0, "rows_out": 0}
        LOGGER.warning("no tags artifact present, content metadata will use title and genres only")

    items.to_parquet(output_dir / "items.parquet", index=False)
    ratings.to_parquet(output_dir / "ratings.parquet", index=False)

    summary: dict[str, Any] = {
        "stage": "clean",
        "output_dir": str(output_dir),
        "thresholds": {
            "min_rating": settings.pipeline.min_rating,
            "max_rating": settings.pipeline.max_rating,
            "min_user_interactions": settings.pipeline.min_user_interactions,
        },
        "items": items_report,
        "ratings": ratings_report,
        "tags": tags_report,
        "distinct_users": int(ratings["user_id"].nunique()),
        "distinct_items": int(ratings["item_id"].nunique()),
    }
    write_report(output_dir / "cleaning_report.json", summary)
    LOGGER.info(
        "clean: users=%d items=%d ratings=%d",
        summary["distinct_users"],
        summary["distinct_items"],
        ratings_report["rows_out"],
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = build_parser("clean", __doc__ or "")
    args = parser.parse_args(argv)

    import os

    if args.interim_root:
        os.environ["RECO_INTERIM_ROOT"] = args.interim_root

    summary = run_stage("clean", run, settings=load_settings())
    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
