"""Orchestrate the full data pipeline: scrape -> clean -> merge."""

import argparse
from pathlib import Path
from typing import Optional

import polars as pl

from .scrape import scrape_all
from .clean import clean_all
from .merge import merge_matches
from .utils import load_config, resolve_path


def run_pipeline(
    seasons: Optional[list[str]] = None,
    skip_scrape: bool = False,
) -> None:
    """Execute the full ETL pipeline."""
    print("=" * 60)
    print("  PL-PREDICT Data Pipeline")
    print("=" * 60)

    if skip_scrape:
        print("\nSkipping scrape, loading from raw cache ...")
        data = {}
        paths = load_config("paths")
        for key, sub in [
            ("fbref", "fbref_matches"),
            ("understat", "understat_matches"),
            ("football_data", "football_data"),
            ("elo", "elo_ratings"),
        ]:
            p = Path(resolve_path(paths["raw_dir"])) / key / f"{sub}.parquet"
            if p.exists():
                data[key] = pl.read_parquet(str(p))
                print(f"  Loaded {p.name}  ({len(data[key])} rows)")
            else:
                print(f"  No cache for {key}")
    else:
        data = scrape_all(seasons=seasons)

    print("\n=== Cleaning ===\n")
    cleaned = clean_all(data)

    print("\n=== Merging ===\n")
    matches = merge_matches(cleaned)

    print("\n=== Pipeline complete ===")
    print(f"  Final dataset: {len(matches)} matches, {len(matches.columns)} columns")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PL Predict data pipeline")
    parser.add_argument(
        "--skip-scrape", action="store_true", help="Use cached raw data"
    )
    parser.add_argument("--seasons", nargs="*", help="Seasons to process e.g. 2024-25")
    args = parser.parse_args()
    run_pipeline(seasons=args.seasons, skip_scrape=args.skip_scrape)
