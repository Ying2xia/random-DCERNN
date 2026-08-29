#!/usr/bin/env python3
"""Command-line entry point for resumable FedCERNN experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fedcernn.experiments import ALL_METHODS, run_experiment


def build_parser() -> argparse.ArgumentParser:
    code_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run or resume a configuration-driven FedCERNN experiment."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=code_root / "configs" / "smoke.yaml",
        help="YAML configuration (defaults to the safe smoke run).",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
        help="Override the configuration's result directory.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=ALL_METHODS,
        default=None,
        help="Run only the selected methods.",
    )
    parser.add_argument(
        "--max-repetitions",
        type=int,
        default=None,
        help="Cap repetitions without editing the YAML.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print expanded one-factor scenarios without fitting.",
    )
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    records = run_experiment(
        arguments.config,
        output_directory=arguments.output_directory,
        methods=arguments.methods,
        max_repetitions=arguments.max_repetitions,
        dry_run=arguments.dry_run,
    )
    if arguments.dry_run:
        print(json.dumps(records, ensure_ascii=False, indent=2))
    else:
        successes = sum(record.get("status") == "success" for record in records)
        errors = sum(record.get("status") == "error" for record in records)
        print(
            f"Stored {successes} successful runs and {errors} errors "
            f"across {len(records)} records."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
