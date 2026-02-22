"""CLI entry point for the benchmark framework.

Usage:
    python -m benchmarks              # Run and print summary
    python -m benchmarks --json       # Output JSON to stdout
    python -m benchmarks --csv        # Output CSV to stdout
    python -m benchmarks --output-dir ./results  # Write JSON + CSV files
"""
import argparse
import os
import sys

from benchmarks.runner import run_benchmark
from benchmarks.export import export_json, export_csv, format_summary_table


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run enrichment quality benchmarks",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON to stdout",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Output results as CSV to stdout",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write JSON and CSV result files",
    )

    args = parser.parse_args()

    # Run the benchmark
    results = run_benchmark()

    # Output based on flags
    if args.json:
        print(export_json(results))
    elif args.csv:
        print(export_csv(results), end="")
    elif args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        json_path = os.path.join(args.output_dir, "benchmark_results.json")
        csv_path = os.path.join(args.output_dir, "benchmark_results.csv")
        export_json(results, path=json_path)
        export_csv(results, path=csv_path)
        print(f"Results written to:")
        print(f"  JSON: {json_path}")
        print(f"  CSV:  {csv_path}")
        print()
        print(format_summary_table(results))
    else:
        # Default: print summary table
        print(format_summary_table(results))


if __name__ == "__main__":
    main()
