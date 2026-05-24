from __future__ import annotations

import argparse
from pathlib import Path

from _path import add_src_to_path

add_src_to_path()

from mimic4_oneehr import MIMIC4OneEHRConverter


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MIMIC-IV demo/raw tables to OneEHR CSVs.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("../data/EMERGE/raw/mimic-iv-demo-2.2"),
        help="MIMIC-IV root containing hosp/ and icu/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../data/EMERGE/oneehr/mimic4_demo"),
        help="Output directory for dynamic.csv, static.csv, label.csv, cohort.csv, notes.csv.",
    )
    parser.add_argument("--notes-file", type=Path, default=None)
    parser.add_argument("--no-chartevents", action="store_true")
    parser.add_argument("--no-prescriptions", action="store_true")
    parser.add_argument("--max-chartevents-rows", type=int, default=None)
    args = parser.parse_args()

    converter = MIMIC4OneEHRConverter(
        args.raw_dir,
        use_chartevents=not args.no_chartevents,
        use_prescriptions=not args.no_prescriptions,
        max_chartevents_rows=args.max_chartevents_rows,
        notes_file=args.notes_file,
    )
    paths = converter.save(args.output_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
