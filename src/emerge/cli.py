from __future__ import annotations

import argparse
from pathlib import Path

from emerge.data.download import download_mimic_iv_demo, download_primekg


def _add_common_download_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Data root.")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="emerge",
        description="EMERGE reproduction CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    mimic_parser = subparsers.add_parser(
        "download-mimic4-demo",
        help="Download the open MIMIC-IV demo v2.2 files from PhysioNet.",
    )
    _add_common_download_args(mimic_parser)
    mimic_parser.add_argument("--version", default="2.2")

    primekg_parser = subparsers.add_parser(
        "download-primekg",
        help="Download PrimeKG files. The lightweight mode only fetches disease features.",
    )
    _add_common_download_args(primekg_parser)
    primekg_parser.add_argument(
        "--full",
        action="store_true",
        help="Also download full kg.csv for graph triples. This is large.",
    )

    args = parser.parse_args()
    if args.command == "download-mimic4-demo":
        download_mimic_iv_demo(args.data_dir, version=args.version)
    elif args.command == "download-primekg":
        download_primekg(args.data_dir, full=args.full)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
