from __future__ import annotations

import argparse
from pathlib import Path

from _path import add_src_to_path

add_src_to_path()

from data_download import download_mimic_iv_demo, download_primekg


def main() -> None:
    parser = argparse.ArgumentParser(description="Download EMERGE reproduction data.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--mimic4-demo", action="store_true", help="Download MIMIC-IV demo v2.2.")
    parser.add_argument("--primekg", action="store_true", help="Download PrimeKG disease features.")
    parser.add_argument("--primekg-full", action="store_true", help="Download full PrimeKG KG triples.")
    parser.add_argument("--mimic-version", default="2.2")
    args = parser.parse_args()

    if not args.mimic4_demo and not args.primekg and not args.primekg_full:
        parser.error("Select at least one of --mimic4-demo, --primekg, or --primekg-full.")

    if args.mimic4_demo:
        download_mimic_iv_demo(args.data_dir, version=args.mimic_version)
    if args.primekg or args.primekg_full:
        download_primekg(args.data_dir, full=args.primekg_full)


if __name__ == "__main__":
    main()
