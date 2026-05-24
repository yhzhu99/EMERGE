from __future__ import annotations

import argparse
from pathlib import Path

from _path import add_src_to_path

add_src_to_path()

from emerge_materialize import MaterializeConfig, materialize_emerge_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build EMERGE HDF5 splits from OneEHR tables and optional PrimeKG files."
    )
    parser.add_argument("--oneehr-dir", type=Path, default=Path("../data/EMERGE/oneehr/mimic4_demo"))
    parser.add_argument("--output-dir", type=Path, default=Path("../data/EMERGE/datasets/mimic4_demo"))
    parser.add_argument("--kg-dir", type=Path, default=Path("../data/EMERGE/kg/primekg"))
    parser.add_argument("--prediction-window-hours", type=int, default=48)
    parser.add_argument("--bucket-hours", type=int, default=12)
    parser.add_argument("--max-numeric-features", type=int, default=59)
    parser.add_argument("--embedding-dim", type=int, default=768)
    parser.add_argument("--abnormal-zscore", type=float, default=2.0)
    parser.add_argument("--kg-similarity-threshold", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = MaterializeConfig(
        oneehr_dir=args.oneehr_dir,
        output_dir=args.output_dir,
        kg_dir=args.kg_dir if args.kg_dir.exists() else None,
        prediction_window_hours=args.prediction_window_hours,
        bucket_hours=args.bucket_hours,
        max_numeric_features=args.max_numeric_features,
        embedding_dim=args.embedding_dim,
        abnormal_zscore=args.abnormal_zscore,
        kg_similarity_threshold=args.kg_similarity_threshold,
        seed=args.seed,
    )
    paths = materialize_emerge_dataset(config)
    for split, path in paths.items():
        print(f"{split}: {path}")


if __name__ == "__main__":
    main()
