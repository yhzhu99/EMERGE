from __future__ import annotations

import argparse
from pathlib import Path

from config import load_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run EMERGE reproduction stages.")
    parser.add_argument(
        "--stage",
        choices=("download", "convert", "index-kg", "preprocess", "train", "all"),
        default="all",
        help="Pipeline stage to run.",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/mimic4_demo.toml"))
    parser.add_argument("--seed", type=int, default=None, help="Train one seed instead of configured seeds.")
    parser.add_argument("--skip-download", action="store_true", help="Skip downloads with --stage all.")
    parser.add_argument("--rebuild-cache", action="store_true", help="Rebuild cached PrimeKG retrieval artifacts.")
    primekg = parser.add_mutually_exclusive_group()
    primekg.add_argument(
        "--primekg-full",
        action="store_true",
        help="Download every official PrimeKG file from Harvard Dataverse.",
    )
    primekg.add_argument(
        "--primekg-lightweight",
        action="store_true",
        help="Download only README.txt and disease_features.tab for smoke tests.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    from datasets.download import download_mimic_iv_demo, download_primekg
    from datasets.mimic4 import convert_mimic4_demo
    from datasets.processed import preprocess_emerge_dataset
    from datasets.rag import KnowledgeRetriever
    from training.runner import train_from_config

    if args.stage == "download":
        download_mimic_iv_demo(cfg.data.raw_dir)
        download_primekg(
            cfg.data.kg_dir,
            full=args.primekg_full,
            lightweight=args.primekg_lightweight,
        )
        return 0

    if args.stage == "convert":
        out_dir = convert_mimic4_demo(cfg)
        print(f"Wrote OneEHR tables to {out_dir}")
        return 0

    if args.stage == "index-kg":
        KnowledgeRetriever.from_primekg(
            cfg.data.kg_dir,
            require_triples=cfg.rag.require_kg_triples,
            triples_per_node_limit=cfg.rag.max_triples_per_node,
            max_features=cfg.rag.vectorizer_max_features,
            retriever_backend=cfg.rag.retriever_backend,
            retriever_model_name=cfg.rag.retriever_model_name,
            retriever_batch_size=cfg.rag.retriever_batch_size,
            retriever_device=cfg.rag.retriever_device,
            cache_dir=cfg.rag.cache_dir,
            use_cache=True,
            rebuild_cache=args.rebuild_cache or cfg.rag.rebuild_cache,
        )
        print(f"PrimeKG retrieval cache is available under {cfg.rag.cache_dir}")
        return 0

    if args.stage == "preprocess":
        out_dir = preprocess_emerge_dataset(cfg)
        print(f"Wrote EMERGE processed artifacts to {out_dir}")
        return 0

    if args.stage == "train":
        seeds = (args.seed,) if args.seed is not None else cfg.trainer.seeds
        summary = train_from_config(cfg, seeds=seeds)
        print(summary.to_string(index=False))
        return 0

    if args.stage == "all":
        if not args.skip_download:
            download_mimic_iv_demo(cfg.data.raw_dir)
            download_primekg(
                cfg.data.kg_dir,
                full=args.primekg_full,
                lightweight=args.primekg_lightweight,
            )
        convert_mimic4_demo(cfg)
        preprocess_emerge_dataset(cfg)
        seeds = (args.seed,) if args.seed is not None else cfg.trainer.seeds
        summary = train_from_config(cfg, seeds=seeds)
        print(summary.to_string(index=False))
        return 0

    parser.error(f"Unsupported stage: {args.stage}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
