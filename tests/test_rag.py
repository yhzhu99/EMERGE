from __future__ import annotations

from pathlib import Path

import pytest

from datasets.rag import KnowledgeRetriever


def test_primekg_requires_official_kg_csv(tmp_path: Path) -> None:
    (tmp_path / "disease_features.tab").write_text(
        "node_index\tmondo_name\tmondo_definition\n1\tsepsis\tlife-threatening infection\n",
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="kg.csv"):
        KnowledgeRetriever.from_primekg(tmp_path, require_triples=True)


def test_primekg_loads_official_triples(tmp_path: Path) -> None:
    (tmp_path / "disease_features.tab").write_text(
        "node_index\tmondo_name\tmondo_definition\n1\tsepsis\tlife-threatening infection\n",
        encoding="utf-8",
    )
    (tmp_path / "kg.csv").write_text(
        "x_name,display_relation,y_name,x_type,y_type\n"
        "sepsis,associated with,inflammation,disease,effect/phenotype\n",
        encoding="utf-8",
    )

    retriever = KnowledgeRetriever.from_primekg(tmp_path, require_triples=True)

    assert retriever.triples_by_name["sepsis"] == [
        "(sepsis, associated with, inflammation)"
    ]
    assert "inflammation" not in retriever.triples_by_name


def test_primekg_writes_retrieval_cache(tmp_path: Path) -> None:
    kg_dir = tmp_path / "kg"
    cache_dir = tmp_path / "cache"
    kg_dir.mkdir()
    (kg_dir / "disease_features.tab").write_text(
        "node_index\tmondo_name\tmondo_definition\n1\tsepsis\tlife-threatening infection\n",
        encoding="utf-8",
    )
    (kg_dir / "kg.csv").write_text(
        "x_name,display_relation,y_name,x_type,y_type\n"
        "sepsis,associated with,inflammation,disease,effect/phenotype\n",
        encoding="utf-8",
    )

    KnowledgeRetriever.from_primekg(
        kg_dir,
        require_triples=True,
        cache_dir=cache_dir,
        use_cache=True,
    )

    cache_entries = list(cache_dir.iterdir())
    assert len(cache_entries) == 1
    assert (cache_entries[0] / "metadata.json").exists()
    assert (cache_entries[0] / "triples.json").exists()
