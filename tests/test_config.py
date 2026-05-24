from __future__ import annotations

from config import load_config


def test_load_mimic4_demo_config() -> None:
    cfg = load_config("configs/mimic4_demo.toml")
    assert cfg.data.table_dir.as_posix() == "data/tables/mimic-iv-demo/mortality"
    assert cfg.data.window_hours == 48
    assert cfg.data.bin_hours == 12
    assert cfg.model.hidden_dim == 128
    assert cfg.rag.require_kg_triples is True
    assert cfg.rag.retriever_backend == "tfidf"
    assert cfg.text.embedding_backend == "tfidf_svd"
    assert cfg.trainer.seeds == (42,)


def test_load_mimic4_demo_lm_config() -> None:
    cfg = load_config("configs/mimic4_demo_lm.toml")
    assert cfg.rag.retriever_backend == "sentence_transformer"
    assert cfg.rag.retriever_model_name == "BAAI/bge-m3"
    assert cfg.text.embedding_backend == "transformer_cls"
    assert cfg.text.note_model_name == "yikuan8/Clinical-Longformer"
