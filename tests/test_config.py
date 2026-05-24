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
    assert cfg.rag.cache_dir.as_posix() == "data/cache/primekg"
    assert cfg.rag.use_cache is True
    assert cfg.text.embedding_backend == "tfidf_svd"
    assert cfg.llm.entity_backend == "rules"
    assert cfg.llm.summary_backend == "template"
    assert cfg.trainer.seeds == (42,)


def test_load_mimic4_demo_lm_config() -> None:
    cfg = load_config("configs/mimic4_demo_lm.toml")
    assert cfg.rag.retriever_backend == "sentence_transformer"
    assert cfg.rag.retriever_model_name == "BAAI/bge-m3"
    assert cfg.text.embedding_backend == "transformer_cls"
    assert cfg.text.note_model_name == "yikuan8/Clinical-Longformer"


def test_load_mimic4_demo_2026_config() -> None:
    cfg = load_config("configs/mimic4_demo_2026.toml")
    assert cfg.rag.retriever_model_name == "Qwen/Qwen3-Embedding-0.6B"
    assert cfg.text.note_model_name == "yikuan8/Clinical-Longformer"
    assert cfg.llm.summary_model_name == "deepseek-v4-flash"
    assert cfg.model.ehr_net == "modern_transformer"
    assert cfg.model.text_fusion == "gated"
    assert cfg.model.modality_fusion == "token_transformer"
    assert cfg.model.num_layers == 2


def test_load_mimic4_demo_modernbert_config() -> None:
    cfg = load_config("configs/mimic4_demo_modernbert.toml")
    assert cfg.text.note_model_name == "answerdotai/ModernBERT-base"
    assert cfg.text.max_length == 8192
    assert cfg.model.modality_fusion == "token_transformer"
