from __future__ import annotations

from config import load_config


def test_load_mimic4_demo_config() -> None:
    cfg = load_config("configs/mimic4_demo.toml")
    assert cfg.data.table_dir.as_posix() == "data/tables/mimic-iv-demo/mortality"
    assert cfg.data.window_hours == 48
    assert cfg.data.bin_hours == 12
    assert cfg.model.hidden_dim == 128
    assert cfg.rag.require_kg_triples is True
    assert cfg.trainer.seeds == (42,)
