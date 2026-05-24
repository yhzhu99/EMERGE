from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    raw_dir: Path = Path("data/raw/mimic-iv-demo/2.2")
    table_dir: Path = Path("data/tables/mimic-iv-demo/mortality")
    processed_dir: Path = Path("data/processed/mimic-iv-demo/mortality")
    kg_dir: Path = Path("data/kg/primekg")
    notes_file: Path | None = None
    task: str = "mortality"
    window_hours: int = 48
    bin_hours: int = 12
    max_numeric_features: int = 59
    embedding_dim: int = 768


@dataclass(frozen=True)
class RAGConfig:
    abnormal_zscore: float = 2.0
    kg_similarity_threshold: float = 0.7
    max_note_entities: int = 12
    max_kg_nodes: int = 10
    max_triples_per_node: int = 5
    vectorizer_max_features: int = 20000
    require_kg_triples: bool = True
    retriever_backend: str = "tfidf"
    retriever_model_name: str = "BAAI/bge-m3"
    retriever_batch_size: int = 64
    retriever_device: str = "auto"


@dataclass(frozen=True)
class TextConfig:
    embedding_backend: str = "tfidf_svd"
    note_model_name: str = "yikuan8/Clinical-Longformer"
    summary_model_name: str = "yikuan8/Clinical-Longformer"
    batch_size: int = 8
    device: str = "auto"
    max_length: int = 4096


@dataclass(frozen=True)
class LLMConfig:
    entity_backend: str = "rules"
    summary_backend: str = "template"
    api_base_url: str = "https://api.deepseek.com"
    api_key_env: str = "DEEPSEEK_API_KEY"
    entity_model_name: str = "qwen-7b"
    summary_model_name: str = "deepseek-chat"
    max_entity_rounds: int = 1
    max_entity_tokens: int = 256
    max_summary_tokens: int = 256
    temperature: float = 0.0
    timeout_seconds: int = 60
    fallback_to_local: bool = True


@dataclass(frozen=True)
class SplitConfig:
    train_size: float = 0.7
    val_size: float = 0.1
    test_size: float = 0.2
    seed: int = 42


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int = 128
    ehr_net: str = "gru"
    text_fusion: str = "concat"
    modality_fusion: str = "ours"
    use_modality: str = "ehr_note_summary"
    num_heads: int = 4
    dropout: float = 0.25


@dataclass(frozen=True)
class TrainerConfig:
    batch_size: int = 256
    max_epochs: int = 100
    patience: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    focus_task: str = "outcome"
    device: str = "auto"
    seeds: tuple[int, ...] = (42,)


@dataclass(frozen=True)
class OutputConfig:
    run_dir: Path = Path("runs/mimic4_demo_mortality")


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    text: TextConfig = field(default_factory=TextConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def load_config(path: str | Path) -> ExperimentConfig:
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))

    data_default = DataConfig()
    rag_default = RAGConfig()
    text_default = TextConfig()
    llm_default = LLMConfig()
    split_default = SplitConfig()
    model_default = ModelConfig()
    trainer_default = TrainerConfig()
    output_default = OutputConfig()

    data_raw = raw.get("data", {})
    rag_raw = raw.get("rag", {})
    text_raw = raw.get("text", {})
    llm_raw = raw.get("llm", {})
    split_raw = raw.get("split", {})
    model_raw = raw.get("model", {})
    trainer_raw = raw.get("trainer", {})
    output_raw = raw.get("output", {})

    data = DataConfig(
        raw_dir=_as_path(data_raw.get("raw_dir"), data_default.raw_dir),
        table_dir=_as_path(data_raw.get("table_dir"), data_default.table_dir),
        processed_dir=_as_path(data_raw.get("processed_dir"), data_default.processed_dir),
        kg_dir=_as_path(data_raw.get("kg_dir"), data_default.kg_dir),
        notes_file=_as_optional_path(data_raw.get("notes_file")),
        task=str(data_raw.get("task", data_default.task)),
        window_hours=int(data_raw.get("window_hours", data_default.window_hours)),
        bin_hours=int(data_raw.get("bin_hours", data_default.bin_hours)),
        max_numeric_features=int(
            data_raw.get("max_numeric_features", data_default.max_numeric_features)
        ),
        embedding_dim=int(data_raw.get("embedding_dim", data_default.embedding_dim)),
    )
    rag = RAGConfig(
        abnormal_zscore=float(rag_raw.get("abnormal_zscore", rag_default.abnormal_zscore)),
        kg_similarity_threshold=float(
            rag_raw.get("kg_similarity_threshold", rag_default.kg_similarity_threshold)
        ),
        max_note_entities=int(rag_raw.get("max_note_entities", rag_default.max_note_entities)),
        max_kg_nodes=int(rag_raw.get("max_kg_nodes", rag_default.max_kg_nodes)),
        max_triples_per_node=int(
            rag_raw.get("max_triples_per_node", rag_default.max_triples_per_node)
        ),
        vectorizer_max_features=int(
            rag_raw.get("vectorizer_max_features", rag_default.vectorizer_max_features)
        ),
        require_kg_triples=bool(rag_raw.get("require_kg_triples", rag_default.require_kg_triples)),
        retriever_backend=str(rag_raw.get("retriever_backend", rag_default.retriever_backend)),
        retriever_model_name=str(
            rag_raw.get("retriever_model_name", rag_default.retriever_model_name)
        ),
        retriever_batch_size=int(
            rag_raw.get("retriever_batch_size", rag_default.retriever_batch_size)
        ),
        retriever_device=str(rag_raw.get("retriever_device", rag_default.retriever_device)),
    )
    text = TextConfig(
        embedding_backend=str(text_raw.get("embedding_backend", text_default.embedding_backend)),
        note_model_name=str(text_raw.get("note_model_name", text_default.note_model_name)),
        summary_model_name=str(
            text_raw.get("summary_model_name", text_default.summary_model_name)
        ),
        batch_size=int(text_raw.get("batch_size", text_default.batch_size)),
        device=str(text_raw.get("device", text_default.device)),
        max_length=int(text_raw.get("max_length", text_default.max_length)),
    )
    llm = LLMConfig(
        entity_backend=str(llm_raw.get("entity_backend", llm_default.entity_backend)),
        summary_backend=str(llm_raw.get("summary_backend", llm_default.summary_backend)),
        api_base_url=str(llm_raw.get("api_base_url", llm_default.api_base_url)),
        api_key_env=str(llm_raw.get("api_key_env", llm_default.api_key_env)),
        entity_model_name=str(llm_raw.get("entity_model_name", llm_default.entity_model_name)),
        summary_model_name=str(llm_raw.get("summary_model_name", llm_default.summary_model_name)),
        max_entity_rounds=int(llm_raw.get("max_entity_rounds", llm_default.max_entity_rounds)),
        max_entity_tokens=int(llm_raw.get("max_entity_tokens", llm_default.max_entity_tokens)),
        max_summary_tokens=int(llm_raw.get("max_summary_tokens", llm_default.max_summary_tokens)),
        temperature=float(llm_raw.get("temperature", llm_default.temperature)),
        timeout_seconds=int(llm_raw.get("timeout_seconds", llm_default.timeout_seconds)),
        fallback_to_local=bool(llm_raw.get("fallback_to_local", llm_default.fallback_to_local)),
    )
    split = SplitConfig(
        train_size=float(split_raw.get("train_size", split_default.train_size)),
        val_size=float(split_raw.get("val_size", split_default.val_size)),
        test_size=float(split_raw.get("test_size", split_default.test_size)),
        seed=int(split_raw.get("seed", split_default.seed)),
    )
    model = ModelConfig(
        hidden_dim=int(model_raw.get("hidden_dim", model_default.hidden_dim)),
        ehr_net=str(model_raw.get("ehr_net", model_default.ehr_net)),
        text_fusion=str(model_raw.get("text_fusion", model_default.text_fusion)),
        modality_fusion=str(model_raw.get("modality_fusion", model_default.modality_fusion)),
        use_modality=str(model_raw.get("use_modality", model_default.use_modality)),
        num_heads=int(model_raw.get("num_heads", model_default.num_heads)),
        dropout=float(model_raw.get("dropout", model_default.dropout)),
    )
    trainer = TrainerConfig(
        batch_size=int(trainer_raw.get("batch_size", trainer_default.batch_size)),
        max_epochs=int(trainer_raw.get("max_epochs", trainer_default.max_epochs)),
        patience=int(trainer_raw.get("patience", trainer_default.patience)),
        learning_rate=float(trainer_raw.get("learning_rate", trainer_default.learning_rate)),
        weight_decay=float(trainer_raw.get("weight_decay", trainer_default.weight_decay)),
        focus_task=str(trainer_raw.get("focus_task", trainer_default.focus_task)),
        device=str(trainer_raw.get("device", trainer_default.device)),
        seeds=_as_seeds(trainer_raw.get("seeds"), trainer_default.seeds),
    )
    output = OutputConfig(run_dir=_as_path(output_raw.get("run_dir"), output_default.run_dir))

    if abs(split.train_size + split.val_size + split.test_size - 1.0) > 1e-6:
        raise ValueError("split train_size + val_size + test_size must equal 1.0")
    return ExperimentConfig(
        data=data,
        rag=rag,
        text=text,
        llm=llm,
        split=split,
        model=model,
        trainer=trainer,
        output=output,
    )


def _as_path(value: object, default: Path) -> Path:
    if value is None:
        return default
    return Path(str(value))


def _as_optional_path(value: object) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    return Path(str(value))


def _as_seeds(value: object, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        return default
    if isinstance(value, int):
        return (value,)
    return tuple(int(seed) for seed in value)  # type: ignore[arg-type]
