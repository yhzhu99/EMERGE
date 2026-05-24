from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import ExperimentConfig
from datasets.rag import (
    KnowledgeRetriever,
    build_patient_summary,
    clean_entity,
    extract_note_entities,
    retrieval_records_to_json,
)
from datasets.schema import load_oneehr_tables
from datasets.text_embedding import TfidfSvdEncoder


@dataclass(frozen=True)
class ProcessedDataset:
    x_ehr: np.ndarray
    note_embeddings: np.ndarray
    summary_embeddings: np.ndarray
    y: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    patient_ids: np.ndarray
    metadata: dict[str, object]


def preprocess_emerge_dataset(cfg: ExperimentConfig) -> Path:
    cfg.data.processed_dir.mkdir(parents=True, exist_ok=True)
    tables = load_oneehr_tables(cfg.data.table_dir)
    dynamic = tables.dynamic.copy()
    static = tables.static.copy()
    label = tables.label.copy()
    cohort = _cohort_or_infer(tables.cohort, dynamic)
    notes = _notes_or_empty(tables.notes, cohort)

    labels = _build_label_frame(label)
    patient_ids = _eligible_patient_ids(cohort, labels)
    split = _split_patients(labels.loc[patient_ids], cfg)
    patient_index = {pid: idx for idx, pid in enumerate(patient_ids)}

    numeric_events = _prepare_numeric_events(dynamic, cohort, cfg)
    feature_names = _select_features(numeric_events, split["train"], cfg.data.max_numeric_features)
    binned = _bin_numeric_events(numeric_events, feature_names, cfg)
    stats = _fit_feature_stats(binned, static, feature_names, split["train"])
    x_ehr, abnormal_by_patient = _build_time_series(
        patient_ids=patient_ids,
        binned=binned,
        static=static,
        feature_names=feature_names,
        stats=stats,
        cfg=cfg,
    )

    dx_entities = _diagnosis_entities(dynamic)
    note_text_by_patient = notes.set_index("patient_id")["text"].fillna("").astype(str).to_dict()
    note_entities_by_patient: dict[str, list[str]] = {}
    patient_entities: dict[str, list[str]] = {}
    for pid in patient_ids:
        note_entities = extract_note_entities(
            note_text_by_patient.get(pid, ""), max_entities=cfg.rag.max_note_entities
        )
        note_entities_by_patient[pid] = note_entities
        patient_entities[pid] = _deduplicate(
            abnormal_by_patient.get(pid, []) + note_entities + dx_entities.get(pid, [])
        )

    retriever = KnowledgeRetriever.from_primekg(
        cfg.data.kg_dir,
        require_triples=cfg.rag.require_kg_triples,
        max_features=cfg.rag.vectorizer_max_features,
    )
    summaries: dict[str, str] = {}
    retrieval_json: dict[str, str] = {}
    for pid in patient_ids:
        retrieved = retriever.retrieve(
            patient_entities[pid],
            threshold=cfg.rag.kg_similarity_threshold,
            top_k=cfg.rag.max_kg_nodes,
            triples_per_node=cfg.rag.max_triples_per_node,
        )
        summaries[pid] = build_patient_summary(
            abnormal_by_patient.get(pid, []),
            note_entities_by_patient.get(pid, []),
            retrieved,
        )
        retrieval_json[pid] = retrieval_records_to_json(retrieved)

    note_embeddings, summary_embeddings = _encode_texts(
        patient_ids=patient_ids,
        split=split,
        notes=note_text_by_patient,
        summaries=summaries,
        embedding_dim=cfg.data.embedding_dim,
    )

    y = np.asarray(
        [[labels.loc[pid, "outcome"], labels.loc[pid, "readmission"]] for pid in patient_ids],
        dtype=np.float32,
    )
    train_idx = np.asarray([patient_index[pid] for pid in split["train"]], dtype=np.int64)
    val_idx = np.asarray([patient_index[pid] for pid in split["val"]], dtype=np.int64)
    test_idx = np.asarray([patient_index[pid] for pid in split["test"]], dtype=np.int64)

    metadata = {
        "table_dir": str(cfg.data.table_dir),
        "kg_dir": str(cfg.data.kg_dir),
        "task": cfg.data.task,
        "window_hours": cfg.data.window_hours,
        "bin_hours": cfg.data.bin_hours,
        "time_steps": cfg.data.window_hours // cfg.data.bin_hours,
        "input_ehr_dim": int(x_ehr.shape[2]),
        "input_note_dim": int(note_embeddings.shape[1]),
        "input_summary_dim": int(summary_embeddings.shape[1]),
        "dynamic_feature_names": feature_names,
        "feature_names": stats["total_feature_names"],
        "splits": {name: len(ids) for name, ids in split.items()},
        "config": _jsonable(asdict(cfg)),
    }
    np.savez_compressed(
        cfg.data.processed_dir / "emerge_dataset.npz",
        x_ehr=x_ehr.astype(np.float32),
        note_embeddings=note_embeddings.astype(np.float32),
        summary_embeddings=summary_embeddings.astype(np.float32),
        y=y.astype(np.float32),
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        patient_ids=np.asarray(patient_ids, dtype=str),
    )
    (cfg.data.processed_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (cfg.data.processed_dir / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    _write_rag_records(
        cfg.data.processed_dir / "rag_records.csv",
        patient_ids,
        patient_entities,
        abnormal_by_patient,
        summaries,
        retrieval_json,
    )
    return cfg.data.processed_dir


def load_processed_dataset(processed_dir: str | Path) -> ProcessedDataset:
    processed_dir = Path(processed_dir)
    arrays = np.load(processed_dir / "emerge_dataset.npz", allow_pickle=True)
    metadata = json.loads((processed_dir / "metadata.json").read_text(encoding="utf-8"))
    return ProcessedDataset(
        x_ehr=arrays["x_ehr"].astype(np.float32),
        note_embeddings=arrays["note_embeddings"].astype(np.float32),
        summary_embeddings=arrays["summary_embeddings"].astype(np.float32),
        y=arrays["y"].astype(np.float32),
        train_idx=arrays["train_idx"].astype(np.int64),
        val_idx=arrays["val_idx"].astype(np.int64),
        test_idx=arrays["test_idx"].astype(np.int64),
        patient_ids=arrays["patient_ids"].astype(str),
        metadata=metadata,
    )


def _cohort_or_infer(cohort: pd.DataFrame | None, dynamic: pd.DataFrame) -> pd.DataFrame:
    if cohort is not None:
        out = cohort.copy()
        for col in ["admittime", "dischtime", "icu_intime", "icu_outtime"]:
            if col in out.columns:
                out[col] = pd.to_datetime(out[col], errors="coerce")
        return out
    event_bounds = dynamic.groupby("patient_id")["event_time"].agg(["min", "max"]).reset_index()
    return event_bounds.rename(columns={"min": "icu_intime", "max": "icu_outtime"})


def _notes_or_empty(notes: pd.DataFrame | None, cohort: pd.DataFrame) -> pd.DataFrame:
    if notes is not None:
        out = notes[["patient_id", "text"]].copy()
        out["patient_id"] = out["patient_id"].astype(str)
        out["text"] = out["text"].fillna("").astype(str)
        return out
    return pd.DataFrame({"patient_id": cohort["patient_id"].astype(str), "text": ""})


def _build_label_frame(label: pd.DataFrame) -> pd.DataFrame:
    pivot = label.pivot_table(
        index="patient_id",
        columns="label_code",
        values="label_value",
        aggfunc="last",
    )
    if "mortality" in pivot.columns and "outcome" not in pivot.columns:
        pivot["outcome"] = pivot["mortality"]
    if "outcome" not in pivot.columns:
        raise ValueError("label.csv must contain label_code='outcome' or 'mortality'.")
    if "readmission" not in pivot.columns:
        pivot["readmission"] = 0.0
    pivot["readmission"] = pivot["readmission"].fillna(0.0)
    return pivot[["outcome", "readmission"]].dropna(subset=["outcome"]).astype(float)


def _eligible_patient_ids(cohort: pd.DataFrame, labels: pd.DataFrame) -> list[str]:
    ids = sorted(set(cohort["patient_id"].astype(str)) & set(labels.index.astype(str)))
    if not ids:
        raise ValueError("No overlapping patients between OneEHR tables and labels.")
    return ids


def _split_patients(labels: pd.DataFrame, cfg: ExperimentConfig) -> dict[str, list[str]]:
    patient_ids = np.asarray(labels.index.astype(str))
    y = labels["outcome"].astype(int).to_numpy()
    stratify = y if len(np.unique(y)) > 1 and min(np.bincount(y)) >= 2 else None
    train_val, test = train_test_split(
        patient_ids,
        test_size=cfg.split.test_size,
        random_state=cfg.split.seed,
        stratify=stratify,
    )
    y_train_val = labels.loc[train_val, "outcome"].astype(int).to_numpy()
    val_size = cfg.split.val_size / (1.0 - cfg.split.test_size)
    stratify_tv = (
        y_train_val
        if len(np.unique(y_train_val)) > 1 and min(np.bincount(y_train_val)) >= 2
        else None
    )
    train, val = train_test_split(
        train_val,
        test_size=val_size,
        random_state=cfg.split.seed,
        stratify=stratify_tv,
    )
    return {"train": sorted(train.tolist()), "val": sorted(val.tolist()), "test": sorted(test.tolist())}


def _prepare_numeric_events(
    dynamic: pd.DataFrame, cohort: pd.DataFrame, cfg: ExperimentConfig
) -> pd.DataFrame:
    events = dynamic[dynamic["code"].str.startswith(("lab:", "chart:"))].copy()
    events["value_num"] = pd.to_numeric(events["value"], errors="coerce")
    events = events.dropna(subset=["value_num"])
    anchor = cohort[["patient_id", "icu_intime"]].copy()
    anchor["icu_intime"] = pd.to_datetime(anchor["icu_intime"], errors="coerce")
    events = events.merge(anchor, on="patient_id", how="inner")
    events["hours"] = (events["event_time"] - events["icu_intime"]).dt.total_seconds() / 3600.0
    events = events[(events["hours"] >= 0) & (events["hours"] < cfg.data.window_hours)]
    return events[["patient_id", "event_time", "code", "value_num", "hours"]]


def _select_features(events: pd.DataFrame, train_ids: list[str], max_features: int) -> list[str]:
    train_events = events[events["patient_id"].isin(train_ids)]
    counts = train_events["code"].value_counts()
    if counts.empty:
        counts = events["code"].value_counts()
    return counts.head(max_features).index.astype(str).tolist()


def _bin_numeric_events(
    events: pd.DataFrame, feature_names: list[str], cfg: ExperimentConfig
) -> pd.DataFrame:
    if not feature_names:
        return pd.DataFrame(columns=["patient_id", "bin", "code", "value_num"])
    out = events[events["code"].isin(feature_names)].copy()
    out["bin"] = np.floor(out["hours"] / cfg.data.bin_hours).astype(int)
    max_bin = cfg.data.window_hours // cfg.data.bin_hours - 1
    out["bin"] = out["bin"].clip(0, max_bin)
    return (
        out.groupby(["patient_id", "bin", "code"], as_index=False)["value_num"]
        .mean()
        .sort_values(["patient_id", "bin", "code"], kind="stable")
    )


def _fit_feature_stats(
    binned: pd.DataFrame, static: pd.DataFrame, feature_names: list[str], train_ids: list[str]
) -> dict[str, object]:
    train = binned[binned["patient_id"].isin(train_ids)]
    means = train.groupby("code")["value_num"].mean().reindex(feature_names).fillna(0.0)
    stds = train.groupby("code")["value_num"].std().reindex(feature_names).fillna(1.0)
    stds = stds.mask(stds < 1e-6, 1.0)

    static_index = static.copy()
    static_index["patient_id"] = static_index["patient_id"].astype(str)
    static_index = static_index.set_index("patient_id")
    age = pd.to_numeric(static_index.get("age", pd.Series(dtype=float)), errors="coerce")
    sex = (
        static_index.get("sex", pd.Series(dtype=object))
        .astype(str)
        .str.upper()
        .map({"M": 1.0, "F": 0.0})
    )
    age_mean, age_std = _mean_std(age.reindex(train_ids))
    sex_mean, sex_std = _mean_std(sex.reindex(train_ids))
    return {
        "means": means.to_dict(),
        "stds": stds.to_dict(),
        "age": age,
        "sex": sex,
        "age_mean": age_mean,
        "age_std": age_std,
        "sex_mean": sex_mean,
        "sex_std": sex_std,
        "total_feature_names": feature_names + ["static:age", "static:sex_male"],
    }


def _build_time_series(
    patient_ids: list[str],
    binned: pd.DataFrame,
    static: pd.DataFrame,
    feature_names: list[str],
    stats: dict[str, object],
    cfg: ExperimentConfig,
) -> tuple[np.ndarray, dict[str, list[str]]]:
    time_steps = cfg.data.window_hours // cfg.data.bin_hours
    feature_index = {feature: idx for idx, feature in enumerate(feature_names)}
    total_dim = len(feature_names) + 2
    patient_index = {pid: idx for idx, pid in enumerate(patient_ids)}
    x = np.zeros((len(patient_ids), time_steps, total_dim), dtype=np.float32)
    observed_z: dict[str, list[tuple[str, float]]] = {pid: [] for pid in patient_ids}

    means: dict[str, float] = stats["means"]  # type: ignore[assignment]
    stds: dict[str, float] = stats["stds"]  # type: ignore[assignment]
    for row in binned.itertuples(index=False):
        pid = str(row.patient_id)
        if pid not in patient_index or row.code not in feature_index:
            continue
        z_value = (float(row.value_num) - means[row.code]) / stds[row.code]
        x[patient_index[pid], int(row.bin), feature_index[row.code]] = z_value
        if abs(z_value) >= cfg.rag.abnormal_zscore:
            observed_z[pid].append((_abnormal_condition(row.code, z_value), abs(z_value)))

    age: pd.Series = stats["age"]  # type: ignore[assignment]
    sex: pd.Series = stats["sex"]  # type: ignore[assignment]
    for pid in patient_ids:
        idx = patient_index[pid]
        x[idx, :, len(feature_names)] = _z_or_zero(
            age.get(pid, np.nan), stats["age_mean"], stats["age_std"]  # type: ignore[arg-type]
        )
        x[idx, :, len(feature_names) + 1] = _z_or_zero(
            sex.get(pid, np.nan), stats["sex_mean"], stats["sex_std"]  # type: ignore[arg-type]
        )

    abnormal = {
        pid: _deduplicate([name for name, _ in sorted(values, key=lambda item: item[1], reverse=True)])[
            :12
        ]
        for pid, values in observed_z.items()
    }
    return x, abnormal


def _diagnosis_entities(dynamic: pd.DataFrame) -> dict[str, list[str]]:
    dx = dynamic[dynamic["code"].str.startswith("dx:")].copy()
    entities: dict[str, list[str]] = {}
    for row in dx.itertuples(index=False):
        value = str(row.value)
        if value and value != "1":
            entities.setdefault(str(row.patient_id), []).append(value)
    return {pid: _deduplicate(vals)[:16] for pid, vals in entities.items()}


def _encode_texts(
    patient_ids: list[str],
    split: dict[str, list[str]],
    notes: dict[str, str],
    summaries: dict[str, str],
    embedding_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    note_encoder = TfidfSvdEncoder(embedding_dim=embedding_dim)
    summary_encoder = TfidfSvdEncoder(embedding_dim=embedding_dim)
    note_encoder.fit([notes.get(pid, "") for pid in split["train"]])
    summary_encoder.fit([summaries.get(pid, "") for pid in split["train"]])
    note_matrix = note_encoder.transform([notes.get(pid, "") for pid in patient_ids])
    summary_matrix = summary_encoder.transform([summaries.get(pid, "") for pid in patient_ids])
    return note_matrix, summary_matrix


def _write_rag_records(
    path: Path,
    patient_ids: list[str],
    entities: dict[str, list[str]],
    abnormal: dict[str, list[str]],
    summaries: dict[str, str],
    retrieved: dict[str, str],
) -> None:
    pd.DataFrame(
        {
            "patient_id": patient_ids,
            "entities": [json.dumps(entities.get(pid, []), ensure_ascii=False) for pid in patient_ids],
            "abnormal_features": [
                json.dumps(abnormal.get(pid, []), ensure_ascii=False) for pid in patient_ids
            ],
            "summary": [summaries.get(pid, "") for pid in patient_ids],
            "retrieved": [retrieved.get(pid, "[]") for pid in patient_ids],
        }
    ).to_csv(path, index=False)


def _mean_std(values: pd.Series) -> tuple[float, float]:
    numeric = pd.to_numeric(values, errors="coerce")
    mean = float(numeric.mean()) if numeric.notna().any() else 0.0
    std = float(numeric.std()) if numeric.notna().sum() > 1 else 1.0
    if std < 1e-6 or np.isnan(std):
        std = 1.0
    return mean, std


def _z_or_zero(value: object, mean: float, std: float) -> float:
    if pd.isna(value):
        return 0.0
    return float((float(value) - mean) / std)


def _feature_label(code: str) -> str:
    parts = code.split(":")
    if len(parts) >= 2:
        return parts[1].replace("_", " ")
    return code.replace("_", " ")


def _abnormal_condition(code: str, z_value: float) -> str:
    label = _feature_label(code)
    lowered = label.lower()
    high = z_value > 0
    mappings = [
        ("heart rate", "tachycardia", "bradycardia"),
        ("respiratory rate", "tachypnea", "bradypnea"),
        ("o2 saturation", "hyperoxia", "hypoxemia"),
        ("oxygen saturation", "hyperoxia", "hypoxemia"),
        ("blood pressure", "hypertension", "hypotension"),
        ("temperature", "fever", "hypothermia"),
        ("glucose", "hyperglycemia", "hypoglycemia"),
        ("potassium", "hyperkalemia", "hypokalemia"),
        ("sodium", "hypernatremia", "hyponatremia"),
        ("chloride", "hyperchloremia", "hypochloremia"),
        ("creatinine", "kidney dysfunction", "low creatinine"),
        ("urea nitrogen", "azotemia", "low blood urea nitrogen"),
        ("hemoglobin", "polycythemia", "anemia"),
        ("hematocrit", "polycythemia", "anemia"),
        ("white blood cells", "leukocytosis", "leukopenia"),
        ("platelet", "thrombocytosis", "thrombocytopenia"),
        ("ph", "alkalosis", "acidosis"),
        ("lactate", "lactic acidosis", "low lactate"),
    ]
    for needle, high_name, low_name in mappings:
        if needle in lowered:
            return high_name if high else low_name
    return f"{'high' if high else 'low'} {label}"


def _deduplicate(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        clean = clean_entity(value)
        key = clean.lower()
        if clean and key not in seen:
            seen.add(key)
            out.append(clean)
    return out


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value
