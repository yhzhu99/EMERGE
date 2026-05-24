from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from kg_rag import (
    KnowledgeRetriever,
    build_patient_summary,
    clean_entity,
    extract_note_entities,
    retrieval_records_to_json,
)
from oneehr_schema import load_oneehr_tables
from text_embedding import TfidfSvdEncoder


@dataclass(frozen=True)
class MaterializeConfig:
    oneehr_dir: Path
    output_dir: Path
    kg_dir: Path | None = None
    prediction_window_hours: int = 48
    bucket_hours: int = 12
    max_numeric_features: int = 59
    embedding_dim: int = 768
    val_fraction: float = 0.1
    test_fraction: float = 0.2
    abnormal_zscore: float = 2.0
    kg_similarity_threshold: float = 0.7
    max_note_entities: int = 12
    max_kg_nodes: int = 10
    max_triples_per_node: int = 5
    seed: int = 42


@dataclass
class PatientRecord:
    patient_id: str
    x: np.ndarray
    note_embedding: np.ndarray
    summary_embedding: np.ndarray
    y: np.ndarray
    note_text: str
    summary_text: str
    entities: list[str]
    abnormal_features: list[str]
    retrieval_json: str


def materialize_emerge_dataset(config: MaterializeConfig) -> dict[str, Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    tables = load_oneehr_tables(config.oneehr_dir)
    dynamic = tables.dynamic.copy()
    static = tables.static.copy()
    label = tables.label.copy()
    cohort = _cohort_or_infer(tables.cohort, dynamic)
    notes = _notes_or_empty(tables.notes, cohort)

    labels = _build_label_frame(label)
    patient_ids = _eligible_patient_ids(cohort, labels)
    split = _split_patients(labels.loc[patient_ids], config)

    numeric_events = _prepare_numeric_events(dynamic, cohort, config)
    feature_names = _select_features(numeric_events, split["train"], config.max_numeric_features)
    binned = _bin_numeric_events(numeric_events, feature_names, config)
    stats = _fit_feature_stats(binned, static, feature_names, split["train"])
    x_by_patient, abnormal_by_patient = _build_time_series(
        patient_ids=patient_ids,
        binned=binned,
        static=static,
        feature_names=feature_names,
        stats=stats,
        config=config,
    )

    dx_entities = _diagnosis_entities(dynamic)
    note_text_by_patient = notes.set_index("patient_id")["text"].fillna("").astype(str).to_dict()
    patient_entities: dict[str, list[str]] = {}
    for pid in patient_ids:
        note_entities = extract_note_entities(
            note_text_by_patient.get(pid, ""), max_entities=config.max_note_entities
        )
        entities = abnormal_by_patient.get(pid, []) + note_entities + dx_entities.get(pid, [])
        patient_entities[pid] = _deduplicate(entities)

    retriever = KnowledgeRetriever.from_primekg(config.kg_dir)
    summaries: dict[str, str] = {}
    retrieval_json: dict[str, str] = {}
    for pid in patient_ids:
        retrieved = retriever.retrieve(
            patient_entities[pid],
            threshold=config.kg_similarity_threshold,
            top_k=config.max_kg_nodes,
            triples_per_node=config.max_triples_per_node,
        )
        summaries[pid] = build_patient_summary(
            abnormal_by_patient.get(pid, []),
            extract_note_entities(note_text_by_patient.get(pid, ""), max_entities=config.max_note_entities),
            retrieved,
        )
        retrieval_json[pid] = retrieval_records_to_json(retrieved)

    note_embeddings, summary_embeddings = _encode_texts(
        patient_ids=patient_ids,
        split=split,
        notes=note_text_by_patient,
        summaries=summaries,
        embedding_dim=config.embedding_dim,
    )

    records = []
    labels_for_patients = labels.loc[patient_ids]
    for pid in patient_ids:
        y = np.array(
            [[labels_for_patients.loc[pid, "outcome"], 0.0, labels_for_patients.loc[pid, "readmission"]]],
            dtype=np.float32,
        )
        records.append(
            PatientRecord(
                patient_id=pid,
                x=x_by_patient[pid],
                note_embedding=note_embeddings[pid],
                summary_embedding=summary_embeddings[pid],
                y=y,
                note_text=note_text_by_patient.get(pid, ""),
                summary_text=summaries[pid],
                entities=patient_entities[pid],
                abnormal_features=abnormal_by_patient.get(pid, []),
                retrieval_json=retrieval_json[pid],
            )
        )

    record_by_pid = {record.patient_id: record for record in records}
    paths = {}
    for split_name, ids in split.items():
        path = config.output_dir / f"data_{split_name}.h5"
        _write_hdf5([record_by_pid[pid] for pid in ids], path)
        paths[split_name] = path

    _write_rag_records(records, config.output_dir / "rag_records.csv")
    _write_metadata(
        config=config,
        output_dir=config.output_dir,
        split=split,
        feature_names=feature_names,
        total_feature_names=stats["total_feature_names"],
    )
    return paths


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
    cohort_ids = set(cohort["patient_id"].astype(str))
    label_ids = set(labels.index.astype(str))
    ids = sorted(cohort_ids & label_ids)
    if not ids:
        raise ValueError("No overlapping patients between cohort.csv and label.csv.")
    return ids


def _split_patients(labels: pd.DataFrame, config: MaterializeConfig) -> dict[str, list[str]]:
    patient_ids = np.array(labels.index.astype(str))
    y = labels["outcome"].astype(int).to_numpy()
    stratify = y if len(np.unique(y)) > 1 and min(np.bincount(y)) >= 2 else None
    train_val, test = train_test_split(
        patient_ids,
        test_size=config.test_fraction,
        random_state=config.seed,
        stratify=stratify,
    )
    y_train_val = labels.loc[train_val, "outcome"].astype(int).to_numpy()
    val_size = config.val_fraction / (1.0 - config.test_fraction)
    stratify_tv = (
        y_train_val
        if len(np.unique(y_train_val)) > 1 and min(np.bincount(y_train_val)) >= 2
        else None
    )
    train, val = train_test_split(
        train_val,
        test_size=val_size,
        random_state=config.seed,
        stratify=stratify_tv,
    )
    return {
        "train": sorted(train.tolist()),
        "val": sorted(val.tolist()),
        "test": sorted(test.tolist()),
    }


def _prepare_numeric_events(
    dynamic: pd.DataFrame, cohort: pd.DataFrame, config: MaterializeConfig
) -> pd.DataFrame:
    events = dynamic[dynamic["code"].str.startswith(("lab:", "chart:"))].copy()
    events["value_num"] = pd.to_numeric(events["value"], errors="coerce")
    events = events.dropna(subset=["value_num"])
    anchor = cohort[["patient_id", "icu_intime"]].copy()
    anchor["icu_intime"] = pd.to_datetime(anchor["icu_intime"], errors="coerce")
    events = events.merge(anchor, on="patient_id", how="inner")
    events["hours"] = (events["event_time"] - events["icu_intime"]).dt.total_seconds() / 3600.0
    events = events[(events["hours"] >= 0) & (events["hours"] < config.prediction_window_hours)]
    return events[["patient_id", "event_time", "code", "value_num", "hours"]]


def _select_features(events: pd.DataFrame, train_ids: list[str], max_features: int) -> list[str]:
    train_events = events[events["patient_id"].isin(train_ids)]
    counts = train_events["code"].value_counts()
    if counts.empty:
        counts = events["code"].value_counts()
    return counts.head(max_features).index.astype(str).tolist()


def _bin_numeric_events(
    events: pd.DataFrame, feature_names: list[str], config: MaterializeConfig
) -> pd.DataFrame:
    if not feature_names:
        return pd.DataFrame(columns=["patient_id", "bin", "code", "value_num"])
    out = events[events["code"].isin(feature_names)].copy()
    out["bin"] = np.floor(out["hours"] / config.bucket_hours).astype(int)
    out["bin"] = out["bin"].clip(0, config.prediction_window_hours // config.bucket_hours - 1)
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
    sex = static_index.get("sex", pd.Series(dtype=object)).astype(str).str.upper().map({"M": 1.0, "F": 0.0})
    age_train = age.reindex(train_ids)
    sex_train = sex.reindex(train_ids)
    age_mean, age_std = _mean_std(age_train)
    sex_mean, sex_std = _mean_std(sex_train)
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
    config: MaterializeConfig,
) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    time_steps = config.prediction_window_hours // config.bucket_hours
    feature_index = {feature: idx for idx, feature in enumerate(feature_names)}
    total_dim = len(feature_names) + 2
    x_by_patient = {
        pid: np.zeros((time_steps, total_dim), dtype=np.float32)
        for pid in patient_ids
    }
    observed_z: dict[str, list[tuple[str, float]]] = {pid: [] for pid in patient_ids}

    means: dict[str, float] = stats["means"]  # type: ignore[assignment]
    stds: dict[str, float] = stats["stds"]  # type: ignore[assignment]
    for row in binned.itertuples(index=False):
        pid = str(row.patient_id)
        if pid not in x_by_patient or row.code not in feature_index:
            continue
        idx = feature_index[row.code]
        value = float(row.value_num)
        z_value = (value - means[row.code]) / stds[row.code]
        x_by_patient[pid][int(row.bin), idx] = z_value
        if abs(z_value) >= config.abnormal_zscore:
            observed_z[pid].append((_abnormal_condition(row.code, z_value), abs(z_value)))

    age: pd.Series = stats["age"]  # type: ignore[assignment]
    sex: pd.Series = stats["sex"]  # type: ignore[assignment]
    for pid in patient_ids:
        age_value = age.get(pid, np.nan)
        sex_value = sex.get(pid, np.nan)
        age_z = _z_or_zero(age_value, stats["age_mean"], stats["age_std"])  # type: ignore[arg-type]
        sex_z = _z_or_zero(sex_value, stats["sex_mean"], stats["sex_std"])  # type: ignore[arg-type]
        x_by_patient[pid][:, len(feature_names)] = age_z
        x_by_patient[pid][:, len(feature_names) + 1] = sex_z

    abnormal = {
        pid: _deduplicate([name for name, _ in sorted(values, key=lambda item: item[1], reverse=True)])[
            :12
        ]
        for pid, values in observed_z.items()
    }
    return x_by_patient, abnormal


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
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    note_encoder = TfidfSvdEncoder(embedding_dim=embedding_dim)
    summary_encoder = TfidfSvdEncoder(embedding_dim=embedding_dim)
    train_notes = [notes.get(pid, "") for pid in split["train"]]
    train_summaries = [summaries.get(pid, "") for pid in split["train"]]
    note_encoder.fit(train_notes)
    summary_encoder.fit(train_summaries)
    note_matrix = note_encoder.transform([notes.get(pid, "") for pid in patient_ids])
    summary_matrix = summary_encoder.transform([summaries.get(pid, "") for pid in patient_ids])
    return (
        {pid: note_matrix[idx] for idx, pid in enumerate(patient_ids)},
        {pid: summary_matrix[idx] for idx, pid in enumerate(patient_ids)},
    )


def _write_hdf5(records: list[PatientRecord], path: Path) -> None:
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as handle:
        for idx, record in enumerate(records):
            group = handle.create_group(str(idx))
            group.create_dataset("PatientID", data=record.patient_id, dtype=string_dtype)
            group.create_dataset("X", data=record.x.astype(np.float32))
            group.create_dataset("Note", data=record.note_embedding.astype(np.float32))
            group.create_dataset("Summary", data=record.summary_embedding.astype(np.float32))
            group.create_dataset("Y", data=record.y.astype(np.float32))
            group.create_dataset("Text", data=record.note_text, dtype=string_dtype)
            group.create_dataset("RAGSummary", data=record.summary_text, dtype=string_dtype)
            group.create_dataset("Entities", data=json.dumps(record.entities), dtype=string_dtype)


def _write_rag_records(records: list[PatientRecord], path: Path) -> None:
    df = pd.DataFrame(
        {
            "patient_id": [record.patient_id for record in records],
            "entities": [json.dumps(record.entities, ensure_ascii=False) for record in records],
            "abnormal_features": [
                json.dumps(record.abnormal_features, ensure_ascii=False) for record in records
            ],
            "summary": [record.summary_text for record in records],
            "retrieved": [record.retrieval_json for record in records],
        }
    )
    df.to_csv(path, index=False)


def _write_metadata(
    config: MaterializeConfig,
    output_dir: Path,
    split: dict[str, list[str]],
    feature_names: list[str],
    total_feature_names: list[str],
) -> None:
    metadata = {
        "oneehr_dir": str(config.oneehr_dir),
        "kg_dir": None if config.kg_dir is None else str(config.kg_dir),
        "prediction_window_hours": config.prediction_window_hours,
        "bucket_hours": config.bucket_hours,
        "time_steps": config.prediction_window_hours // config.bucket_hours,
        "input_ehr_dim": len(total_feature_names),
        "input_note_dim": config.embedding_dim,
        "input_summary_dim": config.embedding_dim,
        "dynamic_feature_names": feature_names,
        "feature_names": total_feature_names,
        "splits": {name: len(ids) for name, ids in split.items()},
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


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
    direction = "high" if high else "low"
    return f"{direction} {label}"


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
