from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from datasets.schema import OneEHRDataset

if TYPE_CHECKING:
    from config import ExperimentConfig


def convert_mimic4_demo(cfg: "ExperimentConfig") -> Path:
    converter = MIMIC4OneEHRConverter(
        cfg.data.raw_dir,
        notes_file=cfg.data.notes_file,
    )
    converter.save(cfg.data.table_dir)
    return cfg.data.table_dir


class MIMIC4OneEHRConverter:
    """Convert MIMIC-IV 2.x tables into OneEHR-compatible CSVs.

    The exported core contract is exactly three tables:
    `dynamic.csv`, `static.csv`, and `label.csv`. `cohort.csv` and
    `notes.csv` are additional EMERGE reproduction artifacts.
    """

    def __init__(
        self,
        raw_dir: str | Path,
        *,
        use_chartevents: bool = True,
        use_labevents: bool = True,
        use_prescriptions: bool = True,
        use_diagnoses: bool = True,
        use_procedures: bool = True,
        max_chartevents_rows: int | None = None,
        notes_file: str | Path | None = None,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.hosp_dir = self.raw_dir / "hosp"
        self.icu_dir = self.raw_dir / "icu"
        if not self.hosp_dir.is_dir() or not self.icu_dir.is_dir():
            raise FileNotFoundError(
                f"Expected MIMIC-IV hosp/ and icu/ directories under {self.raw_dir}"
            )
        self.use_chartevents = use_chartevents
        self.use_labevents = use_labevents
        self.use_prescriptions = use_prescriptions
        self.use_diagnoses = use_diagnoses
        self.use_procedures = use_procedures
        self.max_chartevents_rows = max_chartevents_rows
        self.notes_file = Path(notes_file) if notes_file else None

    def convert(self) -> OneEHRDataset:
        admissions = self._read_hosp("admissions.csv")
        patients = self._read_hosp("patients.csv")
        icustays = self._read_icu("icustays.csv")

        admissions = _parse_datetime_columns(
            admissions, ["admittime", "dischtime", "deathtime", "edregtime", "edouttime"]
        )
        patients = _parse_datetime_columns(patients, ["dod"])
        icustays = _parse_datetime_columns(icustays, ["intime", "outtime"])

        cohort = self._build_cohort(admissions, patients, icustays)
        dynamic_parts: list[pd.DataFrame] = []

        if self.use_labevents:
            dynamic_parts.append(self._convert_labevents(cohort))
        if self.use_chartevents:
            dynamic_parts.append(self._convert_chartevents(cohort))
        if self.use_diagnoses:
            dynamic_parts.append(self._convert_diagnoses(cohort))
        if self.use_procedures:
            dynamic_parts.append(self._convert_procedures(cohort))
        if self.use_prescriptions:
            dynamic_parts.append(self._convert_prescriptions(cohort))

        dynamic_parts = [part for part in dynamic_parts if not part.empty]
        if not dynamic_parts:
            raise RuntimeError("No dynamic event tables were converted.")
        dynamic = (
            pd.concat(dynamic_parts, ignore_index=True)
            .dropna(subset=["patient_id", "event_time", "code"])
            .sort_values(["patient_id", "event_time", "code"], kind="stable")
            .reset_index(drop=True)
        )
        dynamic["value"] = dynamic["value"].fillna("")

        static = self._build_static(cohort)
        label = self._build_labels(cohort, admissions)
        notes = self._build_notes(cohort)
        return OneEHRDataset(dynamic=dynamic, static=static, label=label, cohort=cohort, notes=notes)

    def save(self, output_dir: str | Path) -> dict[str, Path]:
        paths = self.convert().save(Path(output_dir))
        return {
            "dynamic": paths.dynamic,
            "static": paths.static,
            "label": paths.label,
            "cohort": paths.cohort,
            "notes": paths.notes,
        }

    def _read_hosp(self, name: str, **kwargs) -> pd.DataFrame:
        return _read_csv(self.hosp_dir, name, **kwargs)

    def _read_icu(self, name: str, **kwargs) -> pd.DataFrame:
        return _read_csv(self.icu_dir, name, **kwargs)

    def _build_cohort(
        self, admissions: pd.DataFrame, patients: pd.DataFrame, icustays: pd.DataFrame
    ) -> pd.DataFrame:
        first_icu = (
            icustays.sort_values(["hadm_id", "intime"], kind="stable")
            .drop_duplicates("hadm_id", keep="first")
            .copy()
        )
        cohort = admissions.merge(patients, on="subject_id", how="left").merge(
            first_icu[
                [
                    "subject_id",
                    "hadm_id",
                    "stay_id",
                    "first_careunit",
                    "last_careunit",
                    "intime",
                    "outtime",
                    "los",
                ]
            ],
            on=["subject_id", "hadm_id"],
            how="inner",
        )
        cohort["patient_id"] = cohort["hadm_id"].astype("Int64").astype(str)
        cohort["icu_intime"] = cohort["intime"]
        cohort["icu_outtime"] = cohort["outtime"]
        cohort["age"] = cohort.get("anchor_age", np.nan)
        cohort = cohort.sort_values(["subject_id", "admittime"], kind="stable").reset_index(drop=True)
        return cohort

    def _build_static(self, cohort: pd.DataFrame) -> pd.DataFrame:
        static = pd.DataFrame(
            {
                "patient_id": cohort["patient_id"],
                "age": cohort["age"],
                "sex": cohort.get("gender", ""),
                "race": cohort.get("race", ""),
                "insurance": cohort.get("insurance", ""),
                "language": cohort.get("language", ""),
                "marital_status": cohort.get("marital_status", ""),
                "admission_type": cohort.get("admission_type", ""),
                "first_careunit": cohort.get("first_careunit", ""),
            }
        )
        return static.drop_duplicates("patient_id", keep="first")

    def _build_labels(self, cohort: pd.DataFrame, admissions: pd.DataFrame) -> pd.DataFrame:
        mortality = pd.DataFrame(
            {
                "patient_id": cohort["patient_id"],
                "label_time": cohort["dischtime"],
                "label_code": "outcome",
                "label_value": cohort["deathtime"].notna().astype(int),
            }
        )

        admissions_sorted = admissions.sort_values(["subject_id", "admittime"], kind="stable").copy()
        admissions_sorted["next_admit"] = admissions_sorted.groupby("subject_id")["admittime"].shift(-1)
        admissions_sorted["days_to_readmit"] = (
            admissions_sorted["next_admit"] - admissions_sorted["dischtime"]
        ).dt.total_seconds() / 86400.0
        admissions_sorted["readmission"] = (
            admissions_sorted["days_to_readmit"].notna()
            & (admissions_sorted["days_to_readmit"] >= 0)
            & (admissions_sorted["days_to_readmit"] <= 30)
        ).astype(int)
        admissions_sorted.loc[admissions_sorted["deathtime"].notna(), "readmission"] = np.nan
        readmission_map = admissions_sorted.set_index("hadm_id")["readmission"]

        readmission = pd.DataFrame(
            {
                "patient_id": cohort["patient_id"],
                "label_time": cohort["dischtime"],
                "label_code": "readmission",
                "label_value": cohort["hadm_id"].map(readmission_map),
            }
        ).dropna(subset=["label_value"])
        readmission["label_value"] = readmission["label_value"].astype(int)

        label = (
            pd.concat([mortality, readmission], ignore_index=True)
            .dropna(subset=["patient_id", "label_time", "label_value"])
            .sort_values(["patient_id", "label_code"], kind="stable")
            .reset_index(drop=True)
        )
        return label

    def _convert_labevents(self, cohort: pd.DataFrame) -> pd.DataFrame:
        try:
            labs = self._read_hosp("labevents.csv")
            items = self._read_hosp("d_labitems.csv")
        except FileNotFoundError as exc:
            warnings.warn(str(exc), stacklevel=2)
            return _empty_dynamic()

        labs = _parse_datetime_columns(labs, ["charttime"])
        labs = labs.dropna(subset=["hadm_id", "charttime"])
        labs = labs.merge(items[["itemid", "label", "fluid", "category"]], on="itemid", how="left")
        labs = labs.merge(cohort[["hadm_id", "patient_id"]], on="hadm_id", how="inner")
        code = (
            "lab:"
            + labs["label"].fillna(labs["itemid"].astype(str)).map(_clean_code_text)
            + ":"
            + labs["itemid"].astype(str)
        )
        value = labs["valuenum"].where(labs["valuenum"].notna(), labs.get("value", ""))
        return pd.DataFrame(
            {
                "patient_id": labs["patient_id"],
                "event_time": labs["charttime"],
                "code": code,
                "value": value,
            }
        )

    def _convert_chartevents(self, cohort: pd.DataFrame) -> pd.DataFrame:
        try:
            kwargs = {}
            if self.max_chartevents_rows is not None:
                kwargs["nrows"] = self.max_chartevents_rows
            charts = self._read_icu("chartevents.csv", **kwargs)
            items = self._read_icu("d_items.csv")
        except FileNotFoundError as exc:
            warnings.warn(str(exc), stacklevel=2)
            return _empty_dynamic()

        charts = _parse_datetime_columns(charts, ["charttime"])
        charts = charts.dropna(subset=["hadm_id", "charttime"])
        charts = charts.merge(items[["itemid", "label", "category"]], on="itemid", how="left")
        charts = charts.merge(cohort[["hadm_id", "patient_id"]], on="hadm_id", how="inner")
        code = (
            "chart:"
            + charts["label"].fillna(charts["itemid"].astype(str)).map(_clean_code_text)
            + ":"
            + charts["itemid"].astype(str)
        )
        value = charts["valuenum"].where(charts["valuenum"].notna(), charts.get("value", ""))
        return pd.DataFrame(
            {
                "patient_id": charts["patient_id"],
                "event_time": charts["charttime"],
                "code": code,
                "value": value,
            }
        )

    def _convert_diagnoses(self, cohort: pd.DataFrame) -> pd.DataFrame:
        try:
            dx = self._read_hosp("diagnoses_icd.csv")
            dx_items = self._read_hosp("d_icd_diagnoses.csv")
        except FileNotFoundError as exc:
            warnings.warn(str(exc), stacklevel=2)
            return _empty_dynamic()

        dx = dx.dropna(subset=["hadm_id", "icd_code"])
        dx = dx.merge(dx_items, on=["icd_code", "icd_version"], how="left")
        dx = dx.merge(cohort[["hadm_id", "patient_id", "admittime"]], on="hadm_id", how="inner")
        code = "dx:icd" + dx["icd_version"].astype(str) + ":" + dx["icd_code"].astype(str)
        return pd.DataFrame(
            {
                "patient_id": dx["patient_id"],
                "event_time": dx["admittime"],
                "code": code,
                "value": dx["long_title"].fillna("1"),
            }
        )

    def _convert_procedures(self, cohort: pd.DataFrame) -> pd.DataFrame:
        try:
            proc = self._read_hosp("procedures_icd.csv")
            proc_items = self._read_hosp("d_icd_procedures.csv")
        except FileNotFoundError as exc:
            warnings.warn(str(exc), stacklevel=2)
            return _empty_dynamic()

        proc = proc.dropna(subset=["hadm_id", "icd_code"])
        proc = _parse_datetime_columns(proc, ["chartdate"])
        proc = proc.merge(proc_items, on=["icd_code", "icd_version"], how="left")
        proc = proc.merge(cohort[["hadm_id", "patient_id", "admittime"]], on="hadm_id", how="inner")
        event_time = proc["chartdate"].where(proc["chartdate"].notna(), proc["admittime"])
        code = "proc:icd" + proc["icd_version"].astype(str) + ":" + proc["icd_code"].astype(str)
        return pd.DataFrame(
            {
                "patient_id": proc["patient_id"],
                "event_time": event_time,
                "code": code,
                "value": proc["long_title"].fillna("1"),
            }
        )

    def _convert_prescriptions(self, cohort: pd.DataFrame) -> pd.DataFrame:
        try:
            rx = self._read_hosp("prescriptions.csv")
        except FileNotFoundError as exc:
            warnings.warn(str(exc), stacklevel=2)
            return _empty_dynamic()

        rx = _parse_datetime_columns(rx, ["starttime"])
        rx = rx.dropna(subset=["hadm_id", "starttime", "drug"])
        rx = rx.merge(cohort[["hadm_id", "patient_id"]], on="hadm_id", how="inner")
        code = "rx:" + rx["drug"].map(_clean_code_text)
        return pd.DataFrame(
            {
                "patient_id": rx["patient_id"],
                "event_time": rx["starttime"],
                "code": code,
                "value": "1",
            }
        )

    def _build_notes(self, cohort: pd.DataFrame) -> pd.DataFrame:
        if self.notes_file is not None:
            notes = pd.read_csv(self.notes_file)
            patient_col = _first_existing(notes, ["patient_id", "hadm_id", "stay_id", "subject_id"])
            text_col = _first_existing(notes, ["text", "note_text", "TEXT"])
            if patient_col is None or text_col is None:
                raise ValueError("notes_file must contain a patient/hadm/stay id column and a text column.")
            notes = notes.rename(columns={patient_col: "patient_id", text_col: "text"})
            notes["patient_id"] = notes["patient_id"].astype(str)
            notes = notes.groupby("patient_id", as_index=False)["text"].agg(" ".join)
            return cohort[["patient_id"]].merge(notes, on="patient_id", how="left").fillna({"text": ""})

        return self._build_surrogate_notes(cohort)

    def _build_surrogate_notes(self, cohort: pd.DataFrame) -> pd.DataFrame:
        dx = self._convert_diagnoses(cohort)
        proc = self._convert_procedures(cohort)
        chunks: dict[str, list[str]] = {pid: [] for pid in cohort["patient_id"].astype(str)}
        for row in dx.itertuples(index=False):
            if row.value:
                chunks[str(row.patient_id)].append(f"diagnosis {row.value}")
        for row in proc.itertuples(index=False):
            if row.value:
                chunks[str(row.patient_id)].append(f"procedure {row.value}")
        for row in cohort.itertuples(index=False):
            pid = str(row.patient_id)
            chunks[pid].append(
                " ".join(
                    [
                        f"admission type {getattr(row, 'admission_type', '')}",
                        f"care unit {getattr(row, 'first_careunit', '')}",
                        f"race {getattr(row, 'race', '')}",
                        f"insurance {getattr(row, 'insurance', '')}",
                    ]
                )
            )
        return pd.DataFrame(
            {
                "patient_id": list(chunks.keys()),
                "text": [" . ".join(_deduplicate_text(parts)) for parts in chunks.values()],
            }
        )


def _read_csv(root: Path, name: str, **kwargs) -> pd.DataFrame:
    path = root / name
    if not path.exists():
        gz_path = root / f"{name}.gz"
        if gz_path.exists():
            path = gz_path
        else:
            raise FileNotFoundError(f"Expected file not found: {path}")
    return pd.read_csv(path, **kwargs)


def _parse_datetime_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def _clean_code_text(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_") or "unknown"


def _first_existing(df: pd.DataFrame, columns: list[str]) -> str | None:
    for col in columns:
        if col in df.columns:
            return col
    return None


def _deduplicate_text(parts: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for part in parts:
        normalized = " ".join(str(part).split())
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            result.append(normalized)
    return result


def _empty_dynamic() -> pd.DataFrame:
    return pd.DataFrame(columns=["patient_id", "event_time", "code", "value"])
