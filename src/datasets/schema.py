from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DYNAMIC_COLUMNS = ["patient_id", "event_time", "code", "value"]
STATIC_REQUIRED_COLUMNS = ["patient_id"]
LABEL_COLUMNS = ["patient_id", "label_time", "label_code", "label_value"]


@dataclass(frozen=True)
class OneEHRPaths:
    root: Path
    dynamic: Path
    static: Path
    label: Path
    cohort: Path
    notes: Path


@dataclass
class OneEHRDataset:
    dynamic: pd.DataFrame
    static: pd.DataFrame
    label: pd.DataFrame
    cohort: pd.DataFrame | None = None
    notes: pd.DataFrame | None = None

    def validate(self) -> None:
        _require_columns(self.dynamic, DYNAMIC_COLUMNS, "dynamic")
        _require_columns(self.static, STATIC_REQUIRED_COLUMNS, "static")
        _require_columns(self.label, LABEL_COLUMNS, "label")

        self.dynamic["patient_id"] = self.dynamic["patient_id"].astype(str)
        self.dynamic["event_time"] = pd.to_datetime(self.dynamic["event_time"], errors="raise")
        self.dynamic["code"] = self.dynamic["code"].astype(str)

        self.static["patient_id"] = self.static["patient_id"].astype(str)

        self.label["patient_id"] = self.label["patient_id"].astype(str)
        self.label["label_time"] = pd.to_datetime(self.label["label_time"], errors="raise")
        self.label["label_code"] = self.label["label_code"].astype(str)

        if self.cohort is not None:
            _require_columns(self.cohort, ["patient_id"], "cohort")
            self.cohort["patient_id"] = self.cohort["patient_id"].astype(str)
        if self.notes is not None:
            _require_columns(self.notes, ["patient_id", "text"], "notes")
            self.notes["patient_id"] = self.notes["patient_id"].astype(str)

    def save(self, output_dir: Path) -> OneEHRPaths:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.validate()

        paths = OneEHRPaths(
            root=output_dir,
            dynamic=output_dir / "dynamic.csv",
            static=output_dir / "static.csv",
            label=output_dir / "label.csv",
            cohort=output_dir / "cohort.csv",
            notes=output_dir / "notes.csv",
        )
        self.dynamic[DYNAMIC_COLUMNS].to_csv(paths.dynamic, index=False)
        self.static.to_csv(paths.static, index=False)
        self.label[LABEL_COLUMNS].to_csv(paths.label, index=False)
        if self.cohort is not None:
            self.cohort.to_csv(paths.cohort, index=False)
        if self.notes is not None:
            self.notes.to_csv(paths.notes, index=False)
        return paths


def _require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{name} table missing required columns: {missing}")


def load_oneehr_tables(root: Path) -> OneEHRDataset:
    root = Path(root)
    dynamic = pd.read_csv(root / "dynamic.csv")
    static = pd.read_csv(root / "static.csv")
    label = pd.read_csv(root / "label.csv")
    cohort_path = root / "cohort.csv"
    notes_path = root / "notes.csv"
    dataset = OneEHRDataset(
        dynamic=dynamic,
        static=static,
        label=label,
        cohort=pd.read_csv(cohort_path) if cohort_path.exists() else None,
        notes=pd.read_csv(notes_path) if notes_path.exists() else None,
    )
    dataset.validate()
    return dataset
