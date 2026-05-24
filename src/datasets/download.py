from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import requests

MIMIC_IV_DEMO_BASE_URL = "https://physionet.org/files/mimic-iv-demo/{version}/"
PRIMEKG_DATASET_API = "https://dataverse.harvard.edu/api/datasets/:persistentId/"
PRIMEKG_PERSISTENT_ID = "doi:10.7910/DVN/IXA7BM"
PRIMEKG_DATASET_URL = "https://doi.org/10.7910/DVN/IXA7BM"
PRIMEKG_LIGHTWEIGHT_FILES = ("README.txt", "disease_features.tab")
PRIMEKG_REQUIRED_FILES = ("README.txt", "disease_features.tab", "kg.csv")
PRIMEKG_FULL_FILES = (
    "README.txt",
    "nodes.tab",
    "edges.csv",
    "kg.csv",
    "disease_features.tab",
    "drug_features.tab",
    "kg_raw.csv",
    "kg_giant.csv",
    "kg_grouped.csv",
    "kg_grouped_diseases.tab",
    "kg_grouped_diseases_bert_map.tab",
)


def download_file(url: str, output_path: Path, chunk_size: int = 1024 * 1024) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"Already exists: {output_path}")
        return output_path

    print(f"Downloading {url} -> {output_path}")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        with tmp_path.open("wb") as fout:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    fout.write(chunk)
        tmp_path.replace(output_path)
    return output_path


def download_mimic_iv_demo(output_dir: Path, version: str = "2.2") -> Path:
    """Download the open MIMIC-IV demo.

    The PhysioNet page recommends recursive wget. The demo is open access and small
    enough for local smoke tests; it intentionally excludes free-text clinical notes.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_url = MIMIC_IV_DEMO_BASE_URL.format(version=version)
    wget = shutil.which("wget")
    if wget is None:
        raise RuntimeError("wget is required for recursive MIMIC-IV demo download.")
    subprocess.run(
        [
            wget,
            "-r",
            "-N",
            "-c",
            "-np",
            "-nH",
            "--cut-dirs=3",
            "-P",
            str(output_dir),
            base_url,
        ],
        check=True,
    )
    print(f"MIMIC-IV demo is available at {output_dir}")
    return output_dir


def download_primekg(output_dir: Path, *, full: bool = False, lightweight: bool = False) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if full and lightweight:
        raise ValueError("PrimeKG download cannot be both full and lightweight.")

    files = _primekg_dataverse_files()
    if full:
        wanted = PRIMEKG_FULL_FILES
    elif lightweight:
        wanted = PRIMEKG_LIGHTWEIGHT_FILES
    else:
        wanted = PRIMEKG_REQUIRED_FILES

    for filename in wanted:
        if filename not in files:
            raise FileNotFoundError(
                f"PrimeKG official Dataverse file not found: {filename} "
                f"from {PRIMEKG_DATASET_URL}"
            )
        download_file(files[filename], output_dir / filename)
    if lightweight:
        print("Downloaded lightweight PrimeKG files for smoke tests only.")
    elif full:
        print("Downloaded all official PrimeKG files listed on Dataverse.")
    else:
        print("Downloaded official PrimeKG files required by EMERGE RAG.")
    print(f"PrimeKG files are available at {output_dir}")
    return output_dir


def _primekg_dataverse_files() -> dict[str, str]:
    response = requests.get(
        PRIMEKG_DATASET_API,
        params={"persistentId": PRIMEKG_PERSISTENT_ID},
        timeout=60,
    )
    response.raise_for_status()
    files = response.json()["data"]["latestVersion"]["files"]
    resolved: dict[str, str] = {}
    for item in files:
        data_file = item["dataFile"]
        filename = str(data_file["filename"])
        file_id = int(data_file["id"])
        resolved[filename] = f"https://dataverse.harvard.edu/api/access/datafile/{file_id}"
    return resolved
