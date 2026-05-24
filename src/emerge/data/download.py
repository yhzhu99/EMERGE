from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import requests


MIMIC_IV_DEMO_BASE_URL = "https://physionet.org/files/mimic-iv-demo/{version}/"

PRIMEKG_FILES = {
    "disease_features.tab": "https://dataverse.harvard.edu/api/access/datafile/6180618",
    "kg.csv": "https://dataverse.harvard.edu/api/access/datafile/6180620",
    "README.txt": "https://dataverse.harvard.edu/api/access/datafile/6191270",
}


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


def download_mimic_iv_demo(data_dir: Path, version: str = "2.2") -> Path:
    """Download the open MIMIC-IV demo.

    The PhysioNet page recommends recursive wget. The demo is open access and small
    enough for local smoke tests; it intentionally excludes free-text clinical notes.
    """

    output_dir = data_dir / "raw" / f"mimic-iv-demo-{version}"
    output_dir.mkdir(parents=True, exist_ok=True)
    base_url = MIMIC_IV_DEMO_BASE_URL.format(version=version)

    wget = shutil.which("wget")
    if wget:
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
    else:
        zip_path = output_dir / "mimic-iv-demo.zip"
        download_file(base_url.rstrip("/") + ".zip", zip_path)
    print(f"MIMIC-IV demo is available at {output_dir}")
    return output_dir


def download_primekg(data_dir: Path, full: bool = False) -> Path:
    output_dir = data_dir / "kg" / "primekg"
    output_dir.mkdir(parents=True, exist_ok=True)

    download_file(PRIMEKG_FILES["disease_features.tab"], output_dir / "disease_features.tab")
    download_file(PRIMEKG_FILES["README.txt"], output_dir / "README.txt")
    if full:
        download_file(PRIMEKG_FILES["kg.csv"], output_dir / "kg.csv")
    else:
        print("Skipped full PrimeKG kg.csv. Re-run with --full to enable graph triples.")
    print(f"PrimeKG files are available at {output_dir}")
    return output_dir
