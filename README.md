# EMERGE

Official implementation of the paper [**EMERGE: Enhancing Multimodal Electronic Health Records Predictive Modeling with Retrieval-Augmented Generation**](https://dl.acm.org/doi/10.1145/3627673.3679582), accepted by **CIKM 2024**.

EMERGE enhances multimodal EHR prediction by extracting abnormal time-series signals and disease entities from clinical text, retrieving disease knowledge from PrimeKG, generating patient-level RAG summaries, and fusing time-series, clinical-note, and RAG-summary representations with cross-attention.

## Project Layout

```text
.
├── configs/                 # TOML experiment configs
├── src/
│   ├── main.py              # argparse entry point for pipeline stages
│   ├── config.py            # dataclass-based TOML config loader
│   ├── datasets/            # MIMIC-IV conversion, OneEHR schema, RAG preprocessing
│   ├── evaluation/          # AUROC, AUPRC, min(+P, Se), bootstrap metrics
│   ├── models/              # EMERGE model and fusion modules
│   └── training/            # training loop and artifact writing
├── tests/                   # unit and smoke tests
└── data/                    # ignored local raw/table/processed/KG data
```

The reusable EHR input format follows the OneEHR three-table convention:

```text
dynamic.csv: patient_id,event_time,code,value
static.csv:  patient_id,age,sex,...
label.csv:   patient_id,label_time,label_code,label_value
```

EMERGE-specific artifacts are derived from these tables and written under `data/processed/...`.

## Setup

```bash
uv sync
source .venv/bin/activate
```

No package installation step is required; run the source entry point directly.

## Quickstart

Run the MIMIC-IV demo pipeline:

```bash
python src/main.py --stage all --config configs/mimic4_demo.toml
```

Or run stages separately:

```bash
python src/main.py --stage download --config configs/mimic4_demo.toml
python src/main.py --stage convert --config configs/mimic4_demo.toml
python src/main.py --stage preprocess --config configs/mimic4_demo.toml
python src/main.py --stage train --config configs/mimic4_demo.toml --seed 42
```

Default outputs:

```text
data/raw/mimic-iv-demo/2.2/                 # PhysioNet MIMIC-IV demo
data/kg/primekg/                            # PrimeKG files
data/tables/mimic-iv-demo/mortality/        # OneEHR dynamic/static/label tables
data/processed/mimic-iv-demo/mortality/     # EMERGE tensors, RAG records, split
runs/mimic4_demo_mortality/                 # checkpoints, metrics, predictions
```

By default, the PrimeKG download is lightweight and includes `README.txt` and `disease_features.tab`, matching the current Dataverse filenames. Use `--primekg-full` to download all PrimeKG files listed in the original README, including `nodes.tab`, `edges.csv`, `kg.csv`, `drug_features.tab`, and grouped/intermediate KG files.

## Experiment Scope

The default config uses the public MIMIC-IV demo. It validates the full data, RAG, and training pipeline, but it is too small for the paper's benchmark numbers. For the reported tables, use the full MIMIC-III and MIMIC-IV benchmark cohorts with the paper's preprocessing, 70/10/20 split, and reported hyperparameters.

## Citation

ACM reference:

> Yinghao Zhu, Changyu Ren, Zixiang Wang, Xiaochen Zheng, Shiyun Xie, Junlan Feng, Xi Zhu, Zhoujun Li, Liantao Ma, and Chengwei Pan. 2024. EMERGE: Enhancing Multimodal Electronic Health Records Predictive Modeling with Retrieval-Augmented Generation. In Proceedings of the 33rd ACM International Conference on Information and Knowledge Management (CIKM '24). https://doi.org/10.1145/3627673.3679582

BibTeX:

```bibtex
@inproceedings{zhu2024emerge,
  title = {EMERGE: Enhancing Multimodal Electronic Health Records Predictive Modeling with Retrieval-Augmented Generation},
  author = {Zhu, Yinghao and Ren, Changyu and Wang, Zixiang and Zheng, Xiaochen and Xie, Shiyun and Feng, Junlan and Zhu, Xi and Li, Zhoujun and Ma, Liantao and Pan, Chengwei},
  booktitle = {Proceedings of the 33rd ACM International Conference on Information and Knowledge Management},
  series = {CIKM '24},
  year = {2024},
  publisher = {Association for Computing Machinery},
  doi = {10.1145/3627673.3679582}
}
```
