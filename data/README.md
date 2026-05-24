# Data Directory

Generated and downloaded data lives here and is ignored by git.

Expected layout:

```text
data/raw/mimic-iv-demo/2.2/
data/kg/primekg/
data/tables/mimic-iv-demo/mortality/
data/processed/mimic-iv-demo/mortality/
```

PrimeKG files are downloaded from the official Harvard Dataverse source:
https://doi.org/10.7910/DVN/IXA7BM

The EMERGE RAG path requires `data/kg/primekg/disease_features.tab` and
`data/kg/primekg/kg.csv`.
