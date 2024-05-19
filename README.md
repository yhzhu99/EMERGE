# EMERGE: Enhancing Multimodal Electronic Health Records Predictive Modeling with Retrieval-Augmented Generation

Under review

## Usage

Run `pipeline.ipynb` to see model architecture, train, validate, and test the model.

## Folder Structure

```bash
datasets/
    mimic3/
        data_train.h5
        data_val.h5
        data_test.h5
    mimic4/
        ...
models/
    transformer.py
utils/
    binary_classification_metrics.py
    ...
logs/
    # checkpoints are saved here
pipeline.ipynb # the code for EMERGE
```
