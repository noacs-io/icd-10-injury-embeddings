# ICD-10 Injury Embeddings

## Abstract

### Background

Trauma patients present with heterogeneous injury patterns that are challenging to represent in statistical models. Traditional approaches
either use high-dimensional one-hot encoding, resulting in sparse features, or aggregate injuries into summary scores that lose
patient-specific detail. This study developed data-driven ICD-10 embeddings for trauma injuries and evaluated their ability to preserve
injury information.

### Methods

Using the National Trauma Data Bank, we trained autoencoder models on all trauma patients from 2018 to generate dense vector representations
of ICD-10 injury codes. We evaluated embeddings of dimensions 2, 4, 8, 16, and 32 against one-hot encoding using three prediction tasks:
in-hospital mortality, emergency department disposition, and blood transfusion within 24 hours. For each hospital included, we trained
separate logistic regression and LightGBM models using 2018 data from that hospital, then evaluated performance on 2019 data from the same
hospital. Performance was measured using area under the receiver operating characteristic curve (AUC) and stratified by hospital size.

### Results

In LightGBM models, 8-dimensional embeddings yielded AUC improvements compared to one-hot encoding of 0.08 (95% CI: 0.06, 0.10) in small
hospitals, 0.03 (0.02, 0.04) in medium hospitals, and 0.02 (0.01, 0.02) in large hospitals, with comparable performance in major hospitals
(0.00 [-0.01, 0.01]). In logistic regression, 32-dimensional embeddings showed AUC improvements of 0.03 (0.01, 0.05), 0.02 (0.01, 0.03), and
0.02 (0.02, 0.03) for small, medium, and large hospitals respectively, with similar performance in major hospitals (0.01 [0.00, 0.01]).

### Conclusion

ICD-10 embeddings with ≥8 dimensions preserve injury information whilst substantially reducing dimensionality. By capturing latent injury
relationships, these embeddings provide a method to represent traumatic injuries, particularly benefiting smaller centres with limited data.

## Python installation and usage

1. Install:

```bash
pip install "git+https://github.com/noacs-io/icd-10-injury-embeddings.git"
```

2. Basic usage:

```python

from icd_10_injury_embeddings import get_injury_embedding, icd_10_cm_to_international

icd_10_cm_to_international(["S06.0X0A", "S72001A", "s32.401a", "T14.90XA", "S01.01XA", "S72001A"])
# ['S060', 'S7200', 'S3240', 'T149', 'S010', 'S7200']

icd_10_codes = [["S065", "S066"], ["S270", "S2241", "S271"]]
embeddings = get_injury_embedding(icd_10_codes, dim=8)
```

## R installation and usage

1. Install the Torch backend:

```R
pak::pkg_install("torch")
torch::install_torch()
```

2. Install the package with pak or devtools:

```R
pak::pkg_install("noacs-io/icd-10-injury-embeddings")
```

3. Basic usage:

```R
library(icd10InjuryEmbeddings)

icd_10_cm_to_international(c("S06.0X0A", "S72001A", "s32.401a", "T14.90XA", "S01.01XA", "S72001A"))
# [1] "S060" "S7200" "S3240" "T149" "S010" "S7200"

icd_10_codes <- c("S065 S066", "S270 S2241 S271")
embeddings <- get_injury_embedding(icd_10_codes, dim = 8)
```

## Citation

If these embeddings contribute to your work, please cite:

```
TBD
```

## Training code

Reproducible training and evaluation scripts are provided in `python/training_code`. Note that the underlying NTDB data is not provided. The
`ntdb.py` script is the main entry point.

## License

Released under the MIT License (see `LICENSE`).
