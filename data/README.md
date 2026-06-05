# FLaG Training Data

This directory contains the **AMPCliff AC fix split** used for FLaG downstream training and evaluation.

## Layout

```
data/blosum62 average/diff_5-trd_0.9/
├── grampa_e_coli_7_25-{train,valid,test,all-train,diff5-pairs}.csv
└── grampa_s_aureus_7_25-{train,valid,test,all-train,diff5-pairs}.csv
```

- **Condition:** BLOSUM62 average similarity
- **Activity cliff threshold:** diff = 5, similarity threshold = 0.9
- **Datasets:** `e_coli`, `s_aureus`

Shell training scripts resolve paths as:

```text
${REPO_ROOT}/data/blosum62 average/diff_5-trd_0.9/grampa_{dataset}_7_25-{split}.csv
```

Hydra configs use the same layout via `./data/blosum62 average/diff_{diff}-trd_{threshold}/...` placeholders (see `utils/path_helper.py`).

## Citation

If you use this dataset, please cite the published AMPCliff paper:

- DOI: https://doi.org/10.1016/j.jare.2025.04.046
- *Journal of Advanced Research*, 2026, Vol. 80, pp. 287–300

For full activity-cliff dataset generation and split procedures, see [AMPCliff-generation](https://github.com/Kewei2023/AMPCliff-generation).
