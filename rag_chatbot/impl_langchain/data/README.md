# Data

Empty by design until you run the notebooks.

`01_build_retriever.ipynb` expects `xlsum_arabic_clean.parquet` here — copy or
symlink it from `impl_vanilla/data/processed/xlsum_arabic_clean.parquet`
(pull via DVC there first if you don't have it locally:
`dvc pull data/processed/xlsum_arabic_clean.parquet.dvc`).

`03_evaluation.ipynb` will read the eval set directly from
`impl_vanilla/outputs/synthetic_qa/synthetic_qa_pairs.parquet` — no copy
needed there, same file, same questions, so the comparison stays controlled.
