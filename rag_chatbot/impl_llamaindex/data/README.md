# Data

Empty by design until you run the notebooks.

- `01_build_corpus.ipynb` → produces `processed/{sports,tech,history,english_literature}.parquet`
- `02_ingest_and_router.ipynb` → produces `indexes/<domain>/` (persisted VectorStoreIndexes)
- `03_synthetic_qa.ipynb` → produces `synthetic_qa_pairs.parquet`

Run them in that order — each one depends on the previous notebook's output.
