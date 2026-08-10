# Fine-Tuning Qwen2.5-Coder for Text-to-SQL Generation

[![Hugging Face](https://img.shields.io/badge/Model-Qwen2.5--Coder--1.5B--Instruct-blueviolet)](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end pipeline for fine-tuning **Qwen2.5-Coder-1.5B-Instruct** on schema-constrained text-to-SQL generation using **QLoRA**. Built to run entirely on free-tier Kaggle/Colab GPUs (T4, 16GB).

---

## Results

| Metric | Base Model (Zero-Shot) | Fine-Tuned (QLoRA) | Delta |
| :--- | :---: | :---: | :---: |
| **Exact Match Accuracy** | 5.00% | **81.25%** | **+76.25 pp** |
| **Valid SQL Rate** | 92.00% | **97.25%** | **+5.25 pp** |

Evaluated on a held-out 400-row validation split from `b-mc2/sql-create-context`. Exact match uses normalized comparison (case, whitespace, numeric-literal quoting) rather than raw string equality.

**The fine-tuned model beats the zero-shot base model on both axes** — it generates the correct query far more often, *and* the SQL it produces executes cleanly more often.

### Why this took two iterations

The first fine-tuning pass (3,000 training rows, `r=16`) improved exact match (5%→56.67%) but caused a regression in valid-SQL rate (93.7%→88.3%). Tracing the failures showed the model wasn't making syntax mistakes — it was **hallucinating entirely wrong tables/columns from memorized training examples** when a validation question didn't closely resemble anything it had seen. Root cause: too little schema diversity (3,000 of ~78,000 available rows) for a dataset where table names are near-unique per row, encouraging memorization over general schema-reading.

Fix: scaled training data to 12,000 rows, increased LoRA capacity (`r=16→32`, `alpha=32→64`), and added an explicit schema-adherence instruction to the training prompt ("use only the table and column names exactly as given in the schema"). This resolved the hallucination pattern and pushed both metrics above baseline.

---

## Repository Structure

```text
nl2sql_finetune/
├── configs/
│   └── config.yaml                    # Model, data, LoRA, and training hyperparameters
├── data/
│   ├── train.parquet                  # 12,000-row training split
│   └── val.parquet                    # 400-row validation split
├── notebooks/
│   ├── 01-eda-sql.ipynb               # Dataset EDA, train/val split, push to repo
│   ├── 02-baseline-eval-sql.ipynb     # Zero-shot base model evaluation
│   ├── 03-qlora-fine-tuning-sql.ipynb # QLoRA fine-tuning with TRL's SFTTrainer
│   └── 04-finetuned-eval-sql.ipynb    # Fine-tuned model evaluation + comparison
├── outputs/
│   ├── models/sql_qlora_adapter/      # Trained LoRA adapter weights
│   └── results/                       # baseline_results.parquet, finetuned_results.parquet
├── requirements.txt
└── README.md
```

---

## Configuration

All parameters are controlled via [`configs/config.yaml`](configs/config.yaml):

```yaml
model:
  base_model: "Qwen/Qwen2.5-Coder-1.5B-Instruct"

data:
  hf_dataset: "b-mc2/sql-create-context"
  random_seed: 42
  train_size: 12000
  val_size: 400

lora:
  r: 32
  lora_alpha: 64
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
  lora_dropout: 0.1
  bias: "none"

training:
  num_train_epochs: 2
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-4
  max_seq_length: 512
  logging_steps: 10
  save_steps: 50
  save_total_limit: 2
  warmup_ratio: 0.03
```

---

## Dataset

[`b-mc2/sql-create-context`](https://huggingface.co/datasets/b-mc2/sql-create-context) (~78k schema/question/SQL triplets):

* **Schema:** `CREATE TABLE head (age INTEGER, department_id INT, ...)`
* **Question:** *"How many department heads are older than 56?"*
* **SQL:** `SELECT count(*) FROM head WHERE age > 56`

12,000 rows sampled for training, 400 held out for validation, seeded for reproducibility.

---

## Engineering Notes

- **Prompt format:** Chat-template messages (`<|im_start|>system/user/assistant`) with an explicit instruction to use only the table/column names given in the schema — this directly targets the memorization failure mode described above.
- **Post-processing:** Generated output is extracted from markdown code fences if present, truncated at the first hallucinated continuation token, and reduced to a single clean SQL line before scoring.
- **Evaluation metric:** Exact match is computed after normalizing case, whitespace, and numeric-literal quote style (`"116"` vs `116`), since raw string equality otherwise penalizes semantically-correct SQL. Valid SQL is checked by executing the generated query against an in-memory SQLite instance built from the row's own `CREATE TABLE` statement.
- **Memory:** `r=32` LoRA on a 1.5B model with `per_device_train_batch_size=4, gradient_accumulation_steps=4` (effective batch 16) fits comfortably within a T4's 16GB with gradient checkpointing enabled.
- **No experiment-tracking infra required** — results are tracked as versioned parquet/JSON files committed alongside the code, no MLflow or W&B server needed to reproduce or inspect results.

---

## Reproduce

```bash
git clone https://github.com/hossamhamdy333/AI_Portfolio.git
cd AI_Portfolio/nl2sql_finetune
pip install -r requirements.txt
```

Run the notebooks in order on a T4 (or better) GPU instance:

1. `01-eda-sql.ipynb` — no GPU required; builds `data/train.parquet` and `data/val.parquet`
2. `02-baseline-eval-sql.ipynb` — zero-shot baseline
3. `03-qlora-fine-tuning-sql.ipynb` — QLoRA training (~4 hours on a T4 for 12,000 rows / 2 epochs)
4. `04-finetuned-eval-sql.ipynb` — fine-tuned evaluation + baseline comparison
