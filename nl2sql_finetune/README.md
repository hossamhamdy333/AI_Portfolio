#  Fine-Tuning Qwen2.5-Coder for SQL Text-to-Code Generation

[![Hugging Face](https://img.shields.io/badge/Model-Qwen2.5--Coder--1.5B--Instruct-blueviolet)](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end repository for fine-tuning **Qwen2.5-Coder-1.5B-Instruct** on SQL text-to-code generation using **QLoRA (PEFT)**. Designed to optimize schema-constrained text-to-SQL generation and run efficiently on Kaggle and Google Colab GPUs.

---

##  Performance & Results

By standardizing prompt structures, refining SQL extraction logic, and fine-tuning with QLoRA using `TRL`'s `SFTTrainer`, the model achieved significant accuracy gains over the baseline:

| Metric / Configuration | Base Model (Zero-Shot) | Fine-Tuned (QLoRA) | Improvement |
| :--- | :---: | :---: | :---: |
| **Exact Match (EM) Accuracy** | **5.00%** | **56.67%** | **+51.67%** |
| **Trainable Parameters** | — | ~0.5% | Memory Efficient |
| **Training Execution** | — | 3 Epochs (~3,000 steps) | Fast Convergence |

> 💡 **Key Highlight:** The zero-shot baseline struggled heavily with domain-specific formatting, schema adherence, and extracting clean SQL statements without extra conversational filler. QLoRA adaptation eliminated hallucinated columns and strictly enforced valid SQL generation according to the provided `CREATE TABLE` schemas.

---

##  Engineering & Troubleshooting Highlights

During development, several key technical and engineering challenges were resolved:
- **Prompt Standardization:** Aligned input text structure to explicit schema context (`### Context:`), natural language instruction (`### Question:`), and strict output markers (`### Response:`).
- **Post-Processing & Extraction Logic:** Built robust SQL string extraction routines to evaluate generated queries cleanly against ground-truth outputs without trailing reasoning artifacts.
- **PEFT Hardware Optimization:** Resolved TRL/Transformers device mapping conflicts and memory overhead issues to enable smooth training on free-tier GPU instances (16GB VRAM limits).

---

##  Repository Structure

```text
sql_finetune/
├── configs/
│   └── config.yaml          # Full hyperparameter, data, & LoRA configurations
├── data/                    # Data directory (supports cached local datasets)
├── notebooks/
│   └── 01-eda-sql.ipynb     # Exploratory Data Analysis & evaluation suite
├── train.py                 # Fine-tuning script using TRL's SFTTrainer
├── infer.py                 # Standalone script for inference & evaluation
└── README.md
```

---

##  Configuration Specs

All experimental parameters are controlled via [`configs/config.yaml`](configs/config.yaml):

```yaml
model:
  base_model: "Qwen/Qwen2.5-Coder-1.5B-Instruct"

data:
  hf_dataset: "b-mc2/sql-create-context"
  random_seed: 42
  train_size: 3000
  val_size: 300

lora:
  r: 16
  lora_alpha: 32
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
  lora_dropout: 0.05
  bias: "none"

training:
  num_train_epochs: 3
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 2
  learning_rate: 2.0e-4
  max_seq_length: 512
  logging_steps: 10
  save_steps: 50
  save_total_limit: 2
  warmup_ratio: 0.03
```

---

##  Dataset Overview

Trained on the [`b-mc2/sql-create-context`](https://huggingface.co/datasets/b-mc2/sql-create-context) dataset (~78k triplets):

* **Context (`schema`):** `CREATE TABLE head (age INTEGER, department_id INT, ...)`
* **Question (`instruction`):** *"How many department heads are older than 56?"*
* **Answer (`target`):** `SELECT count(*) FROM head WHERE age > 56`

---

##  Quick Start

### 1. Installation

```bash
git clone https://github.com/hossamhamdy333/AI_Portfolio.git
cd AI_Portfolio/sql_finetune
pip install -q transformers datasets peft trl accelerate bitsandbytes pyyaml
```

### 2. Fine-Tuning

Run training using your target configuration:

```bash
python train.py --config configs/config.yaml
```

### 3. Inference & Evaluation

Evaluate your fine-tuned LoRA adapter on custom schema instructions:

```bash
python infer.py --base_model "Qwen/Qwen2.5-Coder-1.5B-Instruct" --adapter_path "./results/final_model"
```
