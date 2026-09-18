---
base_model: Qwen/Qwen2.5-Coder-1.5B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:Qwen/Qwen2.5-Coder-1.5B-Instruct
- lora
- qlora
- sft
- transformers
- trl
- text-to-sql
---

<div align="center">

# Qwen2.5-Coder-1.5B QLoRA Adapter — Text-to-SQL

`peft` `trl` `transformers` `bitsandbytes` `Qwen/Qwen2.5-Coder-1.5B-Instruct`

</div>

---

LoRA adapter for `Qwen/Qwen2.5-Coder-1.5B-Instruct`, fine-tuned with QLoRA for schema-constrained text-to-SQL: given a question and a `CREATE TABLE` schema, generate a SQL query.

This folder is the adapter's model card. The full data analysis, training details, scoring caveats, and limitations are in the parent project's [README](../../../README.md).

## Adapter configuration

From `adapter_config.json` and the parent README:

| Setting | Value |
|---|---|
| PEFT type / task | LoRA / `CAUSAL_LM` |
| Rank / alpha / dropout | 32 / 64 / 0.1 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` (attention only, no MLP) |
| Trainable parameters | 8,716,288 of 1,552,430,592 (0.5615%) |
| Saved with PEFT | 0.19.1 |

## Training

- **Data:** [`b-mc2/sql-create-context`](https://huggingface.co/datasets/b-mc2/sql-create-context) — 12,000 train / 400 validation rows, seeded split
- **Recipe:** TRL `SFTTrainer`, 2 epochs, effective batch 16, learning rate 2e-4, `max_seq_length=512`, 3% warmup; 4-bit NF4 base with double quantization and bf16 compute
- **Hardware:** single T4 (Kaggle), about 4 h 13 min

## Results

On the 400-row validation split (same split for base and fine-tuned):

| Metric | Base (zero-shot) | Fine-tuned |
|---|---|---|
| Exact match | 5.00% | 81.25% |
| Valid SQL rate (executes on SQLite) | 92.00% | 97.25% |

Exact match is a string comparison and understates quality where quoting style differs from the gold query; the parent README explains this and other scoring caveats.

## Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

base_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained(base_id)
base = AutoModelForCausalLM.from_pretrained(base_id, quantization_config=bnb, device_map="auto")
model = PeftModel.from_pretrained(base, "outputs/models/sql_qlora_adapter")  # run from the project root
```

## Limitations

The dataset is dominated by single-table, filter-only queries (only 2.3% contain a `JOIN`), so results say little about multi-table or grouped SQL.
