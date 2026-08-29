<div align="center">

# Fine-Tuning ALLaM-7B for Arabic Legal Instruction Following

QLoRA fine-tuning of **ALLaM-7B-Instruct-preview** for Egyptian/Saudi legal instruction-following across three task types: article analysis, plain-language legal simplification, and judgment prediction. Built to run entirely on free-tier Kaggle GPUs (T4, 16GB, 4-bit quantized).

[![Hugging Face](https://img.shields.io/badge/Adapter-hossam3759180%2Fallam--qlora--legal--adapter-yellow)](https://huggingface.co/hossam3759180/allam-qlora-legal-adapter)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

`Python` `PyTorch` `Transformers` `PEFT/QLoRA` `bitsandbytes` `Gemini (LLM-judge)`

</div>

---

### Contents

- [Results](#results)
- [Why this took several iterations](#why-this-took-several-iterations)
- [Repository structure](#repository-structure)
- [Configuration](#configuration)
- [Dataset](#dataset)
- [Engineering notes](#engineering-notes)
- [Reproduce](#reproduce)

## Results

LLM-judged (Gemini) on a held-out 150-row sample, scored 1–10 on faithfulness, relevance, and fluency.

| Metric | Base Model (Zero-Shot) | Fine-Tuned (QLoRA) | Delta |
|---|:---:|:---:|:---:|
| **Faithfulness** | 4.72 | **7.63** | **+2.91** |
| **Relevance** | 7.15 | **8.77** | **+1.62** |
| **Fluency** | 8.94 | **9.11** | **+0.17** |

By task type:

| Task | Faithfulness (base → fine-tuned) | Relevance | Fluency |
|---|:---:|:---:|:---:|
| Analysis | 3.06 → **6.47** | 4.76 → **7.41** | 8.53 → 8.71 |
| Judgment Prediction | 4.37 → **5.84** | 7.49 → **8.37** | 8.79 → 8.45 |
| Simplification | 5.50 → **9.76** | 7.42 → **9.53** | 9.20 → **9.88** |

**The fine-tuned model beats the zero-shot base model on every metric and every task type.**

## Why this took several iterations

The first fine-tuning pass used only attention-layer LoRA (`q_proj`/`v_proj`, `r=16`) on the raw, imbalanced dataset and lost to baseline on every axis — tracing the failures showed corrupted/dropped-letter Arabic and hallucinated content, concentrated in `analysis`, the smallest and most underrepresented task type (225 of 1,942 rows).

Three fixes, applied together:

1. **Expanded LoRA to include MLP layers** (`gate_proj`/`up_proj`/`down_proj`, not just attention) at `r=32` — most of a model's factual/faithfulness capacity lives in the MLP layers, not attention alone.
2. **Oversampled `analysis` 3x** in training data to correct the imbalance.
3. **Fixed evaluation itself**, which had independent bugs masking the real result: a missing `eos_token_id` caused the model to hallucinate fabricated follow-up instructions after a correct answer, and a missing `min_new_tokens` combined with repetition penalty caused ~35% of *baseline* generations to come back empty, artificially crushing the reference score. Once both were fixed, `repetition_penalty`/`no_repeat_ngram_size` were found to actively corrupt the fine-tuned model's Arabic (dropped letters mid-word) — replaced with a stopping-criteria + truncation approach instead, which resolved it cleanly.

## Repository structure

```text
allam_finetune/
├── configs/
│   └── config.yaml                       # Model, data, QLoRA, and training hyperparameters
├── data/
│   ├── train.parquet                     # Training split (oversampled on `analysis`)
│   ├── val.parquet                       # Held-out validation split
│   └── val_eval_sample.parquet           # 150-row LLM-judge evaluation sample
├── notebooks/
│   ├── 01-eda.ipynb                      # Dataset EDA, split, oversampling, push
│   ├── 02-baseline-eval.ipynb            # Zero-shot base model evaluation
│   ├── 03-qlora-fine-tuning.ipynb        # QLoRA fine-tuning + HF Hub push
│   └── 04-finetuned-eval-allam.ipynb     # Fine-tuned model evaluation + comparison
├── outputs/
│   ├── models/allam_qlora_adapter/       # Adapter config + tokenizer (weights on HF Hub)
│   ├── results/                          # baseline_results.parquet, finetuned_results.parquet
│   ├── reports/                          # eda_report.html, finetuned_metrics.json
│   └── figures/                          # length_distribution.png
├── requirements.txt
└── README.md
```

**Adapter weights are hosted on Hugging Face Hub, not GitHub:** [hossam3759180/allam-qlora-legal-adapter](https://huggingface.co/hossam3759180/allam-qlora-legal-adapter) — a 7B model's LoRA weights at `r=32` across 7 target modules exceed GitHub's 100MB file limit. GitHub holds only the lightweight config/tokenizer metadata.

## Configuration

```yaml
model:
  base_model: "ALLaM-AI/ALLaM-7B-Instruct-preview"

qlora:
  r: 32
  lora_alpha: 64
  lora_dropout: 0.1
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
  bits: 4

training:
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  num_train_epochs: 2
  learning_rate: 2.0e-4
  warmup_ratio: 0.03
  lr_scheduler_type: "cosine"

evaluation:
  benchmark: "mt_bench_style"
  llm_judge_model: "gemini-3.1-flash-lite"
  temperature: 0.0
```

## Dataset

Egyptian and Saudi legal QA pairs across three task types:

- **Analysis** — extract key legal points and domain from a statute excerpt
- **Simplification** — plain-language explanation of legal text
- **Judgment prediction** — predict/reproduce court ruling language from case facts

1,942 base rows, oversampled to correct class imbalance before training (`analysis` tripled to reduce underrepresentation-driven quality gap).

## Engineering notes

- **Prompt format:** Alpaca-style (`### Instruction: / ### Input: / ### Response:`), matching the training format exactly — evaluation reuses the same template rather than a plain-text prompt, since the model is trained to treat `### Response:` as its generation cue.
- **Generation:** Greedy decoding with `eos_token_id` set and `min_new_tokens=20` to prevent premature/empty generations. A custom stopping criterion halts generation on the first hallucinated `### Input:` continuation, backed by a text-truncation safety net — `repetition_penalty` and `no_repeat_ngram_size` were tested and found to destabilize the fine-tuned model's Arabic token selection, so neither is used.
- **Evaluation:** LLM-as-judge (Gemini) scoring faithfulness, relevance, and fluency 1–10, on a fixed 150-row stratified sample reused identically across baseline and fine-tuned runs for a fair comparison.
- **No experiment-tracking infra required** — results are tracked as versioned parquet/JSON files committed alongside the code; no MLflow or W&B server needed to reproduce or inspect results.

## Reproduce

```bash
git clone https://github.com/hossamhamdy333/AI_Portfolio.git
cd AI_Portfolio/allam_finetune
pip install -r requirements.txt
```

Run the notebooks in order on a T4 (or better) GPU instance:

1. `01-eda.ipynb` — EDA, split, oversampling
2. `02-baseline-eval.ipynb` — zero-shot baseline
3. `03-qlora-fine-tuning.ipynb` — QLoRA training + Hugging Face Hub push
4. `04-finetuned-eval-allam.ipynb` — loads the adapter directly from HF Hub, evaluates, compares to baseline

The adapter loads directly from Hugging Face Hub in evaluation — no local weight files needed:

```python
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "hossam3759180/allam-qlora-legal-adapter")
```
