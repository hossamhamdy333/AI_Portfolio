<div align="center">

# Fine-Tuning ALLaM-7B for Arabic Legal Instruction Following

QLoRA fine-tuning of **ALLaM-7B-Instruct-preview** for Egyptian/Saudi legal instruction-following across three task types: article analysis, plain-language legal simplification, and judgment prediction. Built to run entirely on free-tier GPUs (T4, 16GB, 4-bit quantized) — developed on Kaggle, notebooks are Colab-compatible too.

[![Hugging Face](https://img.shields.io/badge/Adapter-hossam3759180%2Fallam--qlora--legal--adapter-yellow)](https://huggingface.co/hossam3759180/allam-qlora-legal-adapter)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

`Python` `PyTorch` `Transformers` `PEFT/QLoRA` `bitsandbytes` `Gemini (LLM-judge)`

</div>

---

### Contents

- [Results](#results)
- [Why this took two rounds of iteration](#why-this-took-two-rounds-of-iteration)
- [Repository structure](#repository-structure)
- [Configuration](#configuration)
- [Dataset](#dataset)
- [Engineering notes](#engineering-notes)
- [Reproduce](#reproduce)

## Results

LLM-judged (Gemini) on the same fixed 150-row held-out sample, scored 1–10 on faithfulness, relevance, and fluency, across three stages: the zero-shot base model, the first fine-tuned adapter (v1), and the current adapter trained on the augmented dataset (v2).

| Metric | Base (Zero-Shot) | Fine-Tuned v1 | Fine-Tuned v2 |
|---|:---:|:---:|:---:|
| **Faithfulness** | 4.72 | 7.63 | **8.47** |
| **Relevance** | 7.15 | 8.77 | **9.44** |
| **Fluency** | 8.94 | 9.11 | **9.88** |

By task type (faithfulness):

| Task | Base | v1 | v2 |
|---|:---:|:---:|:---:|
| Analysis | 3.06 | 6.47 | **9.24** |
| Judgment Prediction | 4.37 | 5.84 | **7.03** |
| Simplification | 5.50 | 9.76 | 9.74 |

`analysis` was the target of the v2 change and moved the most (+2.76 over v1, +6.18 over base). `judgment_prediction` improved as a side effect. `simplification` — untouched by the v2 change — stayed flat, which is the expected result and a useful check that the `analysis` gain is real signal rather than general noise.

## Why this took two rounds of iteration

**Round 1 — getting past a broken baseline.** The first fine-tuning pass used only attention-layer LoRA (`q_proj`/`v_proj`, `r=16`) on the raw, imbalanced dataset and lost to the zero-shot baseline on every axis. Tracing the failures showed corrupted/dropped-letter Arabic and hallucinated content, concentrated in `analysis`, the smallest and most underrepresented task type (225 of 1,942 rows). Three fixes together resolved it:

1. **Expanded LoRA to include MLP layers** (`gate_proj`/`up_proj`/`down_proj`, not just attention) at `r=32` — LoRA papers generally attribute more of a model's factual capacity to the MLP layers than to attention alone.
2. **Oversampled `analysis` 3x** in training data to correct the class imbalance.
3. **Fixed evaluation itself**, which had independent bugs masking the real result: a missing `eos_token_id` caused the model to hallucinate fabricated follow-up instructions after a correct answer, and a missing `min_new_tokens` combined with repetition penalty caused ~35% of *baseline* generations to come back empty, artificially crushing the reference score. Once fixed, `repetition_penalty`/`no_repeat_ngram_size` were found to actively corrupt the fine-tuned model's Arabic (dropped letters mid-word) and were replaced with a stopping-criteria + truncation approach instead.

**Round 2 — fixing the data, not just the training.** Round 1's oversampling balanced the loss function but not the actual information the model saw: `analysis` was still only 225 *distinct* examples, duplicated 3x. `analysis` was correspondingly the weakest task in the v1 results. `notebooks/05_synthetic_data_engineering.ipynb` addresses this directly:

- Benchmarked four prompting strategies (zero-shot, few-shot, chain-of-thought, structured output) against real held-out examples, scored on schema validity and judged faithfulness, before picking one (few-shot won on faithfulness among the strategies with an acceptable schema-validity rate).
- Generated new, distinct `analysis` examples from raw legal text with that strategy, then ran every candidate through a curation pipeline (`src/synthetic_data.py`) — schema check, Arabic-language ratio check, PII regex, and Jaccard-shingle near-duplicate detection against the existing dataset — before any of it was trusted into the training set.
- Retrained (`03_qlora_fine_tuning.ipynb`) on the resulting `data/train_v2.parquet`, pushed as a separate `v2-synthetic` revision on Hugging Face Hub rather than overwriting the v1 adapter, and re-ran the identical evaluation (`04_finetuned_eval_allam.ipynb`) for a fair before/after comparison.

## Repository structure

```text
allam_finetune/
├── configs/
│   └── config.yaml                           # Model, data, QLoRA, and training hyperparameters
├── data/
│   ├── train.parquet                         # v1 training split (analysis oversampled 3x)
│   ├── train_v2.parquet                      # v2 training split (analysis: real + curated synthetic, no duplication)
│   ├── val.parquet                           # Held-out validation split
│   └── val_eval_sample.parquet               # 150-row LLM-judge evaluation sample, reused across all three runs
├── notebooks/
│   ├── 01-eda.ipynb                          # Dataset EDA, split, oversampling, push
│   ├── 02-baseline-eval.ipynb                # Zero-shot base model evaluation
│   ├── 03_qlora_fine_tuning.ipynb            # QLoRA fine-tuning + HF Hub push (v1 and v2)
│   ├── 04_finetuned_eval_allam.ipynb         # Fine-tuned model evaluation + comparison
│   └── 05_synthetic_data_engineering.ipynb   # Builds train_v2.parquet: strategy benchmark + curated generation
├── outputs/
│   ├── baseline_results.parquet              # Per-row zero-shot judge scores
│   ├── finetuned_results.parquet             # Per-row v1 judge scores
│   ├── finetuned_results_v2.parquet          # Per-row v2 judge scores
│   ├── models/allam_qlora_adapter/           # Adapter config + tokenizer (weights on HF Hub)
│   ├── reports/                              # eda_report.html, finetuned_metrics.json, finetuned_metrics_v2.json
│   └── figures/                              # length_distribution.png
├── src/
│   └── synthetic_data.py                     # Curation pipeline used by notebook 05
├── requirements.txt
└── README.md
```

**Adapter weights are hosted on Hugging Face Hub, not GitHub:** [hossam3759180/allam-qlora-legal-adapter](https://huggingface.co/hossam3759180/allam-qlora-legal-adapter) — a 7B model's LoRA weights at `r=32` across 7 target modules exceed GitHub's 100MB file limit. GitHub holds only the lightweight config/tokenizer metadata. `main` is the v1 adapter; `v2-synthetic` is the current one.

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

1,942 base rows. v1 corrected the `analysis` class imbalance by 3x duplication (2,392 training rows total); v2 replaces that duplication with genuine new examples — 225 original + curated synthetic, all distinct — bringing `analysis` to 1,005 unique rows (2,722 training rows total, no duplicated rows anywhere in the set).

## Engineering notes

- **Prompt format:** Alpaca-style (`### Instruction: / ### Input: / ### Response:`), matching the training format exactly — evaluation reuses the same template rather than a plain-text prompt, since the model is trained to treat `### Response:` as its generation cue.
- **Generation:** Greedy decoding with `eos_token_id` set and `min_new_tokens=20` to prevent premature/empty generations. A custom stopping criterion halts generation on the first hallucinated `### Input:` continuation, backed by a text-truncation safety net — `repetition_penalty` and `no_repeat_ngram_size` were tested and found to destabilize the fine-tuned model's Arabic token selection, so neither is used.
- **Evaluation:** LLM-as-judge (Gemini) scoring faithfulness, relevance, and fluency 1–10, on a fixed 150-row stratified sample reused identically across all three runs (base, v1, v2) for a fair comparison.
- **Synthetic data curation:** generated examples are only trusted into training after passing schema, language, PII, and near-duplicate checks against the real data (`src/synthetic_data.py`) — not used as-is from the generator.
- **Resumable long-running jobs:** both the synthetic-data generation loop and the judge-scoring loop checkpoint to disk after every item and skip already-completed work on restart, since both run against free-tier API quotas that can be exhausted or interrupted mid-run.
- **No experiment-tracking infra required** — results are tracked as versioned parquet/JSON files committed alongside the code; no MLflow or W&B server needed to reproduce or inspect results.

## Reproduce

```bash
git clone https://github.com/hossamhamdy333/AI_Portfolio.git
cd AI_Portfolio/allam_finetune
pip install -r requirements.txt
```

Run on a T4 (or better) GPU instance:

1. `01-eda.ipynb` — EDA, split, oversampling
2. `02-baseline-eval.ipynb` — zero-shot baseline
3. `05_synthetic_data_engineering.ipynb` — builds `train_v2.parquet` (optional — already committed; skip to go straight to training)
4. `03_qlora_fine_tuning.ipynb` — QLoRA training on `train_v2.parquet` + Hugging Face Hub push
5. `04_finetuned_eval_allam.ipynb` — loads the adapter directly from HF Hub, evaluates, compares to base and v1

The adapter loads directly from Hugging Face Hub in evaluation — no local weight files needed:

```python
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "hossam3759180/allam-qlora-legal-adapter", revision="v2-synthetic")
```
