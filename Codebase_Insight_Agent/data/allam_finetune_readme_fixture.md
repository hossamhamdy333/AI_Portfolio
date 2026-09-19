<div align="center">

# Fine-Tuning ALLaM-7B for Arabic Legal Instruction Following

`transformers` `ALLaM-AI/ALLaM-7B-Instruct-preview` `google-genai` `pandas` `scikit-learn` `pydantic` `dvc` `pytest` `hossam3759180/allam-qlora-legal-adapter`

</div>

---

### Contents

- [Summary](#summary)
- [Problem & motivation](#problem--motivation)
- [Approach](#approach)
- [Data](#data)
- [Results](#results)
- [What I'd do differently / limitations](#what-id-do-differently--limitations)
- [Stack](#stack)

---

## Summary

QLoRA fine-tunes `ALLaM-AI/ALLaM-7B-Instruct-preview`, a 7B Arabic-native instruct model, on Egyptian/Saudi legal instruction data across three task types (article analysis, plain-language simplification, judgment prediction), entirely on a free-tier 16GB T4 in 4-bit. Judged by Gemini on a fixed 150-row held-out sample, faithfulness goes from 4.72 (zero-shot base) to 7.63 (first fine-tune, v1) to 8.47 (current adapter, v2). The interesting part isn't the fine-tune itself, it's that the first attempt actually lost to the zero-shot baseline, and getting past that took fixing three independent things — the LoRA target modules, the class balance, and the evaluation script itself, which was silently zeroing out ~35% of the baseline's own generations before any of the "real" problems were even visible.

## Problem & motivation

Naively, fine-tuning an instruct model on a few thousand instruction/response pairs should just work: pick a LoRA config, point `SFTTrainer` at the data, done. Two things make that undersell this problem specifically. First, the source data isn't clean by task type — one of the two raw datasets is 75% `judgment_prediction`, and that task type also happens to have a small pool of recurring judge names and boilerplate ruling language, which is exactly the condition under which a model memorizes phrasing instead of learning the task; near-zero training loss on an unbalanced legal dataset is a red flag, not a win. Second, an LLM-judge evaluation script has its own failure modes independent of the model being evaluated — a missing `min_new_tokens` here was silently producing empty generations for roughly a third of the *baseline* model's outputs, which artificially crushed the reference score the fine-tuned model was supposed to beat. Getting a trustworthy before/after number meant fixing the yardstick before trusting anything it measured.

The naive path — attention-only LoRA, whatever class balance the raw data happens to have, default generation settings — was tried first here and failed outright: round 1's model lost to zero-shot on every axis until three separate fixes were applied together.

## Approach

### Data assembly

(`01-eda.ipynb`): two raw sources are merged — `fr3on/eg-legal-instruction-following` (Egyptian legal; `classification` and `keyword_extraction` task types dropped, 2,092 rows, because both collapsed to near-duplicate boilerplate outputs and added no real diversity) and `mbayan/Arabic-LJP` (Saudi legal judgment prediction, 3,752 rows). After exact-duplicate removal (585 duplicate rows in the Egyptian source alone, 28% of it) and a length filter (445 rows dropped outside a 10–2048 word bound), `judgment_prediction` still made up 3,610 of 4,814 rows (75%) — identified as the likely driver of both near-zero training loss and fabricated-verdict hallucinations seen later, so it's capped down to 954 rows (matching the next-largest class) before the train/val split, rather than left to dominate.

### QLoRA fine-tuning

(`03_qlora_fine_tuning.ipynb`, config in `configs/config.yaml`): 4-bit QLoRA, rank 32, alpha 64, dropout 0.1, across all seven attention/MLP projection matrices (`q/k/v/o_proj`, `gate/up/down_proj`). 2 epochs, per-device batch size 4 with 4 gradient-accumulation steps, learning rate 2e-4, cosine schedule, 3% warmup. Alpaca-style prompt format (`### Instruction: / ### Input: / ### Response:`), matched exactly between training and evaluation since the model is trained to treat `### Response:` as its generation cue. The adapter is pushed to Hugging Face Hub as a separate revision per round (`main` = v1, `v2-synthetic` = current) rather than committed to GitHub — a 7B model's rank-32 LoRA weights across 7 modules exceed GitHub's 100MB file limit, so GitHub holds only config/tokenizer metadata.

### Evaluation

(`02-baseline-eval.ipynb`, `04_finetuned_eval_allam.ipynb`): a 150-row sample, stratified by task type and drawn once from the validation split, is reused byte-for-byte across all three evaluated stages (base, v1, v2) so the comparison is apples-to-apples. Gemini (`gemini-3.1-flash-lite`, temperature 0.0, JSON-mode output) scores each generation 1–10 on faithfulness, relevance, and fluency against the real reference response. Both the generation loop and the judging loop checkpoint to a parquet file after every row and skip completed work on restart, since both run against a free-tier API quota that gets exhausted or interrupted mid-run in practice, not hypothetically — the notebooks show exactly that happening (a `RESOURCE_EXHAUSTED` hit mid-batch, resumed on rerun).

### Round-1 fixes

(what turned a losing model into a winning one): expanding LoRA from attention-only (`q_proj`/`v_proj`, r=16) to all seven attention+MLP modules at r=32, since MLP layers generally hold more of a model's factual capacity than attention alone; 3x-oversampling the `analysis` task type in the training set to correct its class imbalance; and fixing the evaluation script itself — a missing `eos_token_id` was letting the model hallucinate fabricated follow-up instructions after a correct answer, and a missing `min_new_tokens` combined with `repetition_penalty` was returning empty generations for roughly 35% of *baseline* rows. Once fixed, `repetition_penalty` and `no_repeat_ngram_size` were found to actively corrupt the fine-tuned model's Arabic (dropped letters mid-word) and were removed in favor of a stopping-criteria-plus-truncation approach that halts on the first hallucinated `### Input:` continuation.

### Round-2 fix — synthetic data, not just reweighted data

(`05_synthetic_data_engineering.ipynb`, `src/synthetic_data.py`): round 1's 3x oversampling balanced the loss function but not what the model actually saw — `analysis` was still only 225 distinct examples repeated three times, and it stayed the weakest task in the v1 results. This notebook generates genuinely new `analysis` examples instead. Four prompting strategies were benchmarked on a 20-example sample before picking one, scored on schema-validity rate (does the output match the required two-header Arabic format) and a 10-of-20 spot-check judged for faithfulness:

| Strategy | Schema-valid rate | Faithfulness (spot-check) |
|---|---|---|
| Zero-shot | 0.00 | 6.00 |
| **Few-shot (3 real examples)** | **0.95** | **9.90** |
| Chain-of-thought | 1.00 | 8.20 |
| Structured output (Gemini `response_schema`) | 0.95 | 7.56 |

Zero-shot's 0% schema-valid rate ruled it out immediately — it never reliably produced the required format at all. Chain-of-thought had a perfect schema rate but noticeably lower faithfulness than few-shot; few-shot won on the metric that actually matters for training-data quality. The full batch (839 candidates) then went through `synthetic_data.py`'s curation pipeline — schema check, an Arabic-character-ratio language check, a regex PII check (email, 14-digit national-ID-shaped numbers), and Jaccard-shingle near-duplicate detection against both the existing dataset and earlier candidates in the same batch — before any of it was trusted into training. 59 of 839 candidates failed the schema check; 0 failed on length, language, PII, or near-duplication; 780 were kept, bringing `analysis` to 1,005 unique training rows with zero duplicates anywhere in the set.

## Data

- Two sources, merged in `01-eda.ipynb`: `fr3on/eg-legal-instruction-following` (Egyptian legal, `analysis`/`simplification` task types kept) and `mbayan/Arabic-LJP` (Saudi legal judgment prediction).
- Raw merge: 5,259 rows across both sources; after exact-duplicate removal, 5,259 rows remained duplicate-free post-merge (all duplicates were within-source, not cross-source); after the 10–2048 word length filter, 4,814 rows; after capping `judgment_prediction` to match the next-largest class, 2,158 rows going into the split.
- Split: 90/10 stratified by task type (`random_state=42`) → 1,942 train rows / 216 validation rows.
- v1 training set: the 1,942 base rows with `analysis` (225 rows) 3x-duplicated in place, giving 2,392 total training rows (859 simplification / 858 judgment_prediction / 675 analysis).
- v2 training set: the same 225 real `analysis` rows plus 780 curated synthetic ones (1,005 distinct `analysis` rows total, no duplication), giving 2,722 total training rows (859 / 858 / 1,005).
- Evaluation sample: 150 rows, stratified by task type, drawn once from the 216-row validation split and reused identically (byte-for-byte, saved to `data/val_eval_sample.parquet`) across the base, v1, and v2 evaluation runs.

## Results

LLM-judged (Gemini) on the same fixed 150-row sample, 1–10 scale:

| Metric | Base (zero-shot) | Fine-tuned v1 | Fine-tuned v2 |
|---|---|---|---|
| Faithfulness | 4.72 | 7.63 | **8.47** |
| Relevance | 7.15 | 8.77 | **9.44** |
| Fluency | 8.94 | 9.11 | **9.88** |

By task type (faithfulness), from `outputs/reports/finetuned_metrics.json` and `finetuned_metrics_v2.json`:

| Task | Base | v1 | v2 |
|---|---|---|---|
| Analysis | 3.06 | 6.47 | **9.24** |
| Judgment prediction | 4.37 | 5.84 | **7.03** |
| Simplification | 5.50 | **9.76** | 9.74 |

`analysis` was the specific target of the v2 change and moved the most: +2.76 over v1, +6.18 over base. `judgment_prediction` improved too, as a side effect of the same round-1 fixes (LoRA scope, evaluation fixes) rather than anything round 2 touched directly. `simplification` — untouched by either round's changes — stayed flat between v1 and v2 (9.76 → 9.74, within noise), which is a useful negative control: it shows the `analysis` gain is a real, localized effect of the synthetic data, not a general evaluation drift between runs.

Synthetic data generation itself: 839 candidates generated with the few-shot strategy, 59 dropped on schema validation, 0 dropped on length/language/PII, 0 near-duplicates found against 1,942 existing rows or against each other, 780 kept (92.9% yield past the schema gate).

## What I'd do differently / limitations

- **The v2 gain on `judgment_prediction` (+1.19 over v1) has no dedicated fix behind it**, unlike `analysis`. It's plausible this is just run-to-run variance in a 150-row LLM-judged sample rather than a real effect, and the project doesn't have a second seed or a larger sample to distinguish the two.
- **The evaluation is entirely LLM-as-judge, with no human-labeled agreement check.** Gemini scoring another model's Arabic legal text 1–10 on faithfulness is a reasonable proxy, but there's no measurement of how well Gemini's scores track a human legal reader's judgment on even a small subset — a systematic bias in the judge (e.g. rewarding a specific phrasing style regardless of legal correctness) wouldn't be caught by this setup.
- **The synthetic-data strategy benchmark's faithfulness scores come from a 10-of-20 spot check per strategy**, not the full 20-example sample — a reasonable cost-saving call at free-tier API limits, but it means the strategy comparison table above rests on 10 judged examples per strategy, not more.
- **The `judgment_prediction` rebalancing (3,610 → 954 rows) is a blunt cap, not a diversity-aware sample.** It fixes the raw count imbalance but doesn't check whether the retained 954 rows still cluster around the same recurring judge names and boilerplate language that motivated the cap in the first place.
- **No held-out legal-expert review of any generated ruling or analysis.** Every faithfulness number in this README is Gemini's opinion of faithfulness; for a legal-domain model specifically, that's a meaningfully different claim than "a lawyer checked this."
- **The near-duplicate check (`find_near_duplicates`) is O(n × m) Jaccard shingling** — noted in the code itself as fine for a few hundred candidates against a few thousand existing rows, but not something that would scale to a much larger synthetic-generation run without a real similarity index (MinHash/LSH).
- **Only one embedding-free schema/PII/language check was tried for curation.** No ablation on whether a stricter or looser dedup threshold than 0.7 Jaccard similarity would have kept or dropped meaningfully different rows.

## Stack

- `transformers` 4.40.0, `peft` 0.10.0 (QLoRA), `trl` 0.8.6 (`SFTTrainer`), `bitsandbytes` 0.43.1 (4-bit quantization), `accelerate` 0.29.3, `torch` 2.2.0
- Base model: `ALLaM-AI/ALLaM-7B-Instruct-preview` (Apache 2.0, Arabic-native tokenizer)
- `google-genai` (`gemini-3.1-flash-lite`) for both the LLM-judge evaluation and synthetic `analysis` example generation
- `pandas`, `pyarrow` for parquet-based data/results storage; `ydata-profiling` for the EDA HTML report
- `scikit-learn` (`train_test_split`, stratified) for the train/val split
- `pydantic` for structured-output validation in the synthetic-data strategy benchmark
- `dvc` + `dvc-gdrive` referenced in requirements for data versioning
- `pytest` 8.2.0 (test suite present but only scaffolding — `tests/__init__.py` — no test cases committed yet)
- Runs on a single free-tier T4 (16GB), developed on Kaggle, Colab-compatible; adapter weights hosted on Hugging Face Hub (`hossam3759180/allam-qlora-legal-adapter`), not GitHub
